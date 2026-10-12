r"""Predict where a hitter hits the ball in a plate-appearance pitch.

Targets (batted balls only, i.e. conditional on contact in play):
  * pull_angle   - degrees; + = pulled side regardless of batter handedness
  * direction    - pull / center / oppo class from pull_angle
  * bb_type      - ground_ball / line_drive / fly_ball / popup
  * distance_ft  - distance from home plate to where the ball was first touched / landed

Features are everything knowable before contact: batter + pitcher handedness,
count, pitch type/location/velocity, base-out state, score, park, plus
empirical-Bayes-shrunk batter tendencies (overall and vs pitcher hand) computed
out-of-fold on training rows so the model can't memorize its own labels.

Run:  .venv\Scripts\python.exe Defense\PositionalValue\spray_model.py train 2023
      .venv\Scripts\python.exe Defense\PositionalValue\spray_model.py validate 2023 2024
      .venv\Scripts\python.exe Defense\PositionalValue\spray_model.py predict --batter 592450 --p-throws R --outs 1 --on1b
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from baseball_data import statcast_pitches, add_spray  # noqa: E402
from positioning import optimal_infield_positions, compare_to_actual, reach_for, drop_battery, FIXED, DEFAULT_REACH_FT  # noqa: E402

SHRINK_K = 30  # pseudo-BIP of league-average behavior mixed into each batter estimate
PULL_DEG = 15  # |pull_angle| thresholds for the pull/center/oppo classes
BB_TYPES = ["ground_ball", "line_drive", "fly_ball", "popup"]

CAT = ["pitch_type", "home_team"]
NUM = ["is_rhb", "is_rhp", "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b", "inning",
       "score_diff", "release_speed", "pfx_x", "pfx_z", "plate_x_in", "plate_z",
       "bat_pull_mean", "bat_pull_vs_hand_mean", "bat_gb_rate", "bat_n"]


def load_batted_balls(seasons):
    frames = [statcast_pitches(f"{y}-03-20", f"{y}-10-05") for y in seasons]
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["type"] == "X") & df["hc_x"].notna() & df["bb_type"].isin(BB_TYPES)].copy()
    df = add_spray(df)
    df = df[df["spray_angle"].abs() <= 50]  # drop coordinate noise (bunts at the plate, etc.)
    df["is_rhb"] = (df["stand"] == "R").astype(int)
    df["is_rhp"] = (df["p_throws"] == "R").astype(int)
    for b in ("on_1b", "on_2b", "on_3b"):
        df[b] = df[b].notna().astype(int)
    df["score_diff"] = df["bat_score"] - df["fld_score"]
    df["plate_x_in"] = df["plate_x"] * np.where(df["is_rhb"] == 1, 1, -1)  # + = away from batter for both L/R
    df["is_gb"] = (df["bb_type"] == "ground_ball").astype(int)
    df["direction"] = np.select([df["pull_angle"] > PULL_DEG, df["pull_angle"] < -PULL_DEG], ["pull", "oppo"], "center")
    for c in CAT:
        df[c] = df[c].astype("category")
    return df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


N_FOLDS = 5


def _shrunk(train, test, keys, col, k=SHRINK_K):
    """Empirical-Bayes batter mean of `col` grouped by `keys` (+ count).

    Train rows use out-of-fold aggregates (the row's own fold is excluded). Plain
    leave-one-out leaks: within a batter the LOO mean is a decreasing function of the
    row's own label, which trees decode. Test rows use the full training aggregate.
    """
    mu = train[col].mean()
    g = train.groupby(keys)[col].agg(["sum", "count"])
    fold = np.random.default_rng(0).integers(0, N_FOLDS, len(train))
    tmp = train[keys].copy()
    tmp["_v"], tmp["_f"] = train[col].to_numpy(), fold
    gf = tmp.groupby(keys + ["_f"])["_v"].agg(["sum", "count"]).rename(columns={"sum": "fs", "count": "fc"})
    tr = tmp.merge(g, left_on=keys, right_index=True, how="left").merge(
        gf, left_on=keys + ["_f"], right_index=True, how="left")
    n_tr = (tr["count"] - tr["fc"]).to_numpy()
    est_tr = ((tr["sum"] - tr["fs"]).to_numpy() + k * mu) / (n_tr + k)
    te = test[keys].merge(g, left_on=keys, right_index=True, how="left").fillna({"sum": 0, "count": 0})
    est_te = (te["sum"].to_numpy() + k * mu) / (te["count"].to_numpy() + k)
    return est_tr, n_tr, est_te, te["count"].to_numpy()


def add_batter_features(train, test):
    train, test = train.copy(), test.copy()
    tr, n_tr, te, n_te = _shrunk(train, test, ["batter"], "pull_angle")
    train["bat_pull_mean"], train["bat_n"], test["bat_pull_mean"], test["bat_n"] = tr, n_tr, te, n_te
    tr, _, te, _ = _shrunk(train, test, ["batter", "p_throws"], "pull_angle")
    train["bat_pull_vs_hand_mean"], test["bat_pull_vs_hand_mean"] = tr, te
    tr, _, te, _ = _shrunk(train, test, ["batter"], "is_gb")
    train["bat_gb_rate"], test["bat_gb_rate"] = tr, te
    return train, test


FEATS = NUM + CAT
POOL_COLS = ["stand", "p_throws", "pitch_type", "release_speed", "pfx_x", "pfx_z", "plate_x_in", "plate_z",
             "home_team"]
MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"


def _reg():
    return HistGradientBoostingRegressor(categorical_features="from_dtype", max_iter=300, learning_rate=0.05,
                                         random_state=0)


def _clf():
    return HistGradientBoostingClassifier(categorical_features="from_dtype", max_iter=300, learning_rate=0.05,
                                          random_state=0)


def fit(train):
    """Fit all models on `train` (which must already have batter features).

    models["bb_type"], ["direction"]: classifiers.  models["angle"], ["distance"]: overall regressors.
    models["by_type"][t] = {"angle", "distance"} regressors fit on batted balls of type t; these drive
    the contact simulator (pick a type, then place the ball).
    """
    models = {"angle": _reg().fit(train[FEATS], train["pull_angle"]),
              "distance": _reg().fit(train[FEATS], train["distance_ft"]),
              "bb_type": _clf().fit(train[FEATS], train["bb_type"]),
              "direction": _clf().fit(train[FEATS], train["direction"]),
              "by_type": {}}
    for t in BB_TYPES:
        sub = train[train["bb_type"] == t]
        models["by_type"][t] = {"angle": _reg().fit(sub[FEATS], sub["pull_angle"]),
                                "distance": _reg().fit(sub[FEATS], sub["distance_ft"])}
    return models


def evaluate(models, train, test):
    """Score fitted models on `test` against simple baselines (both frames need batter features)."""
    results = {}
    hand = test[["stand", "p_throws"]].merge(
        train.groupby(["stand", "p_throws"])["pull_angle"].mean().rename("m"),
        left_on=["stand", "p_throws"], right_index=True, how="left")["m"].to_numpy()
    results["pull_angle MAE (deg)"] = {
        "model": mean_absolute_error(test["pull_angle"], models["angle"].predict(test[FEATS])),
        "league mean": mean_absolute_error(test["pull_angle"], np.full(len(test), train["pull_angle"].mean())),
        "stand x p_throws mean": mean_absolute_error(test["pull_angle"], hand)}
    results["distance MAE (ft)"] = {
        "model": mean_absolute_error(test["distance_ft"], models["distance"].predict(test[FEATS])),
        "league mean": mean_absolute_error(test["distance_ft"], np.full(len(test), train["distance_ft"].mean()))}
    for target in ("direction", "bb_type"):
        clf = models[target]
        p = clf.predict_proba(test[FEATS])
        prior = train[target].value_counts(normalize=True).reindex(clf.classes_).to_numpy()
        results[f"{target} log-loss"] = {
            "model": log_loss(test[target], p, labels=clf.classes_),
            "league prior": log_loss(test[target], np.tile(prior, (len(test), 1)), labels=clf.classes_)}
    return results


def holdout_residuals(models, test):
    """Out-of-sample (angle, distance) residuals per batted-ball type; they give simulated balls realistic spread."""
    res = {}
    for t in BB_TYPES:
        sub = test[test["bb_type"] == t]
        m = models["by_type"][t]
        res[t] = np.column_stack([sub["pull_angle"] - m["angle"].predict(sub[FEATS]),
                                  sub["distance_ft"] - m["distance"].predict(sub[FEATS])])
    return res


def _print(results):
    for metric, vals in results.items():
        print(" ", metric, {k: round(v, 4) for k, v in vals.items()})


def cmd_train(seasons):
    """Fit on `seasons`, print a within-season sanity check, and save the model bundle."""
    data = load_batted_balls(seasons)
    print(f"BIP: {len(data):,}")
    # Sanity holdout: last ~15% of the calendar (time-ordered), fit on the rest.
    cut = data["game_date"].sort_values().iloc[int(len(data) * 0.85)]
    early, late = data[data["game_date"] < cut], data[data["game_date"] >= cut]
    tr, te = add_batter_features(early, late)
    print(f"within-season holdout: train {len(tr):,} (< {cut})  test {len(te):,}")
    holdout_models = fit(tr)
    _print(evaluate(holdout_models, tr, te))
    residuals = holdout_residuals(holdout_models, te)

    # Final model on everything.
    full, _ = add_batter_features(data, data.head(0))
    models = fit(full)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / f"spray_{'_'.join(map(str, seasons))}.joblib"
    joblib.dump({
        "models": models, "residuals": residuals, "seasons": seasons,
        "categories": {c: list(data[c].cat.categories) for c in CAT},
        "train": data[["batter", "p_throws", "stand", "pull_angle", "is_gb", "direction", "bb_type",
                       "distance_ft"]],
        "pool": data[POOL_COLS].sample(min(30000, len(data)), random_state=0).reset_index(drop=True),
    }, out)
    print("saved", out)


def load_bundle(seasons):
    return joblib.load(MODEL_DIR / f"spray_{'_'.join(map(str, seasons))}.joblib")


def cmd_validate(train_seasons, test_season):
    bundle = load_bundle(train_seasons)
    test = load_batted_balls([test_season])
    for c, cats in bundle["categories"].items():  # align category codes with training
        test[c] = pd.Categorical(test[c].astype(object), categories=cats)
    train, test = add_batter_features(bundle["train"], test)
    print(f"validate on {test_season}: test BIP {len(test):,} (batters unseen in training get league mean)")
    _print(evaluate(bundle["models"], train, test))


def predict_contact(bundle, batter, p_throws, balls=0, strikes=0, outs=0, on_1b=0, on_2b=0, on_3b=0,
                    inning=5, score_diff=0, park=None, stand=None, pitch_type=None, n=3000, seed=0):
    """Simulate `n` batted balls for one batter in one situation.

    Unspecified pitch details (type/speed/movement/location) are marginalized by drawing real pitches
    thrown by `p_throws` to that batter's side from the training pool. Returns a DataFrame with
    bb_type, pull_angle, spray_angle (deg, 0 = CF, + = RF side), distance_ft, x_ft, y_ft (home = 0,0).
    """
    rng = np.random.default_rng(seed)
    train, models = bundle["train"], bundle["models"]
    if stand is None:
        sides = train.loc[train["batter"] == batter, "stand"].unique()
        if len(sides) == 1:
            stand = sides[0]
        elif len(sides) > 1:  # switch hitter: bats from the side opposite the pitcher's hand
            stand = "L" if p_throws == "R" else "R"
        else:
            raise ValueError(f"batter {batter} not in training data; pass stand='L' or 'R'")
    pool = bundle["pool"]
    pool = pool[(pool["stand"] == stand) & (pool["p_throws"] == p_throws)]
    if pitch_type:
        pool = pool[pool["pitch_type"] == pitch_type]
    if pool.empty:
        raise ValueError("no pitches in the pool match those filters")
    X = pool.sample(n, replace=True, random_state=seed).reset_index(drop=True)
    if park:
        X["home_team"] = park
    X["batter"] = batter
    X["is_rhb"], X["is_rhp"] = int(stand == "R"), int(p_throws == "R")
    X["balls"], X["strikes"], X["outs_when_up"] = balls, strikes, outs
    X["on_1b"], X["on_2b"], X["on_3b"] = on_1b, on_2b, on_3b
    X["inning"], X["score_diff"] = inning, score_diff
    for c, cats in bundle["categories"].items():
        X[c] = pd.Categorical(X[c].astype(object), categories=cats)
    stub = train[["batter", "p_throws", "pull_angle", "is_gb"]]
    _, X = add_batter_features(stub, X)

    clf = models["bb_type"]
    cum = clf.predict_proba(X[FEATS]).cumsum(axis=1)
    types = clf.classes_[(rng.random(n)[:, None] > cum).sum(axis=1).clip(0, len(clf.classes_) - 1)]
    pull = np.empty(n)
    dist = np.empty(n)
    for t in BB_TYPES:
        idx = np.flatnonzero(types == t)
        if not len(idx):
            continue
        m = models["by_type"][t]
        r = bundle["residuals"][t][rng.integers(0, len(bundle["residuals"][t]), len(idx))]
        pull[idx] = m["angle"].predict(X.iloc[idx][FEATS]) + r[:, 0]
        dist[idx] = np.maximum(m["distance"].predict(X.iloc[idx][FEATS]) + r[:, 1], 5)
    spray = -pull if stand == "R" else pull  # righties pull toward LF (negative spray angle)
    ang = np.radians(spray)
    return pd.DataFrame({"bb_type": types, "pull_angle": pull, "spray_angle": spray, "distance_ft": dist,
                         "x_ft": dist * np.sin(ang), "y_ft": dist * np.cos(ang)})


def _density(sims, xlim, ylim, bin_ft=4, sigma_ft=9):
    """Gaussian-smoothed 2D histogram of landing spots, normalized to its peak (0-1)."""
    from scipy.ndimage import gaussian_filter

    nx, ny = int((xlim[1] - xlim[0]) / bin_ft), int((ylim[1] - ylim[0]) / bin_ft)
    h, _, _ = np.histogram2d(sims["x_ft"], sims["y_ft"], bins=[nx, ny], range=[xlim, ylim])
    h = gaussian_filter(h, sigma_ft / bin_ft)
    return h / h.max() if h.max() > 0 else h


def _heat(ax, sims, xlim, ylim):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("heat", ["#ffffb2", "#fd8d3c", "#e31a1c", "#4a0010"])
    d = _density(sims, xlim, ylim)
    img = np.ma.masked_less(d.T, 0.06)  # leave the field visible where almost nothing lands
    return ax.imshow(img, origin="lower", extent=[*xlim, *ylim], cmap=cmap, vmin=0, vmax=1, alpha=0.8,
                     interpolation="bilinear", zorder=5)


def plot_prediction(bundle, sims, title, path=None, positions=None, actual=None, reach_ft=DEFAULT_REACH_FT):
    """Two heatmaps of simulated landing spots: all contact (left) and ground balls, zoomed to the infield (right)."""
    import matplotlib.pyplot as plt
    from baseball_data.field import plot_field, WALL_FT

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), layout="constrained")
    n_all = len(sims)
    sims = drop_battery(sims)  # grounders the pitcher / catcher fields are not the infielders' to cover
    plot_field(axes[0])
    _heat(axes[0], sims, (-WALL_FT * 0.78, WALL_FT * 0.78), (-25, WALL_FT + 10))
    parts = ", ".join(f"{t.replace('_', ' ')} {(sims['bb_type'] == t).mean():.0%}" for t in BB_TYPES)
    axes[0].set_title(f"All batted balls ({parts})", fontsize=10)

    gb = sims[sims["bb_type"] == "ground_ball"]
    plot_field(axes[1])
    axes[1].set_xlim(-150, 150)
    axes[1].set_ylim(-15, 215)
    im = _heat(axes[1], gb, (-150, 150), (-15, 215))
    axes[1].set_title(f"Ground balls only, infield zoom ({len(gb) / n_all:.0%} of contact; {1 - len(sims) / n_all:.0%} "
        f"of all contact removed as fielded by P/C)", fontsize=10)
    for lab, ((x, y), r) in FIXED.items():
        axes[1].add_patch(plt.Circle((x, y), r, fill=False, ls=":", ec="#444", lw=1.2, zorder=8))
        axes[1].text(x, y, lab, ha="center", va="center", fontsize=8, color="#444", fontweight="bold", zorder=10)
    for lab, (x, y) in (positions or {}).items():
        axes[1].add_patch(plt.Circle((x, y), reach_for(lab, reach_ft), fill=False, ls="--", ec="black", lw=1, zorder=8))
        axes[1].scatter(x, y, s=170, c="white", edgecolor="black", zorder=9)
        axes[1].text(x, y, lab, ha="center", va="center", fontsize=7, fontweight="bold", zorder=10)
    for lab, (x, y) in (actual or {}).items():
        axes[1].scatter(x, y, s=90, marker="s", c="#1f77b4", edgecolor="white", zorder=11)
        axes[1].text(x, y - 9, f"{lab} actual", ha="center", va="top", fontsize=7, color="#1f77b4", zorder=11)
        if positions and lab in positions:
            axes[1].plot([x, positions[lab][0]], [y, positions[lab][1]], c="#1f77b4", lw=1, zorder=10)
    fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, label="relative density of landing spots (1 = hottest)")
    fig.suptitle(title, fontsize=12)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def cmd_predict(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="spray_model.py predict")
    ap.add_argument("--seasons", default="2023", help="comma list identifying the trained bundle")
    ap.add_argument("--batter", type=int, required=True, help="MLBAM player id")
    ap.add_argument("--p-throws", choices=["L", "R"], required=True)
    ap.add_argument("--stand", choices=["L", "R"], help="only needed for batters not in training data")
    ap.add_argument("--balls", type=int, default=0)
    ap.add_argument("--strikes", type=int, default=0)
    ap.add_argument("--outs", type=int, default=0)
    ap.add_argument("--on1b", action="store_true")
    ap.add_argument("--on2b", action="store_true")
    ap.add_argument("--on3b", action="store_true")
    ap.add_argument("--park", help="home team code, e.g. NYY (default: league mix)")
    ap.add_argument("--pitch-type", help="e.g. FF, SL (default: all pitches)")
    ap.add_argument("--reach", type=float, default=DEFAULT_REACH_FT,
                    help="placeholder fielder reach (ft) for the coverage heuristic")
    ap.add_argument("--no-shift-rules", action="store_true",
                    help="drop the 2-per-side / on-the-dirt constraints")
    ap.add_argument("--actual", help='actual positions to compare, e.g. "3B:-60,95;SS:-30,125;2B:30,125;1B:62,88"')
    ap.add_argument("--n", type=int, default=15000, help="simulated batted balls (more = smoother heatmap)")
    ap.add_argument("--out", help="PNG path (default: data/plots/contact_<batter>.png)")
    a = ap.parse_args(argv)
    bundle = load_bundle([int(s) for s in a.seasons.split(",")])
    sims = predict_contact(bundle, a.batter, a.p_throws, a.balls, a.strikes, a.outs,
                           int(a.on1b), int(a.on2b), int(a.on3b), park=a.park, stand=a.stand,
                           pitch_type=a.pitch_type, n=a.n)
    print(sims.groupby("bb_type")[["spray_angle", "distance_ft"]].agg(["mean", "median"]).round(1))
    print("P(type):", sims["bb_type"].value_counts(normalize=True).round(3).to_dict())
    runners = "".join(b for b, f in (("1", a.on1b), ("2", a.on2b), ("3", a.on3b)) if f) or "none"
    out = Path(a.out) if a.out else MODEL_DIR.parent / "plots" / f"contact_{a.batter}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    positions, cov = optimal_infield_positions(sims, a.reach, shift_rules=not a.no_shift_rules)
    print(f"coverage-heuristic positions (reach {a.reach:.0f} ft, expected GB coverage {cov:.1%}):")
    for lab, (x, y) in positions.items():
        print(f"  {lab}: x={x:+.0f} y={y:.0f}")
    actual = None
    if a.actual:
        actual = {k.strip(): tuple(float(v) for v in xy.split(","))
                  for k, xy in (item.split(":") for item in a.actual.split(";"))}
        rows, lost = compare_to_actual(positions, actual, sims, a.reach)
        for lab, r in rows.items():
            print(f"  {lab}: {r['dist_ft']:.0f} ft from optimal (dx {r['dx_ft']:+.0f}, dy {r['dy_ft']:+.0f})")
        print(f"  expected GB coverage lost vs heuristic optimum: {lost:.1%}")
    plot_prediction(bundle, sims, f"Batter {a.batter} vs {a.p_throws}HP | {a.balls}-{a.strikes}, "
                                  f"{a.outs} out, runners: {runners}", out, positions, actual, a.reach)
    print("saved", out)


def cmd_list_pas(argv):
    import argparse
    from baseball_data import savant
    ap = argparse.ArgumentParser(prog="spray_model.py list-pas")
    ap.add_argument("--date", required=True, help="game date, YYYY-MM-DD")
    ap.add_argument("--batter", type=int)
    ap.add_argument("--pitcher", type=int)
    ap.add_argument("--team", help="home or away team code, e.g. NYY")
    ap.add_argument("--show", metavar="GAME_PK:AT_BAT", help="print full detail for one PA, e.g. 745000:42")
    a = ap.parse_args(argv)
    if a.show:
        gp, ab = a.show.split(":")
        pa = savant.get_plate_appearance(int(gp), int(ab), a.date)
        for k, v in pa.items():
            print(f"{k}: {v}")
        return
    pas = savant.find_plate_appearances(a.date, a.batter, a.pitcher, a.team)
    print(pas.to_string() if len(pas) else "no balls in play matched")


if __name__ == "__main__":
    usage = ("usage: spray_model.py train 2023 [2022 ...] | validate <train_seasons,comma> <test_season> "
             "| predict --batter ID --p-throws R [options] | list-pas --date YYYY-MM-DD [--show PK:AB]")
    if len(sys.argv) < 3:
        sys.exit(usage)
    if sys.argv[1] == "train":
        cmd_train([int(s) for s in sys.argv[2:]])
    elif sys.argv[1] == "validate" and len(sys.argv) == 4:
        cmd_validate([int(s) for s in sys.argv[2].split(",")], int(sys.argv[3]))
    elif sys.argv[1] == "predict":
        cmd_predict(sys.argv[2:])
    elif sys.argv[1] == "list-pas":
        cmd_list_pas(sys.argv[2:])
    else:
        sys.exit(usage)
