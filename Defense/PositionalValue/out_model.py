r"""Step 3: learned P(out) for an infielder, with per-fielder range effects.

Data: Savant `oaa-data` plays (responsible-fielder plays, one row per play, positions 3-6 used here).
Features (all knowable for a simulated ground ball): exact distance from the fielder's start to the intercept
point (`intercept_point_distance_for_fielder`), the displacement (dx, dy) and the intercept point itself
(both quantized to 15 ft cells in the source), exit velocity, and position. Plays whose direction group is
"Back" are dropped: they are almost all fly balls / pop-ups (out rate 0.8-0.99) and would pollute
ground-ball probabilities. Success is the realized out (`is_hit_into_play_field_out`).

Fielder effect: p = sigmoid(logit(p_hat) + u_f). p_hat is a position-aware gradient-boosted model with no
player identity; u_f is a one-step ridge-shrunk logit offset from out-of-fold residuals,
u = sum(y - p) / (sum p(1-p) + LAMBDA). Unseen fielders get u = 0 (league average).

Caveats: observed, not counterfactual (positions are confounded with skill; only responsible-fielder plays).
Start/intercept coordinates are coarse, so (dx, dy) carry little beyond the exact distance.

Run:  python Defense/PositionalValue/out_model.py train 2022,2023 [--spray 2023]
      python Defense/PositionalValue/out_model.py validate 2022,2023 2024
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseball_data import oaa_plays_league, oaa_leaderboard  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
ROLE_OF = {"1B": 3, "2B": 4, "3B": 5, "SS": 6}
INFIELD_ROLES = list(ROLE_OF.values())
FEATS = ["dist", "dx", "dy", "ball_x", "ball_y", "ev", "role"]
GROUPS = ["In", "Lateral-1B", "Lateral-3B"]  # Savant direction groups kept ("Back" is dropped)
LAMBDA_DIR = 100.0  # ridge on the per-direction offset, relative to the fielder's overall u
LAMBDA = 30.0   # ridge on the fielder logit offset, in units of Fisher information (tuned in `validate`)
MAX_DIST_FT = 100.0  # beyond this the model was never trained; treat as unreachable
N_FOLDS = 5


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def load_plays(seasons):
    """Infield (roles 3-6) OAA plays with model features and target `y` (realized out)."""
    df = pd.concat([oaa_plays_league(s).assign(season=s) for s in seasons], ignore_index=True)
    df = df[df["intercept_fielder_role_pos"].isin(INFIELD_ROLES)
            & df["intercept_direction_group"].isin(GROUPS)].copy()
    df = df.dropna(subset=["intercept_fielder_starting_position_x", "intercept_fielder_starting_position_y",
                           "intercept_point_x", "intercept_point_y", "intercept_point_distance_for_fielder",
                           "h_launch_speed", "is_hit_into_play_field_out", "target_mlb_id"])
    df["role"] = df["intercept_fielder_role_pos"].astype(int)
    df["dist"] = df["intercept_point_distance_for_fielder"]
    df["dx"] = df["intercept_point_x"] - df["intercept_fielder_starting_position_x"]
    df["dy"] = df["intercept_point_y"] - df["intercept_fielder_starting_position_y"]
    df["ball_x"], df["ball_y"] = df["intercept_point_x"], df["intercept_point_y"]
    df["sx"], df["sy"] = df["intercept_fielder_starting_position_x"], df["intercept_fielder_starting_position_y"]
    df["ev"] = df["h_launch_speed"]
    df["g"] = df["intercept_direction_group"].map({g: i for i, g in enumerate(GROUPS)}).astype(int)
    df["y"] = df["is_hit_into_play_field_out"].astype(int)
    df["fielder"] = df["target_mlb_id"].astype(int)
    return df.reset_index(drop=True)


def _clf():
    # monotone: the farther the ball is from the fielder, the lower the out probability
    return HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, max_leaf_nodes=24, min_samples_leaf=60, l2_regularization=1.0,
        categorical_features=["role"], monotonic_cst={"dist": -1}, random_state=0)


def _oof_prob(df):
    folds = np.random.default_rng(0).integers(0, N_FOLDS, len(df))
    p = np.empty(len(df))
    for k in range(N_FOLDS):
        m = _clf().fit(df.loc[folds != k, FEATS], df.loc[folds != k, "y"])
        p[folds == k] = m.predict_proba(df.loc[folds == k, FEATS])[:, 1]
    return p


def fielder_effects(df, p, lam=LAMBDA):
    """Shrunk logit offset per fielder from residuals y - p."""
    g = pd.DataFrame({"fielder": df["fielder"], "r": df["y"] - p, "w": p * (1 - p)}).groupby("fielder")
    s = g.agg(r=("r", "sum"), w=("w", "sum"), n=("r", "size"))
    return (s["r"] / (s["w"] + lam)).rename("u"), s["n"]


def direction_offsets(df, p, u, lam_dir=LAMBDA_DIR):
    """Per fielder x direction group logit offsets v (relative to u), shrunk toward 0. Returns (v, n) frames."""
    p2 = _sigmoid(_logit(p) + df["fielder"].map(u).fillna(0).to_numpy())
    t = pd.DataFrame({"fielder": df["fielder"], "g": df["g"], "r": df["y"] - p2, "w": p2 * (1 - p2)})
    s = t.groupby(["fielder", "g"]).agg(r=("r", "sum"), w=("w", "sum"), n=("r", "size"))
    v = (s["r"] / (s["w"] + lam_dir)).unstack("g").reindex(columns=range(len(GROUPS))).fillna(0.0)
    n = s["n"].unstack("g").reindex(columns=range(len(GROUPS))).fillna(0).astype(int)
    # Center per group (play-weighted) so offsets are player-specific, not a league-wide group miscalibration.
    for i in range(len(GROUPS)):
        v[i] = v[i] - float((v[i] * n[i]).sum() / max(n[i].sum(), 1))
    return v, n


DIR_FEATS = ["dx", "dy", "dist", "role", "sx", "sy", "ball_x", "ball_y", "ev"]


def fit_dir_clf(df):
    """P(direction group | geometry, EV, role). The group is partly a play-type indicator (lateral plays
    are hard-hit), so geometry alone gets ~0.67 accuracy and geometry + EV + role ~0.79; simulated balls
    use the predicted probabilities, not a hard label."""
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=0)
    return clf.fit(df[DIR_FEATS], df["g"])


def _dir_frame(ball_xy, ev, start_xy, role):
    m, n = len(start_xy), len(ball_xy)
    dx = ball_xy[None, :, 0] - start_xy[:, None, 0]
    dy = ball_xy[None, :, 1] - start_xy[:, None, 1]
    return pd.DataFrame({
        "dx": dx.ravel(), "dy": dy.ravel(), "dist": np.hypot(dx, dy).ravel(), "role": np.full(m * n, role),
        "sx": np.repeat(start_xy[:, 0], n), "sy": np.repeat(start_xy[:, 1], n),
        "ball_x": np.tile(ball_xy[:, 0], m), "ball_y": np.tile(ball_xy[:, 1], m),
        "ev": np.tile(np.asarray(ev, float), m)})[DIR_FEATS]


def fit(df, lam=LAMBDA, lam_dir=LAMBDA_DIR):
    model = _clf().fit(df[FEATS], df["y"])
    p_oof = _oof_prob(df)
    u, n = fielder_effects(df, p_oof, lam)
    names = (df.dropna(subset=["name_fielder"]).drop_duplicates("fielder", keep="last")
               .set_index("fielder")["name_fielder"])
    pos = df.groupby("fielder")["role"].agg(lambda s: int(s.mode().iloc[0]))
    v, vn = direction_offsets(df, p_oof, u, lam_dir)
    table = pd.DataFrame({"u": u, "n": n}).join(names.rename("name")).join(pos.rename("role"))
    for i, g in enumerate(GROUPS):
        table[f"v_{i}"], table[f"n_{i}"] = v[i], vn[i]
    # Range scores in interpretable units: mean change in out prob over the league's own plays
    # (overall = each play at its own direction; per group = plays of that group only).
    base = _logit(p_oof)
    ref = np.random.default_rng(1).choice(len(df), min(20000, len(df)), replace=False)
    b, gref = base[ref], df["g"].to_numpy()[ref]
    d = {"d_out_pct": np.zeros(len(table))}
    for i in range(len(GROUPS)):
        d[f"d_{i}_pct"] = np.zeros(len(table))
    for k, (_, row) in enumerate(table.iterrows()):
        off = row["u"] + np.array([row[f"v_{i}"] for i in range(len(GROUPS))])[gref]
        d["d_out_pct"][k] = 100 * float((_sigmoid(b + off) - _sigmoid(b)).mean())
        for i in range(len(GROUPS)):
            m = gref == i
            d[f"d_{i}_pct"][k] = 100 * float((_sigmoid(b[m] + off[m]) - _sigmoid(b[m])).mean())
    for c, vals in d.items():
        table[c] = vals
    return {"model": model, "fielders": table, "lam": lam, "lam_dir": lam_dir, "dir_clf": fit_dir_clf(df),
            "p_oof": p_oof}


def p_hat(bundle, ball_xy, ev, start_xy, role):
    """Baseline P(out) matrix (n_starts, n_balls); ball_xy (n,2), ev (n,), start_xy (m,2), role int."""
    ball_xy, start_xy = np.asarray(ball_xy, float), np.asarray(start_xy, float).reshape(-1, 2)
    m, n = len(start_xy), len(ball_xy)
    dx = ball_xy[None, :, 0] - start_xy[:, None, 0]
    dy = ball_xy[None, :, 1] - start_xy[:, None, 1]
    dist = np.hypot(dx, dy)
    X = pd.DataFrame({
        "dist": dist.ravel(), "dx": dx.ravel(), "dy": dy.ravel(),
        "ball_x": np.broadcast_to(ball_xy[None, :, 0], (m, n)).ravel(),
        "ball_y": np.broadcast_to(ball_xy[None, :, 1], (m, n)).ravel(),
        "ev": np.broadcast_to(np.asarray(ev, float)[None, :], (m, n)).ravel(),
        "role": np.full(m * n, role)})[FEATS]
    p = bundle["model"].predict_proba(X)[:, 1].reshape(m, n)
    return np.where(dist > MAX_DIST_FT, 0.0, p)


def fielder_u(bundle, fielder_id):
    t = bundle["fielders"]
    return float(t.loc[fielder_id, "u"]) if fielder_id in t.index else 0.0


def fielder_v(bundle, fielder_id):
    """Per-direction logit offsets (In, Lateral-1B, Lateral-3B) relative to u; zeros for unseen fielders."""
    t = bundle["fielders"]
    if fielder_id in t.index:
        return np.array([float(t.loc[fielder_id, f"v_{i}"]) for i in range(len(GROUPS))])
    return np.zeros(len(GROUPS))


def p_out(bundle, ball_xy, ev, start_xy, role, u=0.0, v=None):
    """P(out) matrix (n_starts, n_balls) for a fielder at role (3-6): logit offset u plus, when `v`
    (per-direction offsets from fielder_v) is given, the offset of the direction the ball is from the
    fielder's start (In / toward the 1B line / toward the 3B line)."""
    p = p_hat(bundle, ball_xy, ev, start_xy, role)
    off = u
    if v is not None:
        ball_xy, start_xy = np.asarray(ball_xy, float), np.asarray(start_xy, float).reshape(-1, 2)
        pg = bundle["dir_clf"].predict_proba(_dir_frame(ball_xy, ev, start_xy, role))
        off = u + (pg @ np.asarray(v)).reshape(len(start_xy), len(ball_xy))
    return np.where(p > 0, _sigmoid(_logit(p) + off), 0.0)


# --- exit-velocity sampler for simulated ground balls ---------------------------------------------
def build_ev_model(spray_seasons):
    """Per-batter shrunk mean GB exit velocity plus a pooled residual sample (from Statcast, cached)."""
    from spray_model import load_batted_balls
    d = load_batted_balls(spray_seasons)
    d = d[(d["bb_type"] == "ground_ball") & d["launch_speed"].notna()]
    mu = float(d["launch_speed"].mean())
    g = d.groupby("batter")["launch_speed"].agg(["sum", "count"])
    k = 30
    bm = (g["sum"] + k * mu) / (g["count"] + k)
    resid = (d["launch_speed"] - d["batter"].map(bm)).to_numpy()
    return {"league_mean": mu, "batter_mean": bm.to_dict(),
            "resid": np.random.default_rng(0).choice(resid, min(20000, len(resid)), replace=False)}


def sample_ev(ev_model, batter, n, seed=0):
    rng = np.random.default_rng(seed)
    mu = ev_model["batter_mean"].get(int(batter), ev_model["league_mean"])
    return np.clip(mu + rng.choice(ev_model["resid"], n), 25, 115)


# --- CLI -------------------------------------------------------------------------------------------
def bundle_path(seasons):
    return MODEL_DIR / f"out_{'_'.join(map(str, seasons))}.joblib"


def load_bundle(seasons):
    return joblib.load(bundle_path(seasons))


def cmd_train(seasons, spray_seasons):
    df = load_plays(seasons)
    print(f"plays: {len(df):,}  fielders: {df['fielder'].nunique()}")
    b = fit(df)
    b["ev_model"] = build_ev_model(spray_seasons)
    b["seasons"], b["spray_seasons"] = seasons, spray_seasons
    del b["p_oof"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(b, bundle_path(seasons))
    t = b["fielders"][b["fielders"]["n"] >= 200].sort_values("d_out_pct")
    cols = ["name", "role", "n", "u", "d_out_pct"]
    print("lowest range:\n", t[cols].head(8).round(3).to_string())
    print("highest range:\n", t[cols].tail(8).round(3).to_string())
    print("saved", bundle_path(seasons))


def cmd_validate(train_seasons, test_season):
    tr, te = load_plays(train_seasons), load_plays([test_season])
    model = _clf().fit(tr[FEATS], tr["y"])
    p_tr = _oof_prob(tr)
    ph = model.predict_proba(te[FEATS])[:, 1]
    base_role = tr.groupby("role")["y"].mean()
    pr = te["role"].map(base_role).to_numpy()
    print(f"train {len(tr):,} plays ({train_seasons})  test {len(te):,} plays ({test_season})")
    res = {"role base rate": pr, "p_hat (no fielder)": ph}
    for lam in (10, 30, 100, 300):
        u, _ = fielder_effects(tr, p_tr, lam)
        res[f"p_hat + u_f (lambda={lam})"] = _sigmoid(_logit(ph) + te["fielder"].map(u).fillna(0).to_numpy())
    for k, p in res.items():
        print(f"  {k:32s} log-loss {log_loss(te['y'], p):.4f}  brier {brier_score_loss(te['y'], p):.4f}")
    print("  (mean outcome", round(te["y"].mean(), 3), " mean p_hat", round(ph.mean(), 3), ")")
    # calibration of p_hat
    q = pd.qcut(ph, 8, duplicates="drop")
    print(te.assign(p=ph, q=q).groupby("q", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
            .round(3).to_string())
    # monotone sanity: pushing every ball 10 ft farther away should lower P(out)
    far = te.copy()
    far["dist"] = far["dist"] + 10
    print("  mean dP(out) for +10 ft distance:", round(float(model.predict_proba(far[FEATS])[:, 1].mean() - ph.mean()), 4))
    # does u_f from the training seasons track the test season's published OAA?
    u, n = fielder_effects(tr, p_tr, LAMBDA)
    lb = oaa_leaderboard(test_season, 10).set_index("player_id")
    j = pd.DataFrame({"u": u, "n": n}).join(lb["outs_above_average"], how="inner")
    j = j[j["n"] >= 150]
    print(f"  corr(u_f from {train_seasons}, OAA {test_season}) over {len(j)} fielders:",
          round(float(j["u"].corr(j["outs_above_average"])), 3))

    # --- directional offsets ---
    clf = fit_dir_clf(tr)
    pg = clf.predict_proba(te[DIR_FEATS])
    g_pred = pg.argmax(1)
    print(f"  direction-group classifier accuracy on {test_season}: {(g_pred == te['g']).mean():.3f}")
    ut = te["fielder"].map(u).fillna(0).to_numpy()
    print(f"  {'overall u only':32s} log-loss {log_loss(te['y'], _sigmoid(_logit(ph) + ut)):.4f}")
    for lam_dir in (30, 100, 300, 1000):
        v, vn = direction_offsets(tr, p_tr, u, lam_dir)
        vv = np.column_stack([te["fielder"].map(v[i]).fillna(0).to_numpy() for i in range(len(GROUPS))])
        off_true = vv[np.arange(len(te)), te["g"].to_numpy()]
        off_pred = (pg * vv).sum(1)  # expected offset under the predicted group probabilities
        for tag, off in (("true group", off_true), ("predicted probs", off_pred)):
            print(f"  u + v (lambda_dir={lam_dir}, {tag}):".ljust(34),
                  f"log-loss {log_loss(te['y'], _sigmoid(_logit(ph) + ut + off)):.4f}")
    # directional totals vs the published OAA splits for the test season
    v, vn = direction_offsets(tr, p_tr, u, LAMBDA_DIR)
    cols = {0: "outs_above_average_infront", 1: "outs_above_average_lateral_toward1bline",
            2: "outs_above_average_lateral_toward3bline"}
    for i, c in cols.items():
        jj = pd.DataFrame({"tot": u + v[i], "n": vn[i]}).join(lb[c], how="inner")
        jj = jj[jj["n"] >= 40]
        print(f"  corr(u+v[{GROUPS[i]}], OAA {test_season} {c.split('outs_above_average_')[1]}) "
              f"over {len(jj)} fielders: {jj['tot'].corr(jj[c]):.3f}   (overall-u only: "
              f"{u.reindex(jj.index).corr(jj[c]):.3f})")


if __name__ == "__main__":
    usage = "usage: out_model.py train 2022,2023 [--spray 2023] | validate 2022,2023 2024"
    if len(sys.argv) < 3:
        sys.exit(usage)
    seasons = [int(s) for s in sys.argv[2].split(",")]
    if sys.argv[1] == "train":
        spray = [int(s) for s in sys.argv[sys.argv.index("--spray") + 1].split(",")] if "--spray" in sys.argv else [2023]
        cmd_train(seasons, spray)
    elif sys.argv[1] == "validate" and len(sys.argv) == 4:
        cmd_validate(seasons, int(sys.argv[3]))
    else:
        sys.exit(usage)
