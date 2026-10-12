r"""Predict where a hitter hits the ball in a plate-appearance pitch.

Targets (batted balls only, i.e. conditional on contact in play):
  * pull_angle   - degrees; + = pulled side regardless of batter handedness
  * direction    - pull / center / oppo class from pull_angle
  * bb_type      - ground_ball / line_drive / fly_ball / popup

Features are everything knowable before contact: batter + pitcher handedness,
count, pitch type/location/velocity, base-out state, score, park, plus
empirical-Bayes-shrunk batter tendencies (overall and vs pitcher hand) computed
out-of-fold on training rows so the model can't memorize its own labels.

Run:  .venv\Scripts\python.exe Defense\PositionalValue\spray_model.py train 2023
      .venv\Scripts\python.exe Defense\PositionalValue\spray_model.py validate 2023 2024
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


def _clf():
    return HistGradientBoostingClassifier(categorical_features="from_dtype", max_iter=300, learning_rate=0.05,
                                          random_state=0)


def fit(train):
    """Fit all three models on `train` (which must already have batter features)."""
    feats = NUM + CAT
    reg = HistGradientBoostingRegressor(categorical_features="from_dtype", max_iter=300, learning_rate=0.05,
                                        random_state=0)
    reg.fit(train[feats], train["pull_angle"])
    models = {"angle": reg}
    for target in ("direction", "bb_type"):
        models[target] = _clf().fit(train[feats], train[target])
    return models


def evaluate(models, train, test):
    """Score fitted models on `test` against simple baselines (both frames need batter features)."""
    feats = NUM + CAT
    results = {}
    pred = models["angle"].predict(test[feats])
    base = np.full(len(test), train["pull_angle"].mean())
    hand = test[["stand", "p_throws"]].merge(
        train.groupby(["stand", "p_throws"])["pull_angle"].mean().rename("m"),
        left_on=["stand", "p_throws"], right_index=True, how="left")["m"].to_numpy()
    results["pull_angle MAE (deg)"] = {"model": mean_absolute_error(test["pull_angle"], pred),
                                       "league mean": mean_absolute_error(test["pull_angle"], base),
                                       "stand x p_throws mean": mean_absolute_error(test["pull_angle"], hand)}
    for target in ("direction", "bb_type"):
        clf = models[target]
        p = clf.predict_proba(test[feats])
        prior = train[target].value_counts(normalize=True).reindex(clf.classes_).to_numpy()
        results[f"{target} log-loss"] = {
            "model": log_loss(test[target], p, labels=clf.classes_),
            "league prior": log_loss(test[target], np.tile(prior, (len(test), 1)), labels=clf.classes_)}
    return results


def _print(results):
    for metric, vals in results.items():
        print(" ", metric, {k: round(v, 4) for k, v in vals.items()})


MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"


def cmd_train(seasons):
    """Fit on `seasons`, print a within-season sanity check, and save the model bundle."""
    data = load_batted_balls(seasons)
    print(f"BIP: {len(data):,}")
    # Sanity holdout: last ~15% of the calendar (time-ordered), fit on the rest.
    cut = data["game_date"].sort_values().iloc[int(len(data) * 0.85)]
    early, late = data[data["game_date"] < cut], data[data["game_date"] >= cut]
    tr, te = add_batter_features(early, late)
    print(f"within-season holdout: train {len(tr):,} (< {cut})  test {len(te):,}")
    _print(evaluate(fit(tr), tr, te))
    # Final model on everything. Save the training frame subset needed to rebuild batter features.
    full, _ = add_batter_features(data, data.head(0))
    models = fit(full)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / f"spray_{'_'.join(map(str, seasons))}.joblib"
    cats = {c: list(data[c].cat.categories) for c in CAT}
    joblib.dump({"models": models, "categories": cats, "seasons": seasons,
                 "train": data[["batter", "p_throws", "stand", "pull_angle", "is_gb", "direction", "bb_type"]]}, out)
    print("saved", out)


def cmd_validate(train_seasons, test_season):
    bundle = joblib.load(MODEL_DIR / f"spray_{'_'.join(map(str, train_seasons))}.joblib")
    test = load_batted_balls([test_season])
    for c, cats in bundle["categories"].items():  # align category codes with training
        test[c] = pd.Categorical(test[c].astype(object), categories=cats)
    train, test = add_batter_features(bundle["train"], test)
    print(f"validate on {test_season}: test BIP {len(test):,} (batters unseen in training get league mean)")
    _print(evaluate(bundle["models"], train, test))


if __name__ == "__main__":
    usage = "usage: spray_model.py train 2023 [2022 ...] | validate <train_seasons,comma> <test_season>"
    if len(sys.argv) < 3:
        sys.exit(usage)
    if sys.argv[1] == "train":
        cmd_train([int(s) for s in sys.argv[2:]])
    elif sys.argv[1] == "validate" and len(sys.argv) == 4:
        cmd_validate([int(s) for s in sys.argv[2].split(",")], int(sys.argv[3]))
    else:
        sys.exit(usage)
