"""Scenario runner for the web page: plain values in, JSON-able dict (plus a base64 PNG) out.

Thin orchestration over `spray_model.predict_contact` + `pa_view.evaluate_pa` / `plot_pa`; the model code
stays where it is. Load the model bundles once (`get_models`) and call `run_scenario` per request.

    spec = {"batter": 592450, "p_throws": "R", "balls": 1, "strikes": 2, "outs": 1,
            "runners": {"1B": True}, "park": "NYY", "pitcher": None,
            "fielders": {"3B": 656305, "SS": 669477, "2B": None, "1B": None}}   # None = league average
    res = run_scenario(spec, get_models())
"""
import base64
import contextlib
import io
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # server use: no GUI backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import out_model  # noqa: E402
import pa_view  # noqa: E402
import spray_model  # noqa: E402

SLOTS = ("3B", "SS", "2B", "1B")
NUM_OF = {v: k for k, v in pa_view.SLOT_OF.items()}
DEFAULT_SPRAY = (2026,)
DEFAULT_OAA = (2026,)
MIN_PITCHER_BIP = 30  # same threshold predict_contact uses before trusting a pitcher's own pitch mix
MAX_N, MAX_SEEDS = 20000, 5

DEFAULTS = {"batter": None, "pitcher": None, "stand": None, "p_throws": None, "balls": 0, "strikes": 0,
            "outs": 0, "runners": {}, "park": None, "fielders": {}, "pitch_type": None, "n": 6000,
            "seed": 0, "n_seeds": 1, "directional": True}

_models = {}


class Models:
    """Loaded bundles plus the lookups the page needs (who has history, which parks exist)."""

    def __init__(self, spray_seasons, oaa_seasons):
        self.spray = spray_model.load_bundle(list(spray_seasons))
        self.out = out_model.load_bundle(list(oaa_seasons))
        self.batters = set(self.spray["train"]["batter"].unique())
        pool = self.spray["pool"]
        self.pitcher_bip = pool["pitcher"].value_counts().to_dict() if "pitcher" in pool else {}
        self.parks = sorted(self.spray["categories"]["home_team"])
        self.pitch_types = sorted(self.spray["categories"]["pitch_type"])
        self.fielders = set(self.out["fielders"].index)


def get_models(spray=DEFAULT_SPRAY, oaa=DEFAULT_OAA):
    key = (tuple(spray), tuple(oaa))
    if key not in _models:
        _models[key] = Models(*key)
    return _models[key]


def _int_or_none(v):
    return None if v in (None, "", "-") else int(v)


def normalize(spec, models):
    """Fill defaults, coerce types and validate. Raises ValueError listing every problem found."""
    s = {**DEFAULTS, **{k: v for k, v in spec.items() if v is not None or k in ("pitcher", "park", "stand")}}
    errs = []
    try:
        s["batter"] = _int_or_none(s["batter"])
        s["pitcher"] = _int_or_none(s["pitcher"])
    except (TypeError, ValueError):
        errs.append("batter/pitcher must be MLBAM ids")
    if s["batter"] is None:
        errs.append("batter is required")
    if s["p_throws"] not in ("L", "R"):
        errs.append("p_throws must be 'L' or 'R'")
    if s["stand"] in ("", "S"):
        s["stand"] = None
    if s["stand"] not in (None, "L", "R"):
        errs.append("stand must be 'L' or 'R'")
    for k, hi in (("balls", 3), ("strikes", 2), ("outs", 2)):
        try:
            s[k] = int(s[k])
        except (TypeError, ValueError):
            s[k] = -1
        if not 0 <= s[k] <= hi:
            errs.append(f"{k} must be 0-{hi}")
    s["runners"] = {b: bool(s["runners"].get(b)) for b in ("1B", "2B", "3B")}
    s["park"] = None if s["park"] in (None, "", "-") else str(s["park"]).upper()
    if s["park"] and s["park"] not in models.parks:
        errs.append(f"unknown park {s['park']!r}")
    s["pitch_type"] = s["pitch_type"] or None
    if s["pitch_type"] and s["pitch_type"] not in models.pitch_types:
        errs.append(f"unknown pitch type {s['pitch_type']!r}")
    try:
        s["fielders"] = {slot: _int_or_none(s["fielders"].get(slot)) for slot in SLOTS}
    except (TypeError, ValueError):
        errs.append("fielders must be MLBAM ids")
        s["fielders"] = {slot: None for slot in SLOTS}
    chosen = [v for v in s["fielders"].values() if v is not None]
    if len(chosen) != len(set(chosen)):
        errs.append("the same player is in more than one infield slot")
    s["n"] = int(s["n"])
    s["n_seeds"] = int(s["n_seeds"])
    s["seed"] = int(s["seed"])
    if not 500 <= s["n"] <= MAX_N:
        errs.append(f"n must be 500-{MAX_N}")
    if not 1 <= s["n_seeds"] <= MAX_SEEDS:
        errs.append(f"n_seeds must be 1-{MAX_SEEDS}")
    s["directional"] = bool(s["directional"])
    if errs:
        raise ValueError("; ".join(errs))
    return s


def _py(o):
    """Recursively turn numpy scalars/arrays into plain Python so the result is json.dumps-able."""
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return _py(o.tolist())
    return o


def _combine(results):
    """Merge per-seed results: pool the sims, average coverages and spots, keep the spot spread."""
    first = results[0]
    if len(results) == 1:
        return first, {slot: (0.0, 0.0) for slot in SLOTS}
    out = dict(first)
    out["sims"] = pd.concat([r["sims"] for r in results], ignore_index=True)
    sd = {}
    for key in ("pos_real", "pos_avg"):
        out[key] = {}
        for slot in SLOTS:
            pts = np.array([r[key][slot] for r in results], float)
            out[key][slot] = tuple(pts.mean(axis=0))
            if key == "pos_real":
                sd[slot] = tuple(pts.std(axis=0))
    for key in ("cov_real", "cov_avg", "cov_avg_pos_real_ranges"):
        out[key] = float(np.mean([r[key] for r in results]))
    return out, sd


def run_scenario(spec, models=None, plot=True):
    """Simulate the situation, optimize the four infielders, and return a JSON-able dict.

    n_seeds > 1 repeats the simulate+optimize step with consecutive seeds and averages the spots (single
    runs jitter ~25 ft in 3B x). Cost is ~9 s per seed.
    """
    models = models or get_models()
    s = normalize(spec, models)
    t0 = time.time()
    pa = pa_view.hypothetical_pa(s["batter"], s["p_throws"], s["stand"], s["balls"], s["strikes"], s["outs"],
                                 s["runners"], s["park"])
    swaps = {NUM_OF[slot]: pid for slot, pid in s["fielders"].items() if pid is not None}
    notes, results = [], []
    for i in range(s["n_seeds"]):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):  # predict_contact prints notes about fallbacks
            results.append(pa_view.evaluate_pa(pa, models.spray, models.out, swaps, s["n"], s["seed"] + i,
                                               s["directional"], s["pitcher"], s["pitch_type"]))
        if i == 0:
            notes = [ln.strip() for ln in buf.getvalue().splitlines() if ln.strip()]
    res, spread = _combine(results)

    sims = res["sims"]
    lineup = []
    for row in res["lineup"].to_dict("records"):
        chosen = s["fielders"][row["slot"]] is not None
        last, sep, first = row["name"].partition(", ")  # Savant gives "Last, First"
        row.update(id=s["fielders"][row["slot"]],
                   name=f"{first} {last}" if sep else (row["name"] if chosen else "League average"),
                   has_history=row["n_plays"] > 0 or not chosen)  # a blank slot is not "missing history"
        lineup.append(row)
    spots = {slot: {"real": res["pos_real"][slot], "league_avg": res["pos_avg"][slot],
                    "real_sd": spread[slot]} for slot in SLOTS}
    pit_n = int(models.pitcher_bip.get(s["pitcher"], 0)) if s["pitcher"] is not None else None
    image = None
    if plot:
        buf = io.BytesIO()
        fig = pa_view.plot_pa(pa, res, buf)
        plt.close(fig)
        image = base64.b64encode(buf.getvalue()).decode("ascii")
    return _py({
        "inputs": s,
        "lineup": lineup,
        "spots": spots,
        "coverage": {"real": res["cov_real"], "league_avg": res["cov_avg"],
                     "league_avg_spots_real_ranges": res["cov_avg_pos_real_ranges"],
                     "tailoring_gain": res["cov_real"] - res["cov_avg_pos_real_ranges"]},
        "bb_mix": sims["bb_type"].value_counts(normalize=True).to_dict(),
        "flags": {"batter_has_history": s["batter"] in models.batters,
                  "pitcher_bip": pit_n,
                  "pitcher_has_history": None if pit_n is None else pit_n >= MIN_PITCHER_BIP},
        "notes": notes,
        "plot_png_b64": image,
        "seconds": round(time.time() - t0, 1),
    })
