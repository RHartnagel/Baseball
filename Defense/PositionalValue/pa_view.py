r"""Step 4: per-plate-appearance positioning graph with each fielder's own range.

For one ball in play (picked by date + game_pk + at_bat_number, see `spray_model.py list-pas`):
  1. pull the PA: batter, handedness, count, outs, runners, park, and the 8 fielders on the field;
  2. simulate where that batter hits the ball in that situation (spray model, pitch details marginalized);
  3. place 3B/SS/2B/1B to maximize expected ground-ball outs under the learned out model, using each
     fielder's range effect u_f (out_model.py), subject to the same shift rules / role zones as before;
  4. compare with the same optimization for league-average fielders.

Run:
  .venv\Scripts\python.exe Defense\PositionalValue\pa_view.py --date 2024-06-01 --game 745332 --ab 43
      [--show-actual] [--fielders 6:<id>,4:<id>] [--spray 2023] [--oaa 2022,2023] [--n 6000]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from baseball_data import get_plate_appearance, add_spray  # noqa: E402
import out_model  # noqa: E402
import spray_model  # noqa: E402
from positioning import optimal_infield_positions, coverage_score, drop_battery, FIXED, ZONES  # noqa: E402

SLOT_OF = {3: "1B", 4: "2B", 5: "3B", 6: "SS"}  # Statcast fielder number -> slot label
MAX_PTS = 3000  # simulated grounders used by the optimizer (each candidate spot is scored on all of them)


def make_kernel(out_bundle, us, vs=None):
    """Kernel for positioning.optimal_infield_positions: P(out) of slot's fielder from a candidate spot.

    us: slot -> overall logit effect; vs: slot -> per-direction offsets (In, toward 1B, toward 3B) or None.
    """
    def kernel(slot, cand, pts):
        return out_model.p_out(out_bundle, pts[:, :2], pts[:, 2], cand, out_model.ROLE_OF[slot],
                               us.get(slot, 0.0), (vs or {}).get(slot))
    return kernel


def pa_inputs(pa, swaps=None):
    """Fielder ids by slot (with optional {3..6: id} swaps), the four infielders only."""
    ids = {s: pa["fielders"][s] for s in SLOT_OF.values()}
    for num, pid in (swaps or {}).items():
        ids[SLOT_OF[num]] = pid
    return ids


def describe(out_bundle, ids, us):
    t = out_bundle["fielders"]
    rows = []
    for slot in ("3B", "SS", "2B", "1B"):
        pid = ids[slot]
        if pid in t.index:
            r = t.loc[pid]
            rows.append({"slot": slot, "id": pid, "name": r["name"], "n_plays": int(r["n"]),
                         "u": round(us[slot], 3), "d_out_pct": round(float(r["d_out_pct"]), 2),
                         "in": round(float(r["d_0_pct"]), 1), "to_1B": round(float(r["d_1_pct"]), 1),
                         "to_3B": round(float(r["d_2_pct"]), 1)})
        else:
            rows.append({"slot": slot, "id": pid, "name": "(no OAA data: league avg)", "n_plays": 0,
                         "u": 0.0, "d_out_pct": 0.0, "in": 0.0, "to_1B": 0.0, "to_3B": 0.0})
    return pd.DataFrame(rows)


def evaluate_pa(pa, spray_bundle, out_bundle, swaps=None, n=6000, seed=0, directional=True):
    """Return dict with sims, lineup table, real-range and league-average positions + coverages."""
    r = pa["runners"]
    sims = spray_model.predict_contact(
        spray_bundle, pa["batter"], pa["p_throws"], pa["balls"], pa["strikes"], pa["outs"],
        int(r["1B"]), int(r["2B"]), int(r["3B"]), inning=pa["inning"], score_diff=pa["score_diff"],
        park=pa["home_team"] if pa["home_team"] != "-" else None, stand=pa["stand"], n=n, seed=seed)
    sims["ev"] = out_model.sample_ev(out_bundle["ev_model"], pa["batter"], len(sims), seed)
    ids = pa_inputs(pa, swaps)
    us = {s: out_model.fielder_u(out_bundle, pid) for s, pid in ids.items()}
    vs = {s: out_model.fielder_v(out_bundle, pid) for s, pid in ids.items()} if directional else None
    k_real, k_avg = make_kernel(out_bundle, us, vs), make_kernel(out_bundle, {})
    pos_real, cov_real = optimal_infield_positions(sims, kernel_fn=k_real, max_pts=MAX_PTS)
    pos_avg, cov_avg = optimal_infield_positions(sims, kernel_fn=k_avg, max_pts=MAX_PTS)
    # What the league-average setup would cost this lineup (cross-evaluation under the real ranges).
    cov_avg_pos_real_ranges = coverage_score(pos_avg, sims, kernel_fn=k_real, max_pts=MAX_PTS)
    return {"sims": sims, "ids": ids, "us": us, "lineup": describe(out_bundle, ids, us),
            "pos_real": pos_real, "cov_real": cov_real, "pos_avg": pos_avg, "cov_avg": cov_avg,
            "cov_avg_pos_real_ranges": cov_avg_pos_real_ranges}


def plot_pa(pa, res, path, show_actual=False):
    import matplotlib.pyplot as plt
    from baseball_data.field import plot_field, WALL_FT
    from spray_model import _heat, BB_TYPES

    sims = drop_battery(res["sims"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout="constrained")
    plot_field(axes[0])
    _heat(axes[0], sims, (-WALL_FT * 0.78, WALL_FT * 0.78), (-25, WALL_FT + 10))
    parts = ", ".join(f"{t.replace('_', ' ')} {(sims['bb_type'] == t).mean():.0%}" for t in BB_TYPES)
    axes[0].set_title(f"All batted balls ({parts})", fontsize=10)
    gb = sims[sims["bb_type"] == "ground_ball"]
    plot_field(axes[1])
    axes[1].set_xlim(-150, 150)
    axes[1].set_ylim(-15, 215)
    im = _heat(axes[1], gb, (-150, 150), (-15, 215))
    axes[1].set_title(
        f"Ground balls, infield zoom | coverage {res['cov_real']:.1%} with these fielders "
        f"(league-average fielders: {res['cov_avg']:.1%})", fontsize=10)
    for lab, ((x, y), r) in FIXED.items():
        axes[1].add_patch(plt.Circle((x, y), r, fill=False, ls=":", ec="#444", lw=1.2, zorder=8))
        axes[1].text(x, y, lab, ha="center", va="center", fontsize=8, color="#444", fontweight="bold", zorder=10)
    names = dict(zip(res["lineup"]["slot"], res["lineup"]["name"]))
    scores = dict(zip(res["lineup"]["slot"], res["lineup"]["d_out_pct"]))
    for lab, (x, y) in res["pos_avg"].items():  # league-average-fielder spots, for contrast
        axes[1].scatter(x, y, s=90, marker="D", facecolor="none", edgecolor="#555", lw=1.2, zorder=9)
        if lab in res["pos_real"]:
            rx, ry = res["pos_real"][lab]
            axes[1].plot([x, rx], [y, ry], c="#555", lw=1, ls="--", zorder=8)
    for lab, (x, y) in res["pos_real"].items():
        axes[1].scatter(x, y, s=190, c="white", edgecolor="black", zorder=9)
        axes[1].text(x, y, lab, ha="center", va="center", fontsize=7, fontweight="bold", zorder=10)
        last = names[lab].split(",")[0]
        axes[1].text(x, y + 10, f"{last} {scores[lab]:+.1f}", ha="center", va="bottom", fontsize=7,
                     bbox=dict(fc="white", ec="none", alpha=0.7, pad=1), zorder=11)
    if show_actual:
        a = pa["actual"]
        bx, by = (a["hc_x"] - 125.42) * 2.5, (198.27 - a["hc_y"]) * 2.5
        for ax in axes:
            ax.scatter(bx, by, s=260, marker="*", c="#1f77b4", edgecolor="white", zorder=12)
        axes[1].text(bx, by - 10, f"actual: {a['events']} (to {a['hit_location']:.0f})" if a["hit_location"] else
                     f"actual: {a['events']}", ha="center", va="top", fontsize=8, color="#1f77b4", zorder=12)
    fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, label="relative density of landing spots (1 = hottest)")
    runners = "".join(b[0] for b, f in pa["runners"].items() if f) or "none"
    fig.suptitle(
        f"{pa['away_team']} @ {pa['home_team']} {pa['game_date']}  {pa['inning_topbot']} {pa['inning']}  |  batter {pa['batter']} "
        f"({pa['stand']}) vs {pa['p_throws']}HP | {pa['balls']}-{pa['strikes']}, {pa['outs']} out, runners: {runners}"
        f"\nlabels: fielder and range score (change in out prob, pct points, vs league avg); gray diamonds = "
        f"league-average-fielder spots", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pa_view.py")
    ap.add_argument("--date", help="game date YYYY-MM-DD (real PA mode)")
    ap.add_argument("--game", type=int, help="game_pk (real PA mode)")
    ap.add_argument("--ab", type=int, help="at_bat_number (real PA mode)")
    hyp = ap.add_argument_group("hypothetical PA (instead of --date/--game/--ab)")
    hyp.add_argument("--batter", type=int, help="MLBAM batter id, e.g. 592450 = Aaron Judge")
    hyp.add_argument("--p-throws", choices=["L", "R"])
    hyp.add_argument("--stand", choices=["L", "R"], help="only for batters not in the spray training data")
    hyp.add_argument("--balls", type=int, default=0)
    hyp.add_argument("--strikes", type=int, default=0)
    hyp.add_argument("--outs", type=int, default=0)
    hyp.add_argument("--on1b", action="store_true")
    hyp.add_argument("--on2b", action="store_true")
    hyp.add_argument("--on3b", action="store_true")
    hyp.add_argument("--park", help="home team code, e.g. NYY (default: league mix)")
    ap.add_argument("--spray", default="2023", help="seasons of the spray bundle, comma list")
    ap.add_argument("--oaa", default="2022,2023", help="seasons of the out-model bundle, comma list")
    ap.add_argument("--fielders", help='swap fielders, e.g. "6:665742,4:663538" (Statcast numbers 3=1B 4=2B 5=3B 6=SS)')
    ap.add_argument("--show-actual", action="store_true", help="overlay the actual batted ball")
    ap.add_argument("--n", type=int, default=6000, help="simulated batted balls")
    ap.add_argument("--overall-only", action="store_true",
                    help="use one range score per fielder instead of per-direction (in / toward 1B / toward 3B)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    if a.game is not None:
        if not (a.date and a.ab is not None):
            ap.error("real PA mode needs --date, --game and --ab")
        pa = get_plate_appearance(a.game, a.ab, a.date)
        tag = f"{a.game}_{a.ab}"
    else:
        if a.batter is None or a.p_throws is None:
            ap.error("hypothetical mode needs --batter and --p-throws (or give --date/--game/--ab)")
        pa = {"batter": a.batter, "pitcher": None, "stand": a.stand, "p_throws": a.p_throws,
              "balls": a.balls, "strikes": a.strikes, "outs": a.outs,
              "runners": {"1B": a.on1b, "2B": a.on2b, "3B": a.on3b}, "inning": 5, "inning_topbot": "",
              "score_diff": 0, "home_team": a.park or "-", "away_team": "hypothetical", "game_date": "",
              "fielders": {s: None for s in SLOT_OF.values()},
              "actual": {"hc_x": np.nan, "hc_y": np.nan, "events": None, "hit_location": None, "bb_type": None}}
        if a.show_actual:
            ap.error("--show-actual needs a real PA")
        tag = f"hyp_{a.batter}"
    spray_bundle = spray_model.load_bundle([int(s) for s in a.spray.split(",")])
    out_bundle = out_model.load_bundle([int(s) for s in a.oaa.split(",")])
    swaps = ({int(k): int(v) for k, v in (x.split(":") for x in a.fielders.split(","))} if a.fielders else None)
    res = evaluate_pa(pa, spray_bundle, out_bundle, swaps, a.n, directional=not a.overall_only)

    print(f"PA {tag}: batter {pa['batter']} ({pa['stand']}) vs {pa['p_throws']}HP, "
          f"{pa['balls']}-{pa['strikes']}, {pa['outs']} out, runners {[b for b, f in pa['runners'].items() if f] or 'none'}")
    print(res["lineup"].to_string(index=False))
    print(f"\nexpected GB outs share  real ranges at optimum: {res['cov_real']:.1%}   "
          f"league-avg fielders at optimum: {res['cov_avg']:.1%}")
    print(f"  league-avg spots evaluated with these fielders' ranges: {res['cov_avg_pos_real_ranges']:.1%} "
          f"(tailoring gains {res['cov_real'] - res['cov_avg_pos_real_ranges']:+.2%})")
    print("spots (x, y):  real-range / league-avg")
    for slot in ("3B", "SS", "2B", "1B"):
        (rx, ry), (ax_, ay) = res["pos_real"][slot], res["pos_avg"][slot]
        print(f"  {slot}: ({rx:+.0f}, {ry:.0f}) / ({ax_:+.0f}, {ay:.0f})")
    if a.show_actual:
        act = pa["actual"]
        print("actual:", act["events"], act["bb_type"], "hit_location", act["hit_location"])
    out = Path(a.out) if a.out else HERE.parents[1] / "data" / "plots" / f"pa_{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_pa(pa, res, out, a.show_actual)
    print("saved", out)


if __name__ == "__main__":
    main()
