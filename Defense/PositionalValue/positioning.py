"""Option 1 infield positioning: place four fielders to cover as much expected ground-ball mass as possible.

Heuristic, not a learned model. A fielder at spot f "covers" a ball landing at p with probability
exp(-(|p - f| / reach_ft)^2); a ball is covered by whichever fielder is best placed for it (max over
fielders). We maximize the summed coverage over the simulated ground balls from spray_model.predict_contact.

Rules and role zones are enforced by default (shift_rules=True):
  * two infielders on each side of second base (the line from home plate through second, x = 0) and all
    four on the infield dirt (radius 95 ft around the mound), per the 2023+ shift rules;
  * each position is confined to a zone around "its" base, which keeps fielders off the mound and near the
    baselines: 3B/SS/2B within 45 ft of their bag, 1B within 28 ft of first base;
  * the pitcher (18 ft around the mound) and catcher (30 ft around the plate) are fixed and their
    grounders are dropped first (FIXED), so infielders are not credited for balls P/C would field;
  * 1B has half the reach of the others (REACH_SCALE) and additionally pays a soft penalty for drifting from a spot just behind the bag, because he must be
    able to take the throw for the force at first. ZONES holds all of these placeholders; they are not fit
    to data (position_data averages from Savant are the obvious source to calibrate them).

Known simplifications (all to be replaced by a learned out-probability model, "option 2"):
  * reach_ft is a placeholder, not calibrated to OAA data.
  * Ignores the runner/bag-holding constraints (1B holding a runner, double-play depth, steals).
  * Ignores fielder-specific range/arm, and ball speed (a hard-hit ball is harder to reach).
"""
import numpy as np

MOUND_Y, DIRT_RADIUS_FT = 60.5, 95.0  # infield dirt arc, centered on the pitcher's plate
_B = 90 / np.sqrt(2)  # first/third base offset from home plate, ft
# position -> (zone center, zone radius ft, side of second base, anchor, penalty weight)
# Penalty = weight * n_balls * (distance_to_anchor / radius)^2, in units of "balls covered".
ZONES = {
    "3B": ((-_B, _B), 45.0, "L", None, 0.0),
    "SS": ((0.0, 2 * _B), 45.0, "L", None, 0.0),
    "2B": ((0.0, 2 * _B), 45.0, "R", None, 0.0),
    "1B": ((_B, _B), 28.0, "R", (62.0, 78.0), 0.04),
}
DEFAULT_REACH_FT = 25
# Per-position multiplier on the reach. 1B is tied to the bag (the pitcher covers if he leaves), so his
# coverage radius is much smaller; that coverage logic itself is not modeled.
REACH_SCALE = {"1B": 0.5}


def reach_for(label, reach_ft):
    return reach_ft * REACH_SCALE.get(label, 1.0)
MAX_PTS = 4000


# Pitcher and catcher start in fixed spots and are assumed to field every grounder inside their range, so
# those balls are removed before the four infielders are placed (placeholder ranges, not fit to data).
FIXED = {"P": ((0.0, MOUND_Y), 18.0), "C": ((0.0, 0.0), 30.0)}


def battery_fielded(sims):
    """Boolean mask: ground balls landing inside the pitcher's or catcher's range."""
    hit = np.zeros(len(sims), dtype=bool)
    for (cx, cy), r in FIXED.values():
        hit |= (np.hypot(sims["x_ft"] - cx, sims["y_ft"] - cy) <= r).to_numpy()
    return hit & (sims["bb_type"] == "ground_ball").to_numpy()


def drop_battery(sims):
    return sims[~battery_fielded(sims)]


def _prep(sims, max_ft=200):
    sims = drop_battery(sims)
    gb = sims[(sims["bb_type"] == "ground_ball") & (sims["distance_ft"] <= max_ft)]
    if len(gb) > MAX_PTS:
        gb = gb.sample(MAX_PTS, random_state=0)
    return gb[["x_ft", "y_ft"]].to_numpy(), len(gb)


def _candidates(shift_rules, step=4):
    xs = np.arange(-130, 131, step)
    ys = np.arange(45, 191, step)
    g = np.array([(x, y) for x in xs for y in ys if abs(x) <= y and x != 0])  # fair territory only
    if shift_rules:  # both feet on the infield dirt
        g = g[np.hypot(g[:, 0], g[:, 1] - MOUND_Y) <= DIRT_RADIUS_FT]
    return g


def _kernel(a, b, reach_ft):
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / reach_ft**2).astype(np.float32)


def coverage_score(positions, sims, reach_ft=DEFAULT_REACH_FT):
    """Expected share of simulated ground balls fielded, for {label: (x, y)} positions."""
    pts, n = _prep(sims)
    if n == 0 or not positions:
        return 0.0
    rows = [_kernel(np.array([xy], dtype=float), pts, reach_for(lab, reach_ft))[0]
            for lab, xy in positions.items()]
    return float(np.max(rows, axis=0).mean())


def optimal_infield_positions(sims, reach_ft=DEFAULT_REACH_FT, shift_rules=True, rounds=10):
    """Return ({label: (x, y)}, coverage) maximizing expected ground-ball coverage.

    With shift_rules, each of 3B/SS/2B/1B is limited to its ZONES entry (side of second base, distance
    from its bag) and the infield dirt, 1B has a reduced reach and is pulled toward its anchor. Without,
    four unconstrained slots are labeled F1..F4 left to right.
    """
    pts, n = _prep(sims)
    if n == 0:
        return {}, 0.0
    cand = _candidates(shift_rules)
    slots = list(ZONES) if shift_rules else ["F1", "F2", "F3", "F4"]

    allowed, penalty, kern = {}, {}, {}
    for slot in slots:
        if shift_rules:
            (cx, cy), radius, side, anchor, weight = ZONES[slot]
            ok = np.hypot(cand[:, 0] - cx, cand[:, 1] - cy) <= radius
            ok &= cand[:, 0] < 0 if side == "L" else cand[:, 0] > 0
            idx = np.flatnonzero(ok)
            pen = np.zeros(len(cand))
            if anchor is not None and weight:
                pen = weight * n * (np.hypot(cand[:, 0] - anchor[0], cand[:, 1] - anchor[1]) / radius) ** 2
        else:
            idx, pen = np.arange(len(cand)), np.zeros(len(cand))
        if not len(idx):
            raise ValueError(f"no candidate spots for {slot}; zones are too tight for the grid")
        allowed[slot], penalty[slot] = idx, pen[idx]
        kern[slot] = _kernel(cand[idx], pts, reach_for(slot, reach_ft))  # (allowed candidates, balls)

    def best(base, slot):
        score = np.maximum(kern[slot] - base, 0).sum(axis=1) - penalty[slot]
        return int(score.argmax())  # local index into allowed[slot]

    chosen, cover = {}, {}
    cur = np.zeros(n, dtype=np.float32)
    for slot in slots:  # greedy start
        chosen[slot] = best(cur, slot)
        cover[slot] = kern[slot][chosen[slot]]
        cur = np.maximum(cur, cover[slot])
    for _ in range(rounds):  # coordinate ascent: re-place each fielder given the others
        changed = False
        for slot in slots:
            base = np.max([cover[o] for o in slots if o != slot], axis=0)
            k = best(base, slot)
            if k != chosen[slot]:
                chosen[slot], changed = k, True
                cover[slot] = kern[slot][k]
        if not changed:
            break
    spots = {slot: tuple(float(v) for v in cand[allowed[slot][k]]) for slot, k in chosen.items()}
    if not shift_rules:
        spots = {f"F{i + 1}": sp for i, sp in enumerate(sorted(spots.values(), key=lambda q: q[0]))}
    return spots, coverage_score(spots, sims, reach_ft)


def compare_to_actual(optimal, actual, sims, reach_ft=DEFAULT_REACH_FT):
    """Distance (ft) per position from optimal to actual, plus the coverage lost by standing at `actual`.

    Only labels present in both dicts are compared per-position; coverage uses all actual positions given.
    """
    rows = {}
    for lab, (ax, ay) in actual.items():
        if lab in optimal:
            ox, oy = optimal[lab]
            rows[lab] = {"actual": (ax, ay), "optimal": (ox, oy),
                         "dx_ft": ax - ox, "dy_ft": ay - oy, "dist_ft": float(np.hypot(ax - ox, ay - oy))}
    return rows, coverage_score(optimal, sims, reach_ft) - coverage_score(actual, sims, reach_ft)
