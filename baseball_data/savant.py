"""Baseball Savant (Statcast) pitch-level data and batted-ball spray helpers."""
import io
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import http
from .cache import cached_frame

SEARCH_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
ROW_CAP = 25000  # Savant truncates CSV exports at this many rows

# Columns kept from the ~118-column export; everything the spray model and
# downstream defense work needs, nothing more (keeps the cache small).
KEEP_COLS = [
    "game_pk", "game_date", "game_year", "at_bat_number", "pitch_number",
    "batter", "pitcher", "stand", "p_throws", "home_team", "away_team", "inning_topbot", "inning",
    "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b",
    "bat_score", "fld_score", "pitch_type", "release_speed", "pfx_x", "pfx_z", "plate_x", "plate_z",
    "type", "events", "description", "bb_type", "hit_location", "hc_x", "hc_y",
    "launch_speed", "launch_angle", "hit_distance_sc",
    "if_fielding_alignment", "of_fielding_alignment",
    "fielder_2", "fielder_3", "fielder_4", "fielder_5", "fielder_6", "fielder_7", "fielder_8", "fielder_9",
]

# Statcast hit-coordinate origin (home plate) and feet-per-unit scale.
HOME_X, HOME_Y, FT_PER_UNIT = 125.42, 198.27, 2.5


def _fetch_range(start, end, regular_season=True):
    params = {
        "all": "true", "type": "details", "player_type": "batter",
        "hfGT": "R|" if regular_season else "",
        "game_date_gt": start.isoformat(), "game_date_lt": end.isoformat(),
        "min_pitches": "0", "min_results": "0", "min_abs": "0",
        "group_by": "name", "sort_col": "pitches", "sort_order": "desc",
    }
    text = http.get(SEARCH_URL, params=params).text
    if not text.strip():
        return pd.DataFrame(columns=KEEP_COLS)
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    if len(df) >= ROW_CAP and start < end:  # truncated: split the window and retry
        mid = start + (end - start) // 2
        return pd.concat([_fetch_range(start, mid, regular_season),
                          _fetch_range(mid + timedelta(days=1), end, regular_season)], ignore_index=True)
    return df.reindex(columns=KEEP_COLS)


def statcast_pitches(start, end, chunk_days=3, regular_season=True):
    """Pitch-level Statcast data for [start, end] (ISO strings or dates), cached per chunk."""
    start, end = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    frames, d = [], start
    while d <= end:
        c_end = min(d + timedelta(days=chunk_days - 1), end)
        params = {"start": d.isoformat(), "end": c_end.isoformat(), "rs": regular_season}
        frames.append(cached_frame("statcast", params, lambda d=d, c_end=c_end: _fetch_range(d, c_end, regular_season)))
        d = c_end + timedelta(days=1)
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=KEEP_COLS)


def add_spray(df):
    """Add field-centric coordinates for batted balls.

    x_ft: + toward right field, y_ft: + toward center field (home plate = 0,0).
    spray_angle: degrees, 0 = straight up the middle, -45 = LF line, +45 = RF line.
    pull_angle: spray_angle flipped for lefties so + always means pulled.
    distance_ft: Euclidean distance from home plate to landing/first-contact point.
    """
    out = df.copy()
    out["x_ft"] = (out["hc_x"] - HOME_X) * FT_PER_UNIT
    out["y_ft"] = (HOME_Y - out["hc_y"]) * FT_PER_UNIT
    out["spray_angle"] = np.degrees(np.arctan2(out["x_ft"], out["y_ft"]))
    out["pull_angle"] = np.where(out["stand"] == "R", -out["spray_angle"], out["spray_angle"])
    out["distance_ft"] = np.hypot(out["x_ft"], out["y_ft"])
    return out
