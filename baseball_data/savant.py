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


# Statcast fielder_N columns -> position label (fielder_2 = catcher ... fielder_9 = RF).
FIELDER_COLS = {"C": "fielder_2", "1B": "fielder_3", "2B": "fielder_4", "3B": "fielder_5",
                "SS": "fielder_6", "LF": "fielder_7", "CF": "fielder_8", "RF": "fielder_9"}


def _balls_in_play(day):
    df = statcast_pitches(day, day)
    return df[df["type"] == "X"].copy()


def find_plate_appearances(day, batter=None, pitcher=None, team=None):
    """Balls in play on `day` (ISO string/date), one row per PA, so a PA can be picked without ids.

    `team` matches home or away team code (e.g. "NYY"). Sorted by game, then at-bat.
    """
    bip = _balls_in_play(day)
    if batter is not None:
        bip = bip[bip["batter"] == int(batter)]
    if pitcher is not None:
        bip = bip[bip["pitcher"] == int(pitcher)]
    if team is not None:
        bip = bip[(bip["home_team"] == team) | (bip["away_team"] == team)]
    cols = ["game_pk", "at_bat_number", "game_date", "away_team", "home_team", "inning", "inning_topbot",
            "batter", "pitcher", "stand", "p_throws", "outs_when_up", "events", "bb_type", "hit_location"]
    return bip.sort_values(["game_pk", "at_bat_number"])[cols].reset_index(drop=True)


def get_plate_appearance(game_pk, at_bat_number, day):
    """Situation, fielders and the actual outcome of one ball-in-play PA.

    Statcast is date-keyed, so `day` (the game date) is required. Uses the last pitch of the PA
    (type == "X"); raises ValueError if the PA did not end in a ball in play.
    """
    bip = _balls_in_play(day)
    rows = bip[(bip["game_pk"] == int(game_pk)) & (bip["at_bat_number"] == int(at_bat_number))]
    if rows.empty:
        raise ValueError(f"no ball-in-play PA for game_pk={game_pk}, at_bat_number={at_bat_number} on {day}; "
                         "it may not have ended in a ball in play, or the date is wrong")
    r = rows.iloc[-1]
    fielders = {pos: (int(r[col]) if pd.notna(r[col]) else None) for pos, col in FIELDER_COLS.items()}
    fielders["P"] = int(r["pitcher"])
    runners = {b: bool(pd.notna(r[c])) for b, c in (("1B", "on_1b"), ("2B", "on_2b"), ("3B", "on_3b"))}
    return {
        "game_pk": int(r["game_pk"]), "at_bat_number": int(r["at_bat_number"]),
        "game_date": str(r["game_date"])[:10], "batter": int(r["batter"]), "pitcher": int(r["pitcher"]),
        "stand": r["stand"], "p_throws": r["p_throws"],
        "balls": int(r["balls"]), "strikes": int(r["strikes"]), "outs": int(r["outs_when_up"]),
        "runners": runners, "inning": int(r["inning"]), "inning_topbot": r["inning_topbot"],
        "score_diff": int(r["bat_score"] - r["fld_score"]) if pd.notna(r["bat_score"]) else 0,
        "home_team": r["home_team"], "away_team": r["away_team"],
        "fielders": fielders,
        "actual": {"hc_x": float(r["hc_x"]), "hc_y": float(r["hc_y"]), "events": r["events"],
                   "bb_type": r["bb_type"],
                   "hit_location": int(r["hit_location"]) if pd.notna(r["hit_location"]) else None,
                   "launch_speed": float(r["launch_speed"]), "launch_angle": float(r["launch_angle"])},
    }


# --- Outs Above Average (Savant) -------------------------------------------------------------
OAA_LEADERBOARD_URL = "https://baseballsavant.mlb.com/leaderboard/outs_above_average"
OAA_DATA_URL = "https://baseballsavant.mlb.com/visuals/oaa-data"
TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams"

OAA_PLAY_COLS = [
    "target_mlb_id", "name_fielder", "fld_team_id", "fld_name_abbrev", "year", "api_game_date_month_mm",
    "intercept_fielder_role_pos", "fielder_role_display_name",
    "intercept_fielder_starting_position_x", "intercept_fielder_starting_position_y",
    "intercept_point_x", "intercept_point_y", "intercept_point_distance_for_fielder",
    "intercept_direction_group", "resp_bat_side", "h_launch_speed", "is_home_fld",
    "is_hit_into_play_basehit", "is_hit_into_play_field_out", "is_hit_into_play_field_error",
    "adj_estimated_success_rate", "outs_above_average", "fielding_runs_prevented",
]
_OAA_NUMERIC = [c for c in OAA_PLAY_COLS if c not in
                ("name_fielder", "fld_name_abbrev", "intercept_direction_group", "resp_bat_side",
                 "fielder_role_display_name")]


def mlb_team_ids(season):
    """{team_code: team_id} for the 30 MLB clubs (StatsAPI, cached)."""
    def fetch():
        teams = http.get(TEAMS_URL, params={"sportId": 1, "season": int(season)}).json()["teams"]
        return pd.DataFrame([{"team_id": t["id"], "code": t["abbreviation"]} for t in teams])
    df = cached_frame("mlb_teams", {"season": int(season)}, fetch)
    return dict(zip(df["code"], df["team_id"]))


def oaa_leaderboard(season, min_attempts="q"):
    """OAA leaderboard (1B,2B,3B,SS,LF,CF,RF; no P/C) for one season (one row per player_id).

    `min_attempts`: "q" = qualified only (~270 players), or an int to lower the threshold.
    Directional splits (infront / behind / lateral_toward3bline / lateral_toward1bline / rhh / lhh) are
    OAA totals in outs. Percent columns are converted to fractions (`*_rate`).
    """
    def fetch():
        text = http.get(OAA_LEADERBOARD_URL, params=dict(
            type="Fielder", startYear=int(season), endYear=int(season), split="no", team="", range="year",
            min=min_attempts, pos="", roles="", viz="hide", csv="true")).text
        df = pd.read_csv(io.StringIO(text))
        df = df.rename(columns={"last_name, first_name": "name"})
        for c in ("actual_success_rate_formatted", "adj_estimated_success_rate_formatted",
                  "diff_success_rate_formatted"):
            df[c.replace("_formatted", "").replace("_success_rate", "_rate")] = (
                pd.to_numeric(df[c].astype(str).str.rstrip("%"), errors="coerce") / 100)
        df = df.drop(columns=["year"], errors="ignore")
        df["season"] = int(season)
        return df
    return cached_frame("oaa_leaderboard", {"season": int(season), "min": str(min_attempts)}, fetch)


def oaa_plays(team_id, season):
    """Every OAA play (responsible fielder, positions 3-9) for one fielding team-season, cached.

    Coordinates are feet in the same frame as `add_spray`: home plate = (0, 0), +x toward RF, +y toward
    CF (verified on NYY 2024). Month only, no game_pk/date, so rows cannot be joined to a specific PA.
    """
    def fetch():
        rows = http.get(OAA_DATA_URL, params=dict(type="Fielding_Team", playerId=int(team_id),
                                                  startYear=int(season), endYear=int(season))).json()
        df = pd.DataFrame(rows).reindex(columns=OAA_PLAY_COLS)
        df[_OAA_NUMERIC] = df[_OAA_NUMERIC].apply(pd.to_numeric, errors="coerce")
        return df
    return cached_frame("oaa_plays", {"team": int(team_id), "season": int(season)}, fetch)


def oaa_plays_league(season):
    """All 30 teams' OAA plays for a season (first call ~30 requests at 1 req/s, then cached)."""
    frames = [oaa_plays(tid, season) for tid in mlb_team_ids(season).values()]
    return pd.concat([f for f in frames if len(f)], ignore_index=True)
