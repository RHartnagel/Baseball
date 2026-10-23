"""Player directory from MLB StatsAPI (`sports/1/players`), cached, with accent-insensitive name search."""
import unicodedata

import pandas as pd

from . import http
from .cache import cached_frame

PLAYERS_URL = "https://statsapi.mlb.com/api/v1/sports/1/players"
COLS = ["id", "name", "norm", "team_id", "pos", "bat_side", "pitch_hand", "active"]


def _norm(s):
    """Lowercase, accents stripped, punctuation (periods, hyphens, apostrophes) collapsed."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace(".", "").replace("-", " ").replace("'", "").split())


def load_players(season):
    """One row per player on a 40-man/active roster in `season` (StatsAPI, cached)."""
    def fetch():
        people = http.get(PLAYERS_URL, params={"season": int(season)}).json()["people"]
        rows = [{"id": p["id"], "name": p["fullName"], "norm": _norm(p["fullName"]),
                 "team_id": (p.get("currentTeam") or {}).get("id"),
                 "pos": (p.get("primaryPosition") or {}).get("abbreviation"),
                 "bat_side": (p.get("batSide") or {}).get("code"),
                 "pitch_hand": (p.get("pitchHand") or {}).get("code"),
                 "active": bool(p.get("active"))} for p in people]
        return pd.DataFrame(rows, columns=COLS)
    return cached_frame("mlb_players", {"season": int(season)}, fetch)


def load_directory(seasons):
    """Players from several seasons in one frame (newest season's row wins), so anyone with model
    history is findable even if no longer on a roster."""
    frames = [load_players(s) for s in sorted(seasons, reverse=True)]
    return pd.concat(frames, ignore_index=True).drop_duplicates("id", keep="first").reset_index(drop=True)


def search(players, query, role=None, limit=15):
    """Players whose name matches `query` (every token a substring), best first.

    role: None, "pitcher" (pos P) or "fielder" (anyone not a pitcher). Ranking: exact name, then prefix of
    the full name or last name, then active players, then name length.
    """
    toks = _norm(query).split()
    if not toks:
        return players.iloc[0:0]
    df = players
    if role == "pitcher":
        df = df[df["pos"] == "P"]
    elif role == "fielder":
        df = df[df["pos"] != "P"]
    mask = pd.Series(True, index=df.index)
    for t in toks:
        mask &= df["norm"].str.contains(t, regex=False)
    df = df[mask]
    q = " ".join(toks)
    last = df["norm"].str.split().str[-1]
    rank = (~(df["norm"] == q)).astype(int) * 4 + (~(df["norm"].str.startswith(q) | (last == q))).astype(int) * 2 \
        + (~df["active"]).astype(int)
    return df.assign(_r=rank, _l=df["norm"].str.len()).sort_values(["_r", "_l", "name"]).drop(
        columns=["_r", "_l"]).head(limit)


def resolve(players, name_or_id):
    """One player row (Series) from an MLBAM id or an unambiguous name; raises ValueError otherwise."""
    s = str(name_or_id).strip()
    if s.isdigit():
        hit = players[players["id"] == int(s)]
        if hit.empty:
            raise ValueError(f"no player with id {s}")
        return hit.iloc[0]
    hit = search(players, s, limit=5)
    if hit.empty:
        raise ValueError(f"no player matches {name_or_id!r}")
    exact = hit[hit["norm"] == _norm(s)]
    if len(exact) == 1:
        return exact.iloc[0]
    if len(hit) > 1:
        raise ValueError(f"{name_or_id!r} is ambiguous: " + "; ".join(f"{r.name_} ({r.id})" for r in
                         hit.rename(columns={"name": "name_"}).itertuples()))
    return hit.iloc[0]


def stand_for(bat_side, p_throws):
    """Batting side for a matchup: switch hitters ("S") bat opposite the pitcher's hand."""
    if bat_side == "S":
        return "L" if p_throws == "R" else "R"
    return bat_side
