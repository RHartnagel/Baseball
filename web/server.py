r"""Local web page for testing defensive-positioning scenarios.

    .venv\Scripts\python.exe web\server.py [--port 5000] [--seasons 2022,2023,2024,2025,2026]

Opens nothing by itself; browse to http://127.0.0.1:5000. Localhost only, no auth. A run takes ~9 s per seed,
so requests are serialized with a lock.
"""
import argparse
import sys
import threading
from datetime import date
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Defense" / "PositionalValue"))
from baseball_data import players as P  # noqa: E402
import scenario  # noqa: E402

app = Flask(__name__, static_folder=None)
_run_lock = threading.Lock()
STATE = {}


def init(seasons):
    STATE["models"] = scenario.get_models()
    STATE["players"] = P.load_directory(seasons)
    m = STATE["models"]
    STATE["has_history"] = {
        "batter": lambda pid: pid in m.batters,
        "pitcher": lambda pid: m.pitcher_bip.get(pid, 0) >= scenario.MIN_PITCHER_BIP,
        "fielder": lambda pid: pid in m.fielders,
    }


def _player_row(r, role):
    return {"id": int(r["id"]), "name": r["name"], "pos": r["pos"], "bat_side": r["bat_side"],
            "pitch_hand": r["pitch_hand"], "active": bool(r["active"]),
            "has_history": bool(STATE["has_history"][role](int(r["id"])))}


@app.get("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.get("/api/meta")
def meta():
    m = STATE["models"]
    return jsonify({"parks": m.parks, "pitch_types": m.pitch_types, "slots": list(scenario.SLOTS),
                    "max_n": scenario.MAX_N, "max_seeds": scenario.MAX_SEEDS,
                    "spray_seasons": list(scenario.DEFAULT_SPRAY), "oaa_seasons": list(scenario.DEFAULT_OAA)})


@app.get("/api/players")
def players():
    role = request.args.get("role", "batter")
    if role not in STATE["has_history"]:
        return jsonify({"error": "role must be batter, pitcher or fielder"}), 400
    hits = P.search(STATE["players"], request.args.get("q", ""), None if role == "batter" else role, limit=12)
    return jsonify([_player_row(r, role) for _, r in hits.iterrows()])


@app.post("/api/scenario")
def run():
    spec = request.get_json(silent=True)
    if not isinstance(spec, dict):
        return jsonify({"error": "send a JSON object"}), 400
    spec, notes = dict(spec), []
    directory = STATE["players"].set_index("id", drop=False)

    def lookup(pid):
        try:
            return directory.loc[int(pid)]
        except (KeyError, TypeError, ValueError):
            return None

    pitcher = lookup(spec["pitcher"]) if spec.get("pitcher") not in (None, "") else None
    if pitcher is not None and not spec.get("p_throws"):
        spec["p_throws"] = pitcher["pitch_hand"]
    batter = lookup(spec.get("batter"))
    if (batter is not None and not spec.get("stand") and spec.get("p_throws") in ("L", "R")
            and int(batter["id"]) not in STATE["models"].batters):
        spec["stand"] = P.stand_for(batter["bat_side"], spec["p_throws"])
        notes.append(f"{batter['name']} has no training history: batting {spec['stand']} (from roster "
                     f"data), league-average tendencies")
    try:
        with _run_lock:
            res = scenario.run_scenario(spec, STATE["models"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    for row in res["lineup"]:  # chosen fielder with no OAA plays: show who he is, not a placeholder
        who = lookup(row["id"]) if row["id"] is not None else None
        if who is not None and row["n_plays"] == 0:
            row["name"] = who["name"]
    res["notes"] = notes + res["notes"]
    return jsonify(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--seasons", default=",".join(str(y) for y in range(2022, date.today().year + 1)),
                    help="StatsAPI roster seasons that feed the name search")
    a = ap.parse_args()
    print("loading models and player directory ...", flush=True)
    init([int(s) for s in a.seasons.split(",")])
    print(f"ready: http://127.0.0.1:{a.port}", flush=True)
    app.run(host="127.0.0.1", port=a.port, threaded=True)


if __name__ == "__main__":
    main()
