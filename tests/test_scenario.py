import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Defense" / "PositionalValue"))
import scenario  # noqa: E402

FAKE = SimpleNamespace(parks=["NYY", "BOS"], pitch_types=["FF", "SL"])
BASE = {"batter": 592450, "p_throws": "R"}


def norm(**kw):
    return scenario.normalize({**BASE, **kw}, FAKE)


def test_defaults_and_coercion():
    s = norm(balls="1", runners={"1B": 1}, park="nyy", fielders={"SS": "669477"})
    assert s["balls"] == 1 and s["runners"] == {"1B": True, "2B": False, "3B": False}
    assert s["park"] == "NYY" and s["fielders"] == {"3B": None, "SS": 669477, "2B": None, "1B": None}
    assert s["pitcher"] is None and s["n_seeds"] == 1 and s["directional"] is True


def test_blank_park_and_pitcher_mean_league_average():
    s = norm(park="", pitcher="")
    assert s["park"] is None and s["pitcher"] is None


@pytest.mark.parametrize("bad", [
    {"balls": 4}, {"strikes": 3}, {"outs": 3}, {"p_throws": "X"}, {"park": "ZZZ"}, {"pitch_type": "XX"},
    {"n": 10}, {"n_seeds": 9}, {"stand": "Q"},
    {"fielders": {"SS": 1, "2B": 1}},  # same player twice
])
def test_invalid_specs_raise(bad):
    with pytest.raises(ValueError):
        norm(**bad)


def test_batter_required_and_errors_are_listed_together():
    with pytest.raises(ValueError) as e:
        scenario.normalize({"p_throws": "R", "balls": 9, "outs": 5}, FAKE)
    msg = str(e.value)
    assert "batter is required" in msg and "balls" in msg and "outs" in msg


@pytest.mark.skipif(not (ROOT / "data" / "models" / "spray_2026.joblib").exists()
                    or not (ROOT / "data" / "models" / "out_2026.joblib").exists(),
                    reason="trained model bundles not present")
def test_run_scenario_end_to_end():
    spec = {**BASE, "balls": 1, "strikes": 2, "outs": 1, "runners": {"1B": True}, "park": "NYY", "n": 1500}
    res = scenario.run_scenario(spec, plot=False)
    json.dumps(res)  # JSON-able
    assert set(res["spots"]) == {"3B", "SS", "2B", "1B"}
    assert 0.2 < res["coverage"]["real"] < 0.9
    assert res["coverage"]["real"] >= res["coverage"]["league_avg_spots_real_ranges"] - 1e-9
    assert abs(sum(res["bb_mix"].values()) - 1) < 1e-9
    assert res["flags"]["batter_has_history"] is True
    assert res["plot_png_b64"] is None
