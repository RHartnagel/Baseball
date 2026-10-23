import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseball_data.players import _norm, resolve, search, stand_for  # noqa: E402

ROWS = [
    (592450, "Aaron Judge", "OF", "R", "R", True),
    (669477, "Jose Ramirez", "3B", "S", "R", True),
    (608070, "José Ramírez", "3B", "S", "R", True),
    (605141, "Mookie Betts", "SS", "R", "R", True),
    (547180, "Bryce Harper", "1B", "L", "R", True),
    (999001, "Bryce Harper Sr", "1B", "L", "R", False),
    (543037, "Gerrit Cole", "P", "R", "R", True),
    (999002, "Cole Hamels", "P", "L", "L", False),
    (999003, "J.D. Martinez", "DH", "R", "R", True),
    (999004, "Ke'Bryan Hayes", "3B", "R", "R", True),
]


@pytest.fixture
def players():
    df = pd.DataFrame(ROWS, columns=["id", "name", "pos", "bat_side", "pitch_hand", "active"])
    df["norm"] = df["name"].map(_norm)
    df["team_id"] = 1
    return df


def test_norm_strips_accents_and_punctuation():
    assert _norm("José Ramírez") == "jose ramirez"
    assert _norm("J.D. Martinez") == "jd martinez"
    assert _norm("Ke'Bryan  Hayes") == "kebryan hayes"


def test_search_accent_insensitive_and_token_order(players):
    assert set(search(players, "jose ramirez")["id"]) == {669477, 608070}
    assert list(search(players, "judge aaron")["id"]) == [592450]
    assert list(search(players, "jd martinez")["id"]) == [999003]


def test_search_ranks_exact_and_active_first(players):
    assert list(search(players, "bryce harper")["id"]) == [547180, 999001]


def test_search_role_filter(players):
    assert list(search(players, "cole", role="pitcher")["id"]) == [543037, 999002]
    assert search(players, "judge", role="pitcher").empty
    assert 543037 not in set(search(players, "cole", role="fielder")["id"])


def test_search_empty_query(players):
    assert search(players, "  ").empty


def test_resolve(players):
    assert resolve(players, 592450)["name"] == "Aaron Judge"
    assert resolve(players, "592450")["name"] == "Aaron Judge"
    assert resolve(players, "Mookie Betts")["id"] == 605141
    assert resolve(players, "Bryce Harper")["id"] == 547180  # exact beats the longer name
    with pytest.raises(ValueError, match="ambiguous"):
        resolve(players, "jose ramirez")
    with pytest.raises(ValueError):
        resolve(players, "nobody here")
    with pytest.raises(ValueError):
        resolve(players, 1)


def test_stand_for():
    assert stand_for("R", "L") == "R"
    assert stand_for("S", "R") == "L"
    assert stand_for("S", "L") == "R"
