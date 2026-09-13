#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))

import fund_position as fp  # noqa: E402
from suggest_sizes import build_trade_suggestions  # noqa: E402


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    fp.ensure_position_tables(conn)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fund_values ("
        "fundcode TEXT, vdate TEXT, unfixedvalue REAL, netvalue REAL)"
    )
    return conn


def _nav(conn: sqlite3.Connection, code: str, nav: float, vdate: str = "2026-09-11") -> None:
    conn.execute(
        "INSERT INTO fund_values(fundcode,vdate,unfixedvalue,netvalue) VALUES (?,?,?,?)",
        (code, vdate, nav, nav),
    )
    conn.commit()


def _buy_row(code: str, name: str, score: float, action: str, rank: int) -> dict:
    return {
        "fundcode": code,
        "fundname": name,
        "buy_score": score,
        "decision": action,
        "exec_rank": rank,
    }


def _screen(
    *,
    regime: str = "chop",
    tags: list | None = None,
    buys: list,
    holds: list | None = None,
) -> dict:
    return {
        "market_regime": {"as_of": "2026-09-11", "regime": regime, "tags": tags or []},
        "merged_buy_ranking": {"by_execution": buys},
        "holding_context": {"funds": holds or []},
    }


def test_buy_weights_split_cash() -> None:
    conn = _db()
    fp.deposit_cash(conn, 8000, entry_date="2026-09-11")
    for code in ("A00001", "A00002", "A00003"):
        _nav(conn, code, 1.0)
    screen = _screen(
        buys=[
            _buy_row("A00001", "甲", 100, "加仓", 1),
            _buy_row("A00002", "乙", 50, "加仓", 2),
            _buy_row("A00003", "丙", 50, "加仓", 3),
        ]
    )
    out = build_trade_suggestions(conn, screen)
    # chop 上限 80%：空仓 total=8000 → 预算 6400，按 100:50:50 拆
    assert abs(out["position_plan"]["buy_budget"] - 6400) < 0.01
    by = {i["fundcode"]: i for i in out["items"] if i["side"] == "buy"}
    assert abs(by["A00001"]["amount"] - 3200) < 0.01
    assert abs(by["A00002"]["amount"] - 1600) < 0.01
    assert abs(by["A00003"]["amount"] - 1600) < 0.01


def test_drop_sub_100_no_regurgitate() -> None:
    conn = _db()
    fp.deposit_cash(conn, 8000, entry_date="2026-09-11")
    for code in ("A00001", "A00002", "A00003"):
        _nav(conn, code, 1.0)
    screen = _screen(
        buys=[
            _buy_row("A00001", "甲", 10000, "加仓", 1),
            _buy_row("A00002", "乙", 10000, "加仓", 2),
            _buy_row("A00003", "丙", 1, "加仓", 3),
        ]
    )
    out = build_trade_suggestions(conn, screen)
    by = {i["fundcode"]: i["amount"] for i in out["items"] if i["side"] == "buy"}
    assert by["A00003"] == 0
    assert out["position_plan"]["unallocated_cash"] > 0
    budget = out["position_plan"]["buy_budget"]
    assert abs(by["A00001"] + by["A00002"] + out["position_plan"]["unallocated_cash"] - budget) < 0.02


def test_overweight_chop_zero_buy_budget() -> None:
    conn = _db()
    fp.deposit_cash(conn, 8000, entry_date="2026-09-11")
    _nav(conn, "H00001", 1.0)
    fp.add_trade(conn, fundcode="H00001", side="init", settle_date="2026-01-01", shares=129000, nav=1.0)
    _nav(conn, "A00001", 1.0)
    screen = _screen(buys=[_buy_row("A00001", "甲", 100, "加仓", 1)])
    out = build_trade_suggestions(conn, screen)
    assert out["position_plan"]["buy_budget"] == 0
    assert out["position_plan"]["equity"] > 0.80
    buy = [i for i in out["items"] if i["side"] == "buy"][0]
    assert buy["amount"] == 0
    assert any("上限" in r for r in buy["reasons"])


def test_crash_mode_band() -> None:
    conn = _db()
    fp.deposit_cash(conn, 10000, entry_date="2026-09-11")
    screen = _screen(regime="ice_point", tags=["rebound_trap"], buys=[])
    out = build_trade_suggestions(conn, screen)
    assert out["position_plan"]["crash_mode"] is True
    assert out["position_plan"]["band"] == {"target": 0.20, "min": 0.10, "max": 0.35}


def test_floor_keeps_one_buy() -> None:
    conn = _db()
    fp.deposit_cash(conn, 5000, entry_date="2026-09-11")
    _nav(conn, "A00001", 1.0)
    screen = _screen(regime="mild_up", buys=[_buy_row("A00001", "甲", 100, "加仓", 1)])
    out = build_trade_suggestions(conn, screen)
    assert out["position_plan"]["equity"] < 0.70
    buy = [i for i in out["items"] if i["side"] == "buy"][0]
    assert buy["amount"] >= 100


if __name__ == "__main__":
    test_buy_weights_split_cash()
    test_drop_sub_100_no_regurgitate()
    test_overweight_chop_zero_buy_budget()
    test_crash_mode_band()
    test_floor_keeps_one_buy()
    print("test_suggest_sizes task1: all passed")
