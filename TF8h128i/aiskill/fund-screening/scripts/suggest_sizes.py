# -*- coding: utf-8 -*-
"""建议加仓金额 / 减仓份额：现金加权拆分 + regime 仓位带。"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fund_position as fp  # noqa: E402

TIER_COEF = {
    "加仓": 1.0,
    "轻仓加仓": 0.5,
    "可考虑加仓": 0.3,
    "新建候选": 0.2,
    "新建候选(defer)": 0.2,
}
BANDS = {
    "ice_point": {"target": 0.45, "min": 0.25, "max": 0.55},
    "weak_down": {"target": 0.55, "min": 0.40, "max": 0.65},
    "chop": {"target": 0.70, "min": 0.55, "max": 0.80},
    "mild_up": {"target": 0.80, "min": 0.70, "max": 0.88},
    "strong_up": {"target": 0.88, "min": 0.80, "max": 0.95},
}
CRASH_BAND = {"target": 0.20, "min": 0.10, "max": 0.35}
MIN_BUY = 100.0
BUY_ACTIONS = frozenset(TIER_COEF)
SELL_FRAC = {"减仓观望": 0.15, "减仓": 0.30, "卖出": 1.0}


def _latest_navs(conn: sqlite3.Connection) -> dict[str, float]:
    try:
        rows = conn.execute(
            "SELECT fundcode, unfixedvalue FROM fund_values "
            "WHERE unfixedvalue IS NOT NULL ORDER BY vdate"
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    out: dict[str, float] = {}
    for code, nav in rows:
        out[str(code)] = float(nav)
    return out


def _collect_codes(conn: sqlite3.Connection, screen: dict[str, Any], navs: dict[str, float]) -> set[str]:
    codes = set(navs)
    try:
        for (code,) in conn.execute("SELECT DISTINCT fundcode FROM fund_trades"):
            codes.add(str(code))
    except sqlite3.OperationalError:
        pass
    for row in (screen.get("merged_buy_ranking") or {}).get("by_execution") or []:
        if row.get("fundcode"):
            codes.add(str(row["fundcode"]))
    for row in (screen.get("holding_context") or {}).get("funds") or []:
        if row.get("fundcode"):
            codes.add(str(row["fundcode"]))
    return codes


def _market_value(conn: sqlite3.Connection, codes: set[str], navs: dict[str, float]) -> float:
    total = 0.0
    for code in codes:
        nav = navs.get(code)
        if nav is None or nav <= 0:
            continue
        sh = fp.current_shares(conn, code)
        if sh > 0:
            total += sh * nav
    return total


def _fill_buy_fee(
    conn: sqlite3.Connection, code: str, amount: float, nav: float | None
) -> tuple[float, float | None, list[str]]:
    extra: list[str] = []
    if amount <= 0:
        return 0.0, None, extra
    if nav is None or nav <= 0:
        extra.append("无净值")
        return 0.0, None, extra
    n_rules = conn.execute(
        "SELECT COUNT(*) FROM fund_fee_rules WHERE fundcode=? AND side='buy'",
        (code,),
    ).fetchone()[0]
    rate = fp.lookup_fee_rate(conn, code, "buy", amount)
    if n_rules == 0:
        extra.append("无费率规则")
    fee, shares = fp._buy_fee_and_shares(amount, nav, rate)
    return float(fee), float(shares), extra


def build_trade_suggestions(conn: sqlite3.Connection, screen: dict[str, Any]) -> dict[str, Any]:
    mr = screen.get("market_regime") or {}
    as_of = mr.get("as_of") or ""
    regime = mr.get("regime") or "chop"
    tags = list(mr.get("tags") or [])
    crash_mode = regime == "ice_point" and "rebound_trap" in tags
    band = dict(CRASH_BAND if crash_mode else BANDS.get(regime, BANDS["chop"]))

    navs = _latest_navs(conn)
    codes = _collect_codes(conn, screen, navs)
    market_value = _market_value(conn, codes, navs)
    cash = float(fp.current_cash(conn))
    total = market_value + cash
    equity = (market_value / total) if total > 0 else 0.0
    headroom = band["max"] * total - market_value
    buy_budget = min(cash, max(0.0, headroom))
    max_sell_mv = max(0.0, market_value - band["min"] * total)

    notes: list[str] = []
    if buy_budget <= 0 and cash > 0 and equity > band["max"] + 1e-12:
        notes.append("组合仓位已超 regime 上限")

    buys = [
        r
        for r in (screen.get("merged_buy_ranking") or {}).get("by_execution") or []
        if r.get("decision") in BUY_ACTIONS and r.get("fundcode")
    ]
    buys.sort(key=lambda r: int(r.get("exec_rank") or 0))

    raws: list[float] = []
    weights: list[float] = []
    for r in buys:
        score = r.get("buy_score")
        base = float(score) if score is not None and float(score) > 0 else 1.0
        w = base * TIER_COEF[r["decision"]]
        weights.append(w)
    wsum = sum(weights)
    if wsum > 0 and buy_budget > 0:
        raws = [buy_budget * w / wsum for w in weights]
    else:
        raws = [0.0] * len(buys)

    amounts = list(raws)
    unallocated = 0.0
    for i in range(len(amounts) - 1, -1, -1):
        if amounts[i] > 0 and amounts[i] < MIN_BUY:
            unallocated += amounts[i]
            amounts[i] = 0.0

    if (
        not crash_mode
        and equity < band["min"]
        and cash >= MIN_BUY
        and buy_budget >= MIN_BUY
        and buys
        and all(a <= 0 for a in amounts)
    ):
        amounts[0] = min(MIN_BUY, cash, buy_budget)
        unallocated = max(0.0, unallocated - amounts[0])

    items: list[dict[str, Any]] = []
    over_cap = buy_budget <= 0 and bool(buys)
    for r, w, amt in zip(buys, weights, amounts):
        reasons: list[str] = []
        if over_cap:
            reasons.append("组合仓位已超 regime 上限")
            amt = 0.0
        nav = navs.get(str(r["fundcode"]))
        fee, shares, extra = _fill_buy_fee(conn, str(r["fundcode"]), amt, nav)
        reasons.extend(extra)
        items.append(
            {
                "fundcode": r["fundcode"],
                "fundname": r.get("fundname") or "",
                "side": "buy",
                "action": r.get("decision") or "",
                "amount": round(float(amt), 2),
                "shares": None if shares is None else round(shares, 4),
                "est_fee": round(fee, 2),
                "est_nav": nav,
                "weight": round(w, 4),
                "reasons": reasons,
            }
        )

    return {
        "as_of": as_of,
        "position_plan": {
            "cash": round(cash, 2),
            "market_value": round(market_value, 2),
            "total_assets": round(total, 2),
            "equity": round(equity, 4),
            "regime": regime,
            "crash_mode": crash_mode,
            "band": band,
            "buy_budget": round(buy_budget, 2),
            "unallocated_cash": round(unallocated, 2),
            "max_sell_mv": round(max_sell_mv, 2),
            "notes": notes,
        },
        "items": items,
    }
