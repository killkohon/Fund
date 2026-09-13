#!/usr/bin/env python3
"""基金买入 Top10 与持仓卖出/持有建议。从 fund.db 与 ann/*.pth 计算指标并打分。"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.linalg import hankel

# 项目根目录（TF8h128i）
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polyfit  # noqa: E402

WINDOW = 45
MIN_POINTS = 200
INSIZE = 128

# P0: 趋势/加仓门槛与买入榜前置过滤
MIN_TREND_GRAD = 0.0005  # 过滤极弱正梯度（如 550001≈0.00018）
MIN_ADD_ANN_PCT = 3.0  # 持仓「加仓」最低 ANN%（20日 pred_max 口径）
MIN_BUY_SCALE_YI = 1.0  # 买入榜最低规模（亿元）
EXCLUDE_ETF_FROM_BUY = True

_BUY_EXCLUDE_QDII = re.compile(r"QDII", re.I)
_BUY_EXCLUDE_ETF = re.compile(r"ETF", re.I)
_BUY_RESTRICTED_SG = re.compile(r"暂停|限")


def movingaverage(source: np.ndarray, window: int = WINDOW) -> np.ndarray:
    count = source.size
    target = np.zeros(count, dtype=float)
    total = 0.0
    n = 0
    for i in range(count):
        total += source[i]
        n += 1
        if i >= window:
            total -= source[i - window]
            n = window
        target[i] = total / n
    return target


def matdigserial(hmat: np.ndarray) -> np.ndarray:
    rdat1 = np.asarray(hmat)
    rdata: list[float] = []
    row, col = rdat1.shape
    for c in range(row):
        b = [rdat1[r][c - r] for r in range(c + 1)]
        rdata.append(float(np.mean(b)))
    if col > row:
        for c in range(row, col):
            b = [rdat1[r][c - r] for r in range(row)]
            rdata.append(float(np.mean(b)))
    for r in range(1, row):
        b = [rdat1[c][col - c + r - 1] for c in range(r, row)]
        rdata.append(float(np.mean(b)))
    return np.array(rdata, dtype=float)


def essential_trend(netv: np.ndarray) -> np.ndarray:
    row = math.ceil(netv.size / 2)
    matrix_netv = hankel(netv[:row], netv[row - 1 :])
    mu, ms, mv = np.linalg.svd(matrix_netv)
    dig = np.asmatrix(np.eye(1) * ms[:1])
    matrix_renetv = mu[:, :1] * dig * mv[:1, :]
    return matdigserial(matrix_renetv)


@dataclass
class FundMetrics:
    fundcode: str
    fundname: str
    holding: int
    netv_last: float
    cdfma: float
    idxdv: float
    cdf_dv: float
    grad_last: float
    essv_slope: float
    ann_pred_pct: float | None
    buy_score: float
    sell_signals: int
    add_signals: int
    action: str
    reasons: list[str]


def load_netv(conn: sqlite3.Connection, fundcode: str, count: int = 520) -> tuple[np.ndarray, np.ndarray] | None:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT vdate, unfixedvalue FROM (
            SELECT vdate, unfixedvalue FROM fund_values
            WHERE fundcode = ? ORDER BY vdate DESC LIMIT ?
        ) ORDER BY vdate
        """,
        (fundcode, count),
    )
    rows = cur.fetchall()
    if len(rows) < MIN_POINTS:
        return None
    dates = np.array([r[0] for r in rows])
    netv = np.array([float(r[1]) for r in rows], dtype=float)
    return netv, dates


def load_latest_trade(conn: sqlite3.Connection, fundcode: str) -> dict:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT buyable, salable FROM fund_values
        WHERE fundcode = ? ORDER BY vdate DESC LIMIT 1
        """,
        (fundcode,),
    )
    row = cur.fetchone()
    if not row:
        return {"buyable": True, "salable": True}
    return {"buyable": bool(row[0]), "salable": bool(row[1])}


def trend_passes(tech: dict[str, float]) -> bool:
    return tech["grad_last"] > MIN_TREND_GRAD


def add_position_eligible(m: dict[str, float], pred_pct: float | None) -> bool:
    """加仓硬门槛：梯度与 ANN 均需达标，避免弱动量假多头。"""
    if not trend_passes(m):
        return False
    if pred_pct is None or pred_pct < MIN_ADD_ANN_PCT:
        return False
    return True


def buy_exclusion_reason(
    conn: sqlite3.Connection,
    fundcode: str,
    fundname: str,
    categoryname: str | None,
) -> str | None:
    """买入榜前置排除；返回原因字符串，None 表示可进入候选。"""
    text = f"{fundname} {categoryname or ''}"
    if _BUY_EXCLUDE_QDII.search(text):
        return "QDII"
    if EXCLUDE_ETF_FROM_BUY and _BUY_EXCLUDE_ETF.search(fundname):
        return "场内ETF"

    trade = load_latest_trade(conn, fundcode)
    if not trade.get("buyable", True):
        return "暂停申购"

    try:
        from analyze_buy_context import fetch_em_detail

        em = fetch_em_detail(fundcode)
        if em:
            sg = em.get("sgzt") or ""
            if _BUY_RESTRICTED_SG.search(sg):
                return "申赎受限"
            scale = em.get("scale_yi")
            if scale is not None and scale < MIN_BUY_SCALE_YI:
                return "规模偏小"
    except Exception:
        pass
    return None


def calc_technical(netv: np.ndarray) -> dict[str, float]:
    ma = movingaverage(netv, WINDOW)
    dv = netv - ma
    madv = movingaverage(dv, WINDOW)
    dvstd = float(np.std(dv, ddof=1))
    dvmean = float(np.mean(dv))
    idxdv = float((dv[-1] - dvmean) / dvstd) if dvstd > 0 else 0.0
    cdfma = float(stats.norm.cdf((madv[-1] - np.mean(madv)) / np.std(madv)))
    cdf_dv = float(stats.norm.cdf((dv[-1] - np.mean(dv)) / np.std(dv)))

    x_axis = np.linspace(1, netv.size, netv.size)
    pf = polyfit.polyfit()
    pf.fitting(x_axis, netv)
    grad = pf.gradient(x_axis)
    grad_last = float(grad[-1])

    essv = essential_trend(netv)
    essv_slope = float(essv[-1] - essv[max(0, len(essv) - 21)])

    return {
        "netv_last": float(netv[-1]),
        "cdfma": cdfma,
        "idxdv": idxdv,
        "cdf_dv": cdf_dv,
        "grad_last": grad_last,
        "essv_slope": essv_slope,
    }


def calc_ann_pred_pct(fundcode: str, dates: np.ndarray, netv: np.ndarray) -> float | None:
    pth = ROOT / "ann" / f"{fundcode}.pth"
    if not pth.is_file():
        return None
    try:
        import torch
        import annmodel

        model = annmodel.ANNModel(fundcode, [128, 256, 3])
        model.load_state_dict(
            torch.load(pth, map_location=torch.device(model.device), weights_only=True)
        )
        pred = model.calculate(netv, dates)
        if pred is None or len(pred) == 0:
            return None
        cpred = pred.to("cpu").detach().numpy()
        serial = netv[model.insize - 2 :]
        predmax = float(cpred[-1][0])
        return round((predmax - float(serial[-1])) * 100 / float(serial[-1]), 3)
    except Exception:
        return None


def technical_buy_score(m: dict[str, float]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0

    if m["cdfma"] < 0.35:
        s = max(0.0, (0.35 - m["cdfma"]) / 0.35) * 30
        score += s
        reasons.append(f"CDF(madv)={m['cdfma']:.3f} 偏低")
    elif m["cdfma"] > 0.5:
        score -= 10

    if m["idxdv"] < -0.5:
        s = min(25.0, max(0.0, (-m["idxdv"] - 0.5) / 2.0) * 25)
        score += s
        reasons.append(f"idxdv={m['idxdv']:.2f} 低于均线")

    if m["cdf_dv"] < 0.4:
        score += 10
        reasons.append(f"CDF(dv)={m['cdf_dv']:.2f} 偏弱")

    if m["grad_last"] > 0:
        gnorm = min(1.0, m["grad_last"] * 1000)
        score += gnorm * 20
        reasons.append("拟合梯度为正")
    else:
        score -= 30
        reasons.append("拟合梯度为负，趋势偏弱")

    if m["essv_slope"] > 0:
        score += 10
        reasons.append("SVD主趋势向上")

    return score, reasons


def ann_buy_bonus(pred_pct: float | None) -> tuple[float, list[str]]:
    """ANN 加分：仅奖励 predmax 相对现价的明确上行；≤0 视为无上行确认（非偏空）。

    `ann_pred_pct=(predmax-现价)/现价`；predmax 落在现价下方多为模型偏差，
    不按看空扣分或计入卖点。
    """
    if pred_pct is None:
        return -5.0, ["无 ANN 预测"]
    if pred_pct > 0:
        return min(25.0, pred_pct * 5), [f"ANN预测涨幅 {pred_pct:+.2f}%"]
    return 0.0, [f"ANN predmax未高于现价({pred_pct:+.2f}%)，不按偏空计"]


def count_sell_signals(m: dict[str, float], pred_pct: float | None) -> tuple[int, list[str]]:
    n = 0
    reasons: list[str] = []
    if m["cdfma"] > 0.65:
        n += 1
        reasons.append(f"CDF(madv)={m['cdfma']:.3f} 偏高")
    if m["idxdv"] > 1.0:
        n += 1
        reasons.append(f"idxdv={m['idxdv']:.2f} 远高于均线")
    if m["cdf_dv"] > 0.7:
        n += 1
        reasons.append(f"CDF(dv)={m['cdf_dv']:.2f} 偏强")
    if m["grad_last"] < 0:
        n += 1
        reasons.append("拟合梯度转负")
    # ANN≤0 不计卖点：predmax<现价多为偏差，不作为看空信号
    return n, reasons


def count_add_signals(m: dict[str, float], pred_pct: float | None) -> tuple[int, list[str]]:
    n = 0
    reasons: list[str] = []
    if m["cdfma"] < 0.35:
        n += 1
        reasons.append(f"CDF(madv)={m['cdfma']:.3f} 偏低")
    if m["idxdv"] < -0.5:
        n += 1
        reasons.append(f"idxdv={m['idxdv']:.2f} 低于均线")
    if m["cdf_dv"] < 0.4:
        n += 1
        reasons.append(f"CDF(dv)={m['cdf_dv']:.2f} 偏弱")
    if trend_passes(m):
        n += 1
        reasons.append("拟合梯度为正")
    elif m["grad_last"] > 0:
        reasons.append(f"拟合梯度偏弱(<{MIN_TREND_GRAD})")
    if m["essv_slope"] > 0:
        n += 1
        reasons.append("SVD主趋势向上")
    if pred_pct is not None and pred_pct >= MIN_ADD_ANN_PCT:
        n += 1
        reasons.append(f"ANN预测涨幅 {pred_pct:+.2f}%")
        if pred_pct >= 5.0:
            n += 1
            reasons.append("ANN涨幅较强")
    elif pred_pct is not None and pred_pct > 0:
        reasons.append(f"ANN预测 {pred_pct:+.2f}% 未达加仓门槛({MIN_ADD_ANN_PCT}%)")
    return n, reasons


def holding_action(
    m: dict[str, float], pred_pct: float | None
) -> tuple[str, int, int, list[str]]:
    sell_n, sell_r = count_sell_signals(m, pred_pct)
    add_n, add_r = count_add_signals(m, pred_pct)
    reasons = sell_r + add_r

    if sell_n >= 3:
        return "卖出", sell_n, add_n, reasons
    if sell_n >= 2:
        return "减仓", sell_n, add_n, reasons
    if sell_n == 1:
        if add_n >= 3:
            return "继续持有", sell_n, add_n, reasons + ["多空信号分歧，暂观望"]
        return "减仓", sell_n, add_n, reasons
    if add_n >= 4 and add_position_eligible(m, pred_pct):
        return "加仓", sell_n, add_n, reasons
    if add_n >= 2 and add_position_eligible(m, pred_pct):
        return "加仓", sell_n, add_n, reasons
    if add_n >= 2 and not add_position_eligible(m, pred_pct):
        extra = []
        if not trend_passes(m):
            extra.append(f"梯度未达加仓门槛(>{MIN_TREND_GRAD})")
        if pred_pct is None or pred_pct < MIN_ADD_ANN_PCT:
            extra.append(f"ANN未达加仓门槛(>={MIN_ADD_ANN_PCT}%)")
        return "继续持有", sell_n, add_n, reasons + extra
    return "继续持有", sell_n, add_n, reasons or ["指标中性"]


def list_candidates(conn: sqlite3.Connection, ann_dir: Path) -> list[tuple[str, str, str | None, int]]:
    pth_codes = {p.stem for p in ann_dir.glob("*.pth") if re.fullmatch(r"\d{6}", p.stem)}
    cur = conn.cursor()
    cur.execute("SELECT fundcode, fundname, categoryname, holding FROM fund_meta ORDER BY fundcode")
    rows = []
    for code, name, categoryname, holding in cur.fetchall():
        if code in pth_codes:
            rows.append((code, name, categoryname, int(holding or 0)))
    return rows


def run_screen(top: int, ann_pool: int, fmt: str, with_context: bool = False) -> dict:
    db_path = ROOT / "fund.db"
    ann_dir = ROOT / "ann"
    if not db_path.is_file():
        raise FileNotFoundError(f"找不到 {db_path}")
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"找不到 {ann_dir}")

    conn = sqlite3.connect(db_path)
    candidates = list_candidates(conn, ann_dir)

    technical_rows: list[dict] = []
    for code, name, categoryname, holding in candidates:
        loaded = load_netv(conn, code)
        if loaded is None:
            continue
        netv, dates = loaded
        tech = calc_technical(netv)
        if not trend_passes(tech):
            continue
        t_score, t_reasons = technical_buy_score(tech)
        technical_rows.append(
            {
                "fundcode": code,
                "fundname": name,
                "categoryname": categoryname,
                "holding": holding,
                "dates": dates,
                "netv": netv,
                "tech": tech,
                "t_score": t_score,
                "t_reasons": t_reasons,
            }
        )

    technical_rows.sort(key=lambda x: x["t_score"], reverse=True)
    ann_targets = {r["fundcode"]: r for r in technical_rows[:ann_pool]}

    buy_list: list[FundMetrics] = []
    for row in technical_rows:
        pred_pct = None
        if row["fundcode"] in ann_targets:
            pred_pct = calc_ann_pred_pct(row["fundcode"], row["dates"], row["netv"])
        ann_s, ann_reasons = ann_buy_bonus(pred_pct)
        buy_score = row["t_score"] + ann_s
        buy_list.append(
            FundMetrics(
                fundcode=row["fundcode"],
                fundname=row["fundname"],
                holding=row["holding"],
                netv_last=row["tech"]["netv_last"],
                cdfma=row["tech"]["cdfma"],
                idxdv=row["tech"]["idxdv"],
                cdf_dv=row["tech"]["cdf_dv"],
                grad_last=row["tech"]["grad_last"],
                essv_slope=row["tech"]["essv_slope"],
                ann_pred_pct=pred_pct,
                buy_score=round(buy_score, 2),
                sell_signals=0,
                add_signals=0,
                action="",
                reasons=row["t_reasons"] + ann_reasons,
            )
        )

    buy_list.sort(key=lambda x: x.buy_score, reverse=True)
    category_by_code = {code: cn for code, _, cn, _ in candidates}
    buy_top: list[FundMetrics] = []
    buy_excluded: list[dict[str, str]] = []
    for m in buy_list:
        if m.holding != 0:
            continue
        reason = buy_exclusion_reason(
            conn, m.fundcode, m.fundname, category_by_code.get(m.fundcode)
        )
        if reason:
            if len(buy_excluded) < 20:
                buy_excluded.append({"fundcode": m.fundcode, "reason": reason})
            continue
        buy_top.append(m)
        if len(buy_top) >= top:
            break

    holdings: list[FundMetrics] = []
    for code, name, _categoryname, holding in candidates:
        if holding != 1:
            continue
        loaded = load_netv(conn, code)
        if loaded is None:
            continue
        netv, dates = loaded
        tech = calc_technical(netv)
        pred_pct = calc_ann_pred_pct(code, dates, netv)
        action, sell_n, add_n, reasons = holding_action(tech, pred_pct)
        holdings.append(
            FundMetrics(
                fundcode=code,
                fundname=name,
                holding=1,
                netv_last=tech["netv_last"],
                cdfma=tech["cdfma"],
                idxdv=tech["idxdv"],
                cdf_dv=tech["cdf_dv"],
                grad_last=tech["grad_last"],
                essv_slope=tech["essv_slope"],
                ann_pred_pct=pred_pct,
                buy_score=0.0,
                sell_signals=sell_n,
                add_signals=add_n,
                action=action,
                reasons=reasons,
            )
        )

    action_order = {"卖出": 0, "减仓": 1, "继续持有": 2, "加仓": 3}
    holdings.sort(key=lambda x: (action_order.get(x.action, 9), -x.sell_signals, -x.add_signals))

    conn.close()
    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "window": WINDOW,
        "eligible_with_model": len(candidates),
        "trend_passed": len(technical_rows),
        "filters": {
            "min_trend_grad": MIN_TREND_GRAD,
            "min_add_ann_pct": MIN_ADD_ANN_PCT,
            "min_buy_scale_yi": MIN_BUY_SCALE_YI,
            "exclude_etf_from_buy": EXCLUDE_ETF_FROM_BUY,
        },
        "buy_excluded_sample": buy_excluded,
        "buy_top10": [asdict(m) for m in buy_top],
        "holdings": [asdict(m) for m in holdings],
    }

    if with_context and (buy_top or holdings):
        _scripts = Path(__file__).resolve().parent
        if str(_scripts) not in sys.path:
            sys.path.insert(0, str(_scripts))
        from analyze_buy_context import analyze_portfolio_context, print_portfolio_context_markdown

        portfolio_ctx = analyze_portfolio_context(
            result["buy_top10"],
            result["holdings"],
        )
        result["market_regime"] = portfolio_ctx.get("market_regime")
        result["buy_context"] = portfolio_ctx.get("buy_context")
        result["holding_context"] = portfolio_ctx.get("holding_context")
        from merged_buy_ranking import build_merged_buy_ranking

        result["merged_buy_ranking"] = build_merged_buy_ranking(
            result["buy_top10"],
            result["holdings"],
            result.get("buy_context"),
            result.get("holding_context"),
            result.get("market_regime"),
        )
        from suggest_sizes import build_trade_suggestions

        _size_conn = sqlite3.connect(db_path)
        try:
            result["trade_suggestions"] = build_trade_suggestions(_size_conn, result)
        finally:
            _size_conn.close()

    if fmt == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_markdown(result)
        if with_context and (result.get("buy_context") or result.get("holding_context")):
            print_portfolio_context_markdown(
                {
                    "market_regime": result.get("market_regime"),
                    "buy_context": result.get("buy_context") or {"funds": [], "market_overview": []},
                    "holding_context": result.get("holding_context") or {"funds": [], "market_overview": []},
                    "merged_buy_ranking": result.get("merged_buy_ranking"),
                    "trade_suggestions": result.get("trade_suggestions"),
                }
            )
    return result


def print_markdown(result: dict) -> None:
    print(f"# 基金筛选报告\n")
    print(f"生成时间: {result['generated_at']}  |  窗口: {result['window']}日")
    print(f"有模型基金: {result['eligible_with_model']}  |  趋势过滤后: {result['trend_passed']}")
    flt = result.get("filters", {})
    if flt:
        print(
            f"过滤: grad>{flt.get('min_trend_grad')} | 加仓ANN>={flt.get('min_add_ann_pct')}% "
            f"| 买入排除QDII/ETF/受限/规模<{flt.get('min_buy_scale_yi')}亿\n"
        )
    else:
        print()

    print("## 买入推荐 Top10（未持仓）\n")
    print("| 排名 | 代码 | 名称 | 得分 | CDF(madv) | idxdv | ANN涨幅% | 要点 |")
    print("|------|------|------|------|-----------|-------|----------|------|")
    for i, m in enumerate(result["buy_top10"], 1):
        ann = "N/A" if m["ann_pred_pct"] is None else f"{m['ann_pred_pct']:+.2f}"
        brief = "; ".join(m["reasons"][:2])
        print(
            f"| {i} | {m['fundcode']} | {m['fundname'][:8]} | {m['buy_score']:.1f} "
            f"| {m['cdfma']:.3f} | {m['idxdv']:.2f} | {ann} | {brief} |"
        )

    print("\n## 持仓建议\n")
    if not result["holdings"]:
        print("当前无持仓基金（holding=1）。\n")
        return
    print("| 代码 | 名称 | 建议 | 卖/加 | CDF(madv) | idxdv | ANN涨幅% | 原因 |")
    print("|------|------|------|-------|-----------|-------|----------|------|")
    for m in result["holdings"]:
        ann = "N/A" if m["ann_pred_pct"] is None else f"{m['ann_pred_pct']:+.2f}"
        brief = "; ".join(m["reasons"][:3])
        sig = f"{m['sell_signals']}/{m['add_signals']}"
        print(
            f"| {m['fundcode']} | {m['fundname'][:8]} | **{m['action']}** | {sig} "
            f"| {m['cdfma']:.3f} | {m['idxdv']:.2f} | {ann} | {brief} |"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="基金买入Top10与持仓建议")
    parser.add_argument("--top", type=int, default=10, help="买入推荐数量")
    parser.add_argument("--ann-pool", type=int, default=80, help="技术分前N名才跑ANN推理")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument(
        "--with-context",
        action="store_true",
        help="对买入Top10与持仓追加消息面、市场面及综合决策分析（需网络）",
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    run_screen(
        top=args.top,
        ann_pool=args.ann_pool,
        fmt=args.format,
        with_context=args.with_context,
    )


if __name__ == "__main__":
    main()
