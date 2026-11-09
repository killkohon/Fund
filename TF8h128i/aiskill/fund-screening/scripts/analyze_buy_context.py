#!/usr/bin/env python3
"""为买入 Top10 补充消息面、市场面分析。可独立运行或由 screen_funds.py 调用。"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

ManagerTier = Literal["stable", "deprecated", "watch", "fresh"]

# 经理变更折旧：<6M 高风险 / 6-18M 观察 / >18M 或库内未同步 中性
MANAGER_FRESH_DAYS = 180
MANAGER_WATCH_DAYS = 547

ANN_STRONG_PCT = 15.0
SUBSTANTIVE_MSG_FLAGS = frozenset({"申赎受限", "经理变更(<6M)", "公告风险"})
_HOLDING_LOCK_PAT = re.compile(
    r"(\d+)\s*个月?\s*持有|(\d+)\s*年\s*持有|一年持有|持有期"
)

import numpy as np

try:
    import requests
except ImportError:
    requests = None  # type: ignore

ROOT = Path(__file__).resolve().parents[3]
EM_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://fund.eastmoney.com/",
}
EM_BASE = "https://fundmobapi.eastmoney.com/FundMApi/FundBaseTypeInformation.ashx"
EM_NOTICE = "https://fundmobapi.eastmoney.com/FundMApi/FundNoticeList.ashx"


def _period_return(netv: np.ndarray, days: int) -> float | None:
    if len(netv) < days + 1:
        return None
    base = float(netv[-days - 1])
    if base == 0:
        return None
    return round((float(netv[-1]) - base) / base * 100, 2)


def ensure_manager_columns(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(fund_meta)")
    cols = {row[1] for row in cur.fetchall()}
    if "manager_since" not in cols:
        cur.execute("ALTER TABLE fund_meta ADD COLUMN manager_since DATE DEFAULT NULL")
    if "manager_synced" not in cols:
        cur.execute("ALTER TABLE fund_meta ADD COLUMN manager_synced INTEGER NOT NULL DEFAULT 0")
    conn.commit()


def _manager_tokens(name: str) -> frozenset[str]:
    if not name:
        return frozenset()
    normalized = name.replace("，", ",")
    return frozenset(part.strip() for part in normalized.split(",") if part.strip())


def managers_equivalent(db_manager: str, live_manager: str) -> bool:
    """库内与 API 经理等价；库内为现任子集时视为增聘而非变更。"""
    db_set = _manager_tokens(db_manager)
    live_set = _manager_tokens(live_manager)
    if not live_set:
        return bool(not db_set)
    if db_set == live_set:
        return True
    if db_set and db_set <= live_set:
        return True
    return False


def _parse_date(value: str | date | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _manager_tier_from_since(since: date | None, today: date) -> ManagerTier:
    if since is None:
        return "deprecated"
    days = (today - since).days
    if days < MANAGER_FRESH_DAYS:
        return "fresh"
    if days < MANAGER_WATCH_DAYS:
        return "watch"
    return "deprecated"


def sync_manager_fields(
    conn: sqlite3.Connection,
    fundcode: str,
    live_manager: str,
    *,
    manager_since: date | None,
    manager_synced: int,
) -> None:
    since_s = manager_since.isoformat() if manager_since else None
    conn.execute(
        """
        UPDATE fund_meta
        SET manager = ?, manager_since = ?, manager_synced = ?
        WHERE fundcode = ?
        """,
        (live_manager, since_s, manager_synced, fundcode),
    )
    conn.commit()


def resolve_manager_status(
    conn: sqlite3.Connection,
    fundcode: str,
    meta: dict[str, Any],
    live_manager: str | None,
    today: date | None = None,
) -> dict[str, Any]:
    """经理字段同步 + 四级折旧标签。返回 message 片段与 DB 写回结果。"""
    today = today or date.today()
    db_manager = (meta.get("manager") or "").strip()
    live = (live_manager or "").strip()
    since = _parse_date(meta.get("manager_since"))
    synced = int(meta.get("manager_synced") or 0)

    if not live:
        if db_manager:
            return {
                "line": f"基金经理：{db_manager}",
                "flags": [],
                "tier": "stable",
                "manager_since": since,
                "manager_synced": synced,
                "live_manager": db_manager,
            }
        return {
            "line": "",
            "flags": [],
            "tier": "stable",
            "manager_since": since,
            "manager_synced": synced,
            "live_manager": "",
        }

    if managers_equivalent(db_manager, live):
        tier = _manager_tier_from_since(since, today)
        flags: list[str] = []
        if tier == "fresh":
            flags.append("经理变更(<6M)")
        elif tier == "watch":
            flags.append("经理变更(观察)")
        sync_manager_fields(conn, fundcode, live, manager_since=since, manager_synced=1)
        return {
            "line": f"基金经理：{live}",
            "flags": flags,
            "tier": tier,
            "manager_since": since,
            "manager_synced": 1,
            "live_manager": live,
        }

    # 库内与 API 不一致
    if not synced:
        # 首次发现库内滞后：同步，不计入变更风险
        sync_manager_fields(conn, fundcode, live, manager_since=None, manager_synced=1)
        old = db_manager or "空"
        return {
            "line": f"基金经理：{live}（库内原 {old}，记录滞后已同步）",
            "flags": [],
            "tier": "deprecated",
            "manager_since": None,
            "manager_synced": 1,
            "live_manager": live,
        }

    # 已同步过，现职变更（新事件）
    sync_manager_fields(conn, fundcode, live, manager_since=today, manager_synced=1)
    old = db_manager or "空"
    return {
        "line": f"基金经理：{old} → {live}（新变更，<{MANAGER_FRESH_DAYS}日）",
        "flags": ["经理变更(<6M)"],
        "tier": "fresh",
        "manager_since": today,
        "manager_synced": 1,
        "live_manager": live,
    }


def load_meta(conn: sqlite3.Connection, fundcode: str) -> dict[str, Any]:
    ensure_manager_columns(conn)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT fundcode, fundname, category, categoryname, company, manager, memo,
               manager_since, manager_synced
        FROM fund_meta WHERE fundcode = ?
        """,
        (fundcode,),
    )
    row = cur.fetchone()
    if not row:
        return {"fundcode": fundcode}
    return {
        "fundcode": row[0],
        "fundname": row[1],
        "category": row[2],
        "categoryname": row[3] or "未分类",
        "company": row[4],
        "manager": row[5],
        "memo": row[6],
        "manager_since": row[7],
        "manager_synced": int(row[8] or 0),
    }


def load_netv(conn: sqlite3.Connection, fundcode: str, count: int = 120) -> np.ndarray | None:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT unfixedvalue FROM (
            SELECT unfixedvalue, vdate FROM fund_values
            WHERE fundcode = ? ORDER BY vdate DESC LIMIT ?
        ) ORDER BY vdate ASC
        """,
        (fundcode, count),
    )
    rows = cur.fetchall()
    if len(rows) < 25:
        return None
    return np.array([float(r[0]) for r in rows], dtype=float)


def load_latest_trade(conn: sqlite3.Connection, fundcode: str) -> dict[str, Any]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT vdate, buyable, salable, dividend, netvalue, unfixedvalue
        FROM fund_values WHERE fundcode = ? ORDER BY vdate DESC LIMIT 1
        """,
        (fundcode,),
    )
    row = cur.fetchone()
    if not row:
        return {}
    div = row[3]
    div_s = "" if div is None or str(div).lower() in ("nan", "none", "") else str(div)
    return {
        "vdate": row[0],
        "buyable": bool(row[1]),
        "salable": bool(row[2]),
        "dividend": div_s,
        "netvalue": float(row[4]),
        "unfixedvalue": float(row[5]),
    }


def recent_dividends(conn: sqlite3.Connection, fundcode: str, years: int = 2) -> list[dict[str, str]]:
    cur = conn.cursor()
    since = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
    cur.execute(
        """
        SELECT vdate, dividend FROM fund_values
        WHERE fundcode = ? AND vdate >= ?
          AND dividend IS NOT NULL AND dividend != '' AND dividend != 'nan'
        ORDER BY vdate DESC LIMIT 5
        """,
        (fundcode, since),
    )
    out = []
    for vdate, div in cur.fetchall():
        if div and str(div).strip() and str(div).lower() != "nan":
            out.append({"vdate": vdate, "dividend": str(div).strip()})
    return out


def fetch_em_detail(fundcode: str) -> dict[str, Any]:
    if requests is None:
        return {}
    try:
        r = requests.get(
            EM_BASE,
            params={
                "FCODE": fundcode,
                "deviceid": "Wap",
                "plat": "Wap",
                "product": "EFund",
                "version": "2.0.0",
            },
            headers=EM_HEADERS,
            timeout=8,
        )
        d = r.json()
        if d.get("ErrCode") != 0 or not d.get("Datas"):
            return {}
        x = d["Datas"]
        endnav = x.get("ENDNAV")
        scale_yi = round(float(endnav) / 1e8, 2) if endnav else None
        return {
            "ftype": x.get("FTYPE"),
            "sgzt": x.get("SGZT"),
            "shzt": x.get("SHZT"),
            "syl_1m": _f(x.get("SYL_Y")),
            "syl_3m": _f(x.get("SYL_3Y")),
            "syl_6m": _f(x.get("SYL_6Y")),
            "syl_1y": _f(x.get("SYL_1N")),
            "rank_1m": x.get("RANKM"),
            "rank_3m": x.get("RANKQ"),
            "rank_6m": x.get("RANKHY"),
            "rank_1y": x.get("RANKY"),
            "scale_yi": scale_yi,
            "risk": x.get("RISKLEVEL"),
            "manager_live": x.get("JJJL"),
            "company_live": x.get("JJGS"),
            "fsrq": x.get("FSRQ"),
        }
    except Exception:
        return {}


def fetch_em_notices(fundcode: str, limit: int = 3) -> list[dict[str, str]]:
    if requests is None:
        return []
    try:
        r = requests.get(
            EM_NOTICE,
            params={
                "FCODE": fundcode,
                "pageIndex": 1,
                "pageSize": limit,
                "deviceid": "Wap",
                "plat": "Wap",
                "product": "EFund",
                "version": "2.0.0",
            },
            headers=EM_HEADERS,
            timeout=8,
        )
        d = r.json()
        if d.get("ErrCode") != 0:
            return []
        out = []
        for item in d.get("Datas") or []:
            title = (item.get("TITLE") or "").strip()
            pub = (item.get("PUBLISHDATE") or "")[:10]
            if title:
                out.append({"title": title, "date": pub})
        return out
    except Exception:
        return []


def _f(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return None


def build_peer_stats(conn: sqlite3.Connection) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """按 categoryname 与主题标签分别统计近20日收益，用于市场面比较。"""
    cur = conn.cursor()
    cur.execute("SELECT fundcode, fundname, categoryname FROM fund_meta")
    cat_buckets: dict[str, list[float]] = defaultdict(list)
    theme_buckets: dict[str, list[float]] = defaultdict(list)
    for code, name, cat in cur.fetchall():
        netv = load_netv(conn, code, 30)
        if netv is None:
            continue
        ret = _period_return(netv, 20)
        if ret is None:
            continue
        cat_buckets[cat or "未分类"].append(ret)
        theme = infer_theme(name or "", cat or "", None)
        theme_buckets[theme].append(ret)
    return cat_buckets, theme_buckets


def infer_theme(fundname: str, categoryname: str, ftype: str | None) -> str:
    text = f"{fundname} {categoryname} {ftype or ''}"
    rules = [
        ("黄金", r"黄金|金ETF|金银"),
        ("军工", r"军工|国防|航空|航天"),
        ("消费", r"消费|白酒|食品|饮料"),
        ("科技", r"科技|芯片|半导体|人工智能|信息|互联"),
        ("医药", r"医药|医疗|生物|健康"),
        ("新能源", r"新能源|光伏|锂电|碳中和|绿色"),
        ("海外", r"QDII|海外|港股|美股|恒生|纳斯达克"),
        ("债券", r"债|利率|信用"),
        ("大宗商品", r"大宗|商品|原油|资源"),
    ]
    for theme, pat in rules:
        if re.search(pat, text):
            return theme
    if "股票" in text:
        return "股票型"
    if "混合" in text:
        return "混合型"
    if "指数" in text:
        return "指数型"
    return categoryname or "综合"


def _notice_is_substantive_risk(title: str) -> bool:
    """系统维护类公告不计入实质风险。"""
    if any(k in title for k in ("清盘", "巨额赎回", "限购")):
        return True
    if "暂停" in title and any(k in title for k in ("申购", "赎回", "申赎", "大额")):
        return True
    return False


def _message_verdict(flags: list[str], sg: str, sh: str) -> str:
    high_risk = any(
        "风险" in f or "暂停" in f or f == "经理变更(<6M)" or f == "申赎受限"
        for f in flags
    )
    if len(flags) >= 2 and high_risk:
        return "谨慎"
    if high_risk:
        return "谨慎"
    if flags:
        return "关注"
    if sg == "开放申购" and sh == "开放赎回":
        return "可执行"
    return "中性"


def analyze_message(
    conn: sqlite3.Connection,
    fundcode: str,
    meta: dict,
    trade: dict,
    em: dict,
    divs: list,
    notices: list,
) -> dict[str, Any]:
    points: list[str] = []
    flags: list[str] = []

    sg = em.get("sgzt") or ("开放申购" if trade.get("buyable", True) else "暂停申购")
    sh = em.get("shzt") or ("开放赎回" if trade.get("salable", True) else "暂停赎回")
    points.append(f"申赎：{sg} / {sh}")
    if "暂停" in sg or "暂停" in sh or not trade.get("buyable", True):
        flags.append("申赎受限")

    mgr = resolve_manager_status(conn, fundcode, meta, em.get("manager_live"))
    if mgr["line"]:
        points.append(mgr["line"])
    flags.extend(mgr["flags"])

    if em.get("scale_yi") is not None:
        scale = em["scale_yi"]
        points.append(f"规模约 {scale} 亿元")
        if scale < 1:
            flags.append("规模偏小")
        elif scale > 200:
            flags.append("规模偏大")

    if divs:
        points.append(f"近2年分红：{divs[0]['vdate']} {divs[0]['dividend']}")
    elif trade.get("dividend"):
        points.append(f"最新净值日分红标记：{trade['dividend']}")

    keywords = ("分红", "清盘", "转型", "暂停", "恢复", "经理", "变更", "巨额赎回", "限购")
    hot_notices = [n for n in notices if any(k in n["title"] for k in keywords)]
    show = hot_notices[:2] if hot_notices else notices[:2]
    for n in show:
        points.append(f"公告 {n['date']}：{n['title'][:40]}")
        if _notice_is_substantive_risk(n["title"]):
            flags.append("公告风险")

    if em.get("rank_1y"):
        points.append(f"近1年同类排名 {em['rank_1y']}")

    verdict = _message_verdict(flags, sg, sh)

    return {
        "summary": "；".join(points[:6]),
        "flags": flags,
        "verdict": verdict,
        "details": points,
        "manager_tier": mgr["tier"],
    }


def analyze_market(
    meta: dict,
    netv: np.ndarray | None,
    em: dict,
    cat_stats: dict[str, list[float]],
    theme_stats: dict[str, list[float]],
    theme: str,
) -> dict[str, Any]:
    points: list[str] = []
    cat = meta.get("categoryname") or "未分类"
    ret20 = _period_return(netv, 20) if netv is not None else None
    ret60 = _period_return(netv, 60) if netv is not None else None

    peers = theme_stats.get(theme) or []
    peer_label = f"主题「{theme}」"
    if len(peers) < 5:
        peers = cat_stats.get(cat, [])
        peer_label = f"同类「{cat}」"
    diff: float | None = None
    if ret20 is not None and peers:
        med = float(np.median(peers))
        diff = round(ret20 - med, 2)
        points.append(
            f"近20日收益 {ret20:+.2f}% vs {peer_label}中位 {med:+.2f}%"
            f"（{'强于' if diff > 0 else '弱于'} {abs(diff):.2f}pct）"
        )
    elif ret20 is not None:
        points.append(f"近20日收益 {ret20:+.2f}%")

    theme_rets = theme_stats.get(theme, [])
    if theme_rets:
        tmed = float(np.median(theme_rets))
        points.append(f"板块「{theme}」全市场近20日中位收益 {tmed:+.2f}%（样本{len(theme_rets)}）")

    if em.get("syl_1m") is not None:
        points.append(f"官方近1月/3月/1年：{em.get('syl_1m')}% / {em.get('syl_3m')}% / {em.get('syl_1y')}%")

    verdict = "中性"
    if ret20 is not None and peers:
        med = float(np.median(peers))
        if ret20 < med - 3 and (em.get("syl_3m") or 0) > med:
            verdict = "超跌反弹窗口"
        elif ret20 > med + 5:
            verdict = "相对板块偏强（追价性价比下降）"
        elif ret20 < med - 1:
            verdict = "相对同类偏弱"
        else:
            verdict = "贴合同类"
    if theme in ("黄金", "大宗商品") and ret20 is not None and ret20 < -2:
        verdict = "商品回调中（与技术面超跌共振）"

    return {
        "theme": theme,
        "ret20": ret20,
        "ret60": ret60,
        "ret20_vs_peer": diff,
        "summary": "；".join(points[:4]),
        "verdict": verdict,
        "details": points,
    }


def analyze_fund(
    conn: sqlite3.Connection,
    fundcode: str,
    cat_stats: dict[str, list[float]],
    theme_stats: dict[str, list[float]],
) -> dict[str, Any]:
    meta = load_meta(conn, fundcode)
    trade = load_latest_trade(conn, fundcode)
    netv = load_netv(conn, fundcode, 90)
    divs = recent_dividends(conn, fundcode)
    em = fetch_em_detail(fundcode)
    notices = fetch_em_notices(fundcode)
    time.sleep(0.15)

    theme = infer_theme(meta.get("fundname", ""), meta.get("categoryname", ""), em.get("ftype"))
    msg = analyze_message(conn, fundcode, meta, trade, em, divs, notices)
    mkt = analyze_market(meta, netv, em, cat_stats, theme_stats, theme)

    return {
        "fundcode": fundcode,
        "fundname": meta.get("fundname", ""),
        "memo": meta.get("memo"),
        "theme": theme,
        "message": msg,
        "market": mkt,
    }


def build_market_overview(
    items: list[dict], theme_stats: dict[str, list[float]]
) -> list[dict[str, Any]]:
    theme_cnt: dict[str, int] = defaultdict(int)
    for item in items:
        theme_cnt[item["theme"]] += 1
    overview = []
    for theme, cnt in sorted(theme_cnt.items(), key=lambda x: -x[1]):
        rs = theme_stats.get(theme, [])
        med = round(float(np.median(rs)), 2) if rs else None
        overview.append(
            {"theme": theme, "count": cnt, "median_ret20": med, "sample_size": len(rs)}
        )
    return overview


def infer_signal_regime(
    grad_last: float | None,
    ann_pred_pct: float | None,
    ret20_vs_peer: float | None,
    *,
    min_trend_grad: float = 0.0005,
    min_add_ann_pct: float = 3.0,
) -> str:
    """grad × ANN上行确认 × 相对同类 → 信号模式标签。

    ANN 仅在 predmax 相对现价达到加仓门槛时视为「上行确认」；
    ≤0 或未达门槛不视为看空（predmax 偏低多为模型偏差）。
    """
    grad = grad_last if grad_last is not None else 0.0
    grad_pos = grad > min_trend_grad
    grad_neg = grad < 0
    ann_upside = ann_pred_pct is not None and ann_pred_pct >= min_add_ann_pct
    weak_peer = ret20_vs_peer is not None and ret20_vs_peer < -1.0
    strong_peer = ret20_vs_peer is not None and ret20_vs_peer > 1.0

    if grad_pos and ann_upside and weak_peer:
        return "超跌反弹"
    if grad_pos and ann_upside and strong_peer:
        return "趋势延续"
    if grad_pos and ann_upside:
        return "温和偏多"
    if grad_pos:
        return "温和偏多"
    if grad_neg and ann_upside:
        return "分歧观望"
    if grad_neg:
        return "趋势偏弱"
    return "中性"


def _ann_strong(ann_pred_pct: float | None) -> bool:
    return ann_pred_pct is not None and ann_pred_pct >= ANN_STRONG_PCT


def _substantive_message_risk(msg: dict[str, Any]) -> bool:
    flags = set(msg.get("flags") or [])
    if flags & SUBSTANTIVE_MSG_FLAGS:
        return True
    if msg.get("verdict") != "谨慎":
        return False
    soft = {"经理变更(观察)", "规模偏小", "规模偏大"}
    return any(f not in soft for f in flags)


def detect_holding_lock(fundname: str, memo: str | None = None) -> dict[str, Any]:
    text = f"{fundname} {memo or ''}"
    m = _HOLDING_LOCK_PAT.search(text)
    if not m:
        return {"locked": False}
    if "一年持有" in text:
        period = "1年"
    elif m.group(1):
        period = f"{m.group(1)}个月"
    elif m.lastindex and m.lastindex >= 2 and m.group(2):
        period = f"{m.group(2)}年"
    else:
        period = "持有期"
    return {
        "locked": True,
        "period": period,
        "note": f"持有期产品（{period}），卖出/减仓需确认是否已过锁定期",
    }


def synthesize_holding_decision(
    tech_action: str,
    sell_signals: int,
    add_signals: int,
    ann_pred_pct: float | None,
    msg: dict[str, Any],
    mkt: dict[str, Any],
    *,
    fundname: str = "",
    memo: str | None = None,
    grad_last: float | None = None,
    market_regime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """结合技术面持仓建议与消息面、市场面，输出综合决策。"""
    reasons: list[str] = [f"技术面建议：{tech_action}"]
    final = tech_action
    substantive_msg = _substantive_message_risk(msg)

    if msg["verdict"] == "谨慎":
        reasons.append("消息面偏谨慎")
        if tech_action == "加仓":
            final = "轻仓加仓"
        elif (
            tech_action == "继续持有"
            and sell_signals >= 1
            and substantive_msg
            and not (_ann_strong(ann_pred_pct) and sell_signals < 2)
        ):
            final = "减仓观望"
        elif tech_action == "继续持有" and sell_signals >= 1 and _ann_strong(ann_pred_pct):
            reasons.append("ANN较强，暂缓消息面降级")
    elif msg["verdict"] == "关注":
        reasons.append("消息面需关注")
        if tech_action == "加仓":
            final = "轻仓加仓"
    elif msg["verdict"] == "可执行" and tech_action == "加仓":
        reasons.append("消息面支持加仓执行")

    if mkt["verdict"] in ("超跌反弹窗口", "商品回调中（与技术面超跌共振）"):
        if tech_action in ("加仓", "继续持有"):
            reasons.append(f"市场面：{mkt['verdict']}")
            if tech_action == "继续持有" and msg["verdict"] in ("可执行", "中性", "关注"):
                final = "可考虑加仓"
        elif tech_action == "减仓":
            final = "继续持有"
            reasons.append("市场面仍超跌，暂缓减仓")

    if mkt["verdict"] == "相对板块偏强（追价性价比下降）":
        reasons.append("市场面：相对板块偏强，追价性价比下降")
        if tech_action == "加仓":
            final = "继续持有"

    if mkt["verdict"] == "相对同类偏弱" and tech_action == "加仓":
        reasons.append("相对同类仍偏弱，宜分批")

    # ANN≤0 不降档：predmax<现价不按偏空；仅正向较强时附加理由
    if _ann_strong(ann_pred_pct) and tech_action in ("加仓", "继续持有"):
        reasons.append(f"ANN预测 {ann_pred_pct:+.2f}% 较强")

    if tech_action == "卖出":
        final = "卖出"
        if msg["verdict"] == "谨慎":
            reasons.append("消息面与技术面共振看空")

    regime_adjusted = False
    regime_name = None
    if market_regime:
        regime_name = market_regime.get("regime")
        if regime_name in ("ice_point", "weak_down"):
            if final == "加仓":
                final = "轻仓加仓"
                reasons.append(f"全市场 regime={regime_name}，加仓软降档为轻仓加仓")
                regime_adjusted = True
            elif final == "可考虑加仓":
                final = "继续持有"
                reasons.append(f"全市场 regime={regime_name}，市场弱势顺延观察")
                regime_adjusted = True
            elif final == "轻仓加仓":
                reasons.append(f"全市场 regime={regime_name}，维持轻仓节奏")
        elif "rebound_trap" in (market_regime.get("tags") or []):
            if final in ("加仓", "可考虑加仓", "轻仓加仓"):
                reasons.append("全市场标签 rebound_trap：警惕反弹失败")

    holding_lock = detect_holding_lock(fundname, memo)
    if holding_lock["locked"] and final in ("减仓", "卖出", "减仓观望"):
        reasons.append(holding_lock["note"])

    signal_regime = infer_signal_regime(
        grad_last, ann_pred_pct, mkt.get("ret20_vs_peer")
    )

    weights = {
        "tech": 0.4,
        "ann": 0.35,
        "message": 0.15,
        "market": 0.10,
        "market_regime": 0.05,
    }
    return {
        "action": final,
        "reasons": reasons,
        "signal_regime": signal_regime,
        "weights": weights,
        "holding_lock": holding_lock,
        "market_regime": regime_name,
        "regime_adjusted": regime_adjusted,
    }


def analyze_fund_list(
    codes: list[str],
    ranks: dict[str, int] | None = None,
    tech_by_code: dict[str, dict[str, Any]] | None = None,
    market_regime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    db_path = ROOT / "fund.db"
    conn = sqlite3.connect(db_path)
    ensure_manager_columns(conn)
    cat_stats, theme_stats = build_peer_stats(conn)

    items = []
    for code in codes:
        item = analyze_fund(conn, code, cat_stats, theme_stats)
        if ranks and code in ranks:
            item["rank"] = ranks[code]
        if tech_by_code and code in tech_by_code:
            tech = tech_by_code[code]
            item["grad_last"] = tech.get("grad_last")
            item["ann_pred_pct"] = tech.get("ann_pred_pct")
            item["signal_regime"] = infer_signal_regime(
                tech.get("grad_last"),
                tech.get("ann_pred_pct"),
                item["market"].get("ret20_vs_peer"),
            )
            if int(tech.get("holding") or 0) == 1:
                item["tech_action"] = tech.get("action", "")
                item["sell_signals"] = tech.get("sell_signals", 0)
                item["add_signals"] = tech.get("add_signals", 0)
                item["decision"] = synthesize_holding_decision(
                    tech.get("action", "继续持有"),
                    tech.get("sell_signals", 0),
                    tech.get("add_signals", 0),
                    tech.get("ann_pred_pct"),
                    item["message"],
                    item["market"],
                    fundname=item.get("fundname", ""),
                    memo=item.get("memo"),
                    grad_last=tech.get("grad_last"),
                    market_regime=market_regime,
                )
        items.append(item)

    conn.close()
    return {
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "funds": items,
        "market_overview": build_market_overview(items, theme_stats),
    }


def analyze_portfolio_context(
    buy_top10: list[dict[str, Any]],
    holdings: list[dict[str, Any]],
) -> dict[str, Any]:
    buy_codes = [m["fundcode"] for m in buy_top10]
    buy_ranks = {m["fundcode"]: i for i, m in enumerate(buy_top10, 1)}

    hold_codes = [m["fundcode"] for m in holdings]
    action_order = {"卖出": 0, "减仓": 1, "减仓观望": 2, "继续持有": 3, "可考虑加仓": 4, "轻仓加仓": 5, "加仓": 6}
    hold_ranks = {
        m["fundcode"]: i
        for i, m in enumerate(
            sorted(holdings, key=lambda x: action_order.get(x.get("action", ""), 9)),
            1,
        )
    }
    tech_by_code = {m["fundcode"]: m for m in holdings}
    buy_tech_by_code = {m["fundcode"]: m for m in buy_top10}

    db_path = ROOT / "fund.db"
    conn = sqlite3.connect(db_path)
    from market_regime import WEAK_REGIMES, assess_market_regime

    try:
        try:
            regime = assess_market_regime(conn)
        except Exception as e:
            regime = {
                "as_of": None,
                "regime": "chop",
                "tags": [],
                "scenario": "中性",
                "advice": f"市场评估失败，回退中性：{e}",
                "pool": {},
                "index": {"status": "unavailable"},
            }

        buy_ctx = analyze_fund_list(
            buy_codes, buy_ranks, buy_tech_by_code, market_regime=None
        )
        weak = regime.get("regime") in WEAK_REGIMES
        for item in buy_ctx["funds"]:
            item["execution_hint"] = "defer_or_tranche" if weak else "normal"
            item["market_regime_note"] = (
                f"全市场 {regime.get('regime')}：新建仓宜缓/分批"
                if weak
                else f"全市场 {regime.get('regime')}"
            )
        buy_ctx["market_regime"] = regime

        hold_ctx = analyze_fund_list(
            hold_codes, hold_ranks, tech_by_code, market_regime=regime
        )
        hold_ctx["market_regime"] = regime

        return {
            "market_regime": regime,
            "buy_context": buy_ctx,
            "holding_context": hold_ctx,
        }
    finally:
        conn.close()


# 兼容旧名
def analyze_top10(codes: list[str], ranks: dict[str, int] | None = None) -> dict[str, Any]:
    return analyze_fund_list(codes, ranks)


def _print_fund_message_section(title: str, funds: list[dict]) -> None:
    print(f"\n## {title}\n")
    for item in sorted(funds, key=lambda x: x.get("rank", 99)):
        rank = item.get("rank", "-")
        msg = item["message"]
        print(f"### {rank}. {item['fundcode']} {item['fundname']}")
        print(f"- **结论**：{msg['verdict']}")
        print(f"- **要点**：{msg['summary']}")
        if msg["flags"]:
            print(f"- **关注**：{', '.join(msg['flags'])}")
        print()


def _print_market_section(title: str, overview: list[dict], funds: list[dict]) -> None:
    print(f"\n## {title}\n")
    print("### 组合行业分布\n")
    print("| 主题 | 数量 | 板块近20日中位收益 | 板块样本数 |")
    print("|------|------|-------------------|------------|")
    for row in overview:
        med = "N/A" if row["median_ret20"] is None else f"{row['median_ret20']:+.2f}%"
        print(f"| {row['theme']} | {row['count']} | {med} | {row.get('sample_size', 'N/A')} |")
    print()
    print("### 逐基市场面\n")
    for item in sorted(funds, key=lambda x: x.get("rank", 99)):
        rank = item.get("rank", "-")
        mkt = item["market"]
        print(f"#### {rank}. {item['fundcode']} {item['fundname']}（{mkt['theme']}）")
        print(f"- **结论**：{mkt['verdict']}")
        print(f"- **要点**：{mkt['summary']}")
        print()


def _items_by_code(ts: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for item in (ts or {}).get("items") or []:
        code = item.get("fundcode")
        if code:
            out.setdefault(code, []).append(item)
    return out


def _fmt_suggestion(items: list[dict[str, Any]] | None, side: str | None = None) -> str:
    parts: list[str] = []
    for item in items or []:
        if side and item.get("side") != side:
            continue
        if item.get("side") == "buy":
            parts.append(f"买{item.get('amount') or 0:.0f}元")
        else:
            sh = item.get("shares") or 0
            parts.append(f"卖{sh:.2f}份/约{item.get('amount') or 0:.0f}元")
    return "；".join(parts) if parts else "—"


def _print_position_plan_section(ts: dict[str, Any] | None) -> None:
    if not ts or not ts.get("position_plan"):
        return
    plan = ts["position_plan"]
    band = plan.get("band") or {}
    print("\n## 仓位计划\n")
    print(
        f"- **当前仓位**：{plan.get('equity', 0):.1%} "
        f"（市值 {plan.get('market_value')} / 总资产 {plan.get('total_assets')} / 现金 {plan.get('cash')}）"
    )
    print(
        f"- **目标带**：{band.get('target', 0):.0%} "
        f"（{band.get('min', 0):.0%}–{band.get('max', 0):.0%}） "
        f"regime={plan.get('regime')} 暴跌模式={plan.get('crash_mode')}"
    )
    print(f"- **买入预算**：{plan.get('buy_budget')} 元")
    for note in plan.get("notes") or []:
        print(f"- **说明**：{note}")
    print()


def _print_holding_decision_section(
    funds: list[dict], ts: dict[str, Any] | None = None
) -> None:
    by_code = _items_by_code(ts)
    print("\n## 持仓综合决策（技术 + 消息面 + 市场面）\n")
    print("| 代码 | 名称 | 信号模式 | 技术建议 | 综合决策 | 卖/加 | 建议数量 | 消息面 | 市场面 | 要点 |")
    print("|------|------|----------|----------|----------|-------|----------|--------|--------|------|")
    for item in sorted(funds, key=lambda x: x.get("rank", 99)):
        dec = item.get("decision", {})
        action = dec.get("action", item.get("tech_action", ""))
        regime = dec.get("signal_regime", item.get("signal_regime", ""))
        brief = "; ".join(dec.get("reasons", [])[:2])
        sig = f"{item.get('sell_signals', 0)}/{item.get('add_signals', 0)}"
        qty = _fmt_suggestion(by_code.get(item["fundcode"]))
        print(
            f"| {item['fundcode']} | {item['fundname'][:8]} | {regime} "
            f"| {item.get('tech_action', '')} | **{action}** | {sig} | {qty} "
            f"| {item['message']['verdict']} | {item['market']['verdict']} | {brief} |"
        )
    print()
    for item in sorted(funds, key=lambda x: x.get("rank", 99)):
        dec = item.get("decision", {})
        if not dec:
            continue
        print(f"### {item['fundcode']} {item['fundname']} → **{dec['action']}**")
        for r in dec.get("reasons", []):
            print(f"- {r}")
        print()


def print_context_markdown(ctx: dict[str, Any]) -> None:
    """仅买入 Top10 上下文（兼容旧接口）。"""
    _print_fund_message_section("买入 Top10 消息面分析", ctx["funds"])
    _print_market_section("买入 Top10 市场面分析", ctx["market_overview"], ctx["funds"])


def _fmt_pool_metric(pool: dict[str, Any], key: str) -> str:
    val = pool.get(key)
    if val is None:
        return "N/A"
    if key == "down_ratio":
        return f"{val:.1f}%"
    return f"{val:+.2f}%"


def _print_market_regime_section(regime: dict[str, Any]) -> None:
    pool = regime.get("pool") or {}
    index = regime.get("index") or {}
    print("\n## 今日市场评估\n")
    print(f"- **日期**：{regime.get('as_of') or 'N/A'}")
    print(
        f"- **regime**：{regime.get('regime', 'N/A')} "
        f"（{regime.get('scenario', 'N/A')}）"
    )
    print(f"- **建议**：{regime.get('advice', 'N/A')}")
    print(
        f"- **池**：{_fmt_pool_metric(pool, 'day_median')} / "
        f"{_fmt_pool_metric(pool, 'down_ratio')} / "
        f"{_fmt_pool_metric(pool, 'ret5_median')} / "
        f"{_fmt_pool_metric(pool, 'ret10_median')}"
    )
    idx_parts = [f"status={index.get('status', 'N/A')}"]
    hs300 = index.get("hs300")
    if isinstance(hs300, dict) and hs300.get("ret_1d") is not None:
        idx_parts.append(f"hs300 1d={hs300['ret_1d']:+.2f}%")
    cyb = index.get("cyb")
    if isinstance(cyb, dict) and cyb.get("ret_1d") is not None:
        idx_parts.append(f"cyb 1d={cyb['ret_1d']:+.2f}%")
    print(f"- **指数**：{'，'.join(idx_parts)}")
    print()


def _print_merged_buy_ranking_section(
    merged: dict[str, Any] | None, ts: dict[str, Any] | None = None
) -> None:
    if not merged:
        return
    execution = merged.get("by_execution") or []
    tech_rows = merged.get("by_tech_score") or []
    watched = [r for r in tech_rows if r.get("decision") == "新建观察"]
    if not execution and not watched:
        return
    by_code = _items_by_code(ts)
    print("\n## 买入合并排序（Top10 + 持仓加仓/轻仓加仓/可考虑加仓）\n")
    print(f"- **口径**：{merged.get('method', '')}")
    print(f"- **样本数**：{merged.get('n', 0)} | regime={merged.get('regime')} | as_of={merged.get('as_of')}")
    if execution:
        print("\n### 执行优先序（建议按此操作）\n")
        print("| 序 | 代码 | 名称 | buy_score | ANN% | 决策 | 来源 | 建议数量 | 市场面 |")
        print("|----|------|------|-----------|------|------|------|----------|--------|")
    for r in execution:
        ann = r.get("ann_pred_pct")
        ann_s = "N/A" if ann is None else f"{ann:+.1f}"
        qty = _fmt_suggestion(by_code.get(r["fundcode"]), side="buy")
        print(
            f"| {r.get('exec_rank', '-')} | {r['fundcode']} | {r['fundname'][:10]} "
            f"| {r.get('buy_score', 0):.2f} | {ann_s} | {r.get('decision', '')} "
            f"| {r.get('source', '')} | {qty} | {r.get('market_verdict') or ''} |"
        )
    if watched:
        print("\n### 新建观察（不进入执行序，建议金额为 0）\n")
        print("| 代码 | 名称 | buy_score | 原因 |")
        print("|------|------|-----------|------|")
        for r in watched:
            why = "、".join(r.get("exclude_reason") or [])
            print(f"| {r['fundcode']} | {r['fundname'][:10]} | {r.get('buy_score', 0):.2f} | {why} |")
    if tech_rows:
        print("\n### 技术分序（buy_score）\n")
        print("| 序 | 代码 | 名称 | buy_score | ANN% | 决策 | 来源 |")
        print("|----|------|------|-----------|------|------|------|")
    for r in tech_rows:
        ann = r.get("ann_pred_pct")
        ann_s = "N/A" if ann is None else f"{ann:+.1f}"
        print(
            f"| {r.get('tech_rank', '-')} | {r['fundcode']} | {r['fundname'][:10]} "
            f"| {r.get('buy_score', 0):.2f} | {ann_s} | {r.get('decision', '')} "
            f"| {r.get('source', '')} |"
        )
    print()


def print_portfolio_context_markdown(portfolio: dict[str, Any]) -> None:
    buy = portfolio["buy_context"]
    hold = portfolio["holding_context"]
    regime = portfolio.get("market_regime") or buy.get("market_regime")
    if regime:
        _print_market_regime_section(regime)
    _print_fund_message_section("买入 Top10 消息面分析", buy["funds"])
    _print_market_section("买入 Top10 市场面分析", buy["market_overview"], buy["funds"])
    if hold["funds"]:
        _print_fund_message_section("持仓 消息面分析", hold["funds"])
        _print_market_section("持仓 市场面分析", hold["market_overview"], hold["funds"])
        _print_holding_decision_section(hold["funds"], portfolio.get("trade_suggestions"))
    _print_position_plan_section(portfolio.get("trade_suggestions"))
    _print_merged_buy_ranking_section(
        portfolio.get("merged_buy_ranking"), portfolio.get("trade_suggestions")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="买入Top10/持仓消息面/市场面分析")
    parser.add_argument("codes", nargs="*", help="基金代码列表")
    parser.add_argument("--stdin-json", action="store_true", help="从 stdin 读取 screen_funds JSON")
    parser.add_argument("--holdings-only", action="store_true", help="仅分析 stdin JSON 中的 holdings")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    args = parser.parse_args()

    if args.stdin_json:
        data = json.load(sys.stdin)
        portfolio = analyze_portfolio_context(
            data.get("buy_top10", []),
            data.get("holdings", []),
        )
        if not args.holdings_only:
            from merged_buy_ranking import build_merged_buy_ranking

            portfolio["merged_buy_ranking"] = build_merged_buy_ranking(
                data.get("buy_top10", []),
                data.get("holdings", []),
                portfolio.get("buy_context"),
                portfolio.get("holding_context"),
                portfolio.get("market_regime"),
            )
            from suggest_sizes import build_trade_suggestions

            _conn = sqlite3.connect(ROOT / "fund.db")
            try:
                portfolio["trade_suggestions"] = build_trade_suggestions(
                    _conn, {**data, **portfolio}
                )
            finally:
                _conn.close()
        if args.holdings_only:
            portfolio = {"holding_context": portfolio["holding_context"]}
        if args.format == "json":
            print(json.dumps(portfolio, ensure_ascii=False, indent=2))
        else:
            if "buy_context" in portfolio:
                print_portfolio_context_markdown(portfolio)
            else:
                hold = portfolio["holding_context"]
                _print_fund_message_section("持仓 消息面分析", hold["funds"])
                _print_market_section("持仓 市场面分析", hold["market_overview"], hold["funds"])
                _print_holding_decision_section(hold["funds"])
        return

    codes = args.codes
    if not codes:
        print("错误：未提供基金代码。可传代码列表或使用 --stdin-json", file=sys.stderr)
        sys.exit(1)
    ranks = {c: i for i, c in enumerate(codes, 1)}
    ctx = analyze_fund_list(codes, ranks)
    if args.format == "json":
        print(json.dumps(ctx, ensure_ascii=False, indent=2))
    else:
        print_context_markdown(ctx)


if __name__ == "__main__":
    main()
