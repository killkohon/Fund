# 设计：筛选报告加仓/减仓建议数量

日期：2026-09-13  
状态：已批准（2026-09-13）  
关联：`aiskill/fund-screening`、`fund_position.py`、`fund.db`  
取代：`docs/superpowers/specs/2026-07-15-position-sizing-design.md` 中的数量公式与 JSON（单基目标 12%、fill 15/60/25）。表结构以已落地的 `fund_trades` / `fund_cash_ledger` / `fund_fee_rules` 为准（见 2026-09-12 流水 UI 设计及后续现金账本）。

## 1. 背景与目标

筛选已给出加仓/减仓等决策，但没有金额或份额。`fund.db` 现有现金流水、持仓流水、申购/赎回费率。

**目标**：在 `--with-context` 报告中输出可执行的建议数量；买入受现金与组合仓位带约束；卖出按决策档固定比例，并估算手续费。

**非目标**

- 不改技术分、ANN、持仓 `decision.action`、合并排序名次
- 不自动写 `fund_trades` / `fund_cash_ledger`（只建议，不落单）
- 不做同主题合计上限
- 不为打到目标仓位而把「继续持有」改成减仓
- 赎回款不计入当日买入预算

## 2. 已确认规则

| 项 | 取值 |
|----|------|
| 模块 | 独立 `suggest_sizes.py`，不塞进 `merged_buy_ranking` |
| 买入拆分 | 同一现金池；`weight = buy_score × 档位系数`；不足从执行序末位往前砍 |
| 档位系数 | 加仓 1.0 · 轻仓加仓 0.5 · 可考虑加仓 0.3 · 新建候选 0.2 |
| 单笔门槛 | 建议买入金额 &lt; 100 元则砍为 0 |
| `buy_score` 缺失或 ≤0 | 权重底数按 1.0 |
| 减仓观望 / 减仓 / 卖出 | 确认份额的 15% / 30% / 100% |
| 继续持有 | 建议数量 0 |
| 无确认份额 | 卖出建议 0，备注「无确认份额」 |
| 锁定期 | 仍给数量，备注锁定期，不改成 0 |
| 无费率规则 | 费率按 0，备注「无费率规则」 |
| 买入金额口径 | 含费现金支出，与 `fund_position.add_trade(buy)` 预扣一致 |
| 卖出口径 | 建议份额 + 约估到账金额（份额×净值 − 赎回费） |
| 仓位定义 | 持仓市值 / (持仓市值 + 现金) |
| 仓位带 | 见第 4 节；永不 100%；非暴跌不清空；暴跌仍留 ≥10% |
| 暴跌模式 | 仅 `regime=ice_point` 且 tags 含 `rebound_trap` |

## 3. 数据来源

全部只读，复用 `fund_position.py`，不复制计费公式。

| 输入 | 来源 |
|------|------|
| 现金 | `current_cash`（最新 `fund_cash_ledger.balance`；pending 申购已扣） |
| 确认份额 | `current_shares`（最近 `init` 之后累加 confirmed buy/sell） |
| 净值 | `fund_values` 该基最新 `unfixedvalue` |
| 申购费 | `lookup_fee_rate(..., "buy", amount)` + `_buy_fee_and_shares` |
| 赎回费 | `_fifo_sell_fee`（持有天数 FIFO） |
| 买入候选 | `merged_buy_ranking.by_execution` 中 decision ∈ {加仓, 轻仓加仓, 可考虑加仓, 新建候选} |
| 卖出候选 | `holding_context.funds` 中 `decision.action` ∈ {减仓观望, 减仓, 卖出} |
| 市场 | 顶层 `market_regime` |

`holding=1` 但份额为 0：仍可出现在买入侧；卖出侧数量为 0。  
无现金账本行：现金=0，买入建议全 0，卖出仍可算。  
无任何 trades：市值=0，仓位=0，走底仓规则（非暴跌且现金≥100 时至少一笔买入）。

## 4. 组合仓位带

| regime | 目标 | 下限 | 上限 |
|--------|------|------|------|
| ice_point | 45% | 25% | 55% |
| weak_down | 55% | 40% | 65% |
| chop | 70% | 55% | 80% |
| mild_up | 80% | 70% | 88% |
| strong_up | 88% | 80% | 95% |
| 暴跌模式 | 20% | 10% | 35% |

`total = market_value + cash`  
`equity = market_value / total`（total=0 则 equity=0）  
`band` = 上表对应行；暴跌模式覆盖 ice_point 的 45% 带。

**买入预算**

```
headroom = band.max * total - market_value
buy_budget = min(cash, max(0, headroom))
```

已超上限 → `buy_budget = 0`。加仓决策仍保留，建议金额为 0，备注「组合仓位已超 regime 上限」。

**底仓**

非暴跌且 `equity < band.min` 且 `cash >= 100`：截断后若所有买入均为 0，则把执行序第一只买入候选恢复为 `min(100, cash, buy_budget)`（`buy_budget` 此时应 &gt;0，因仓位在下限以下则 headroom 为正）。

**卖出总量帽**

卖出后市值不得使 equity 低于 `band.min`：

```
max_sell_mv = max(0, market_value - band.min * total)
```

若建议卖出市值之和超过 `max_sell_mv`，按优先级从低到高缩：减仓观望 → 减仓 → 卖出，同档按执行序/代码升序先缩。缩到份额为 0 即可。暴跌模式用 10% 下限，因此不会建议卖到空仓。

赎回所得不加入当日 `buy_budget`。

## 5. 单基数量

### 5.1 买入

候选来自合并执行序。权重 `w_i = max(buy_score, 1.0) * tier_coef`。

```
raw_i = buy_budget * w_i / sum(w)
```

从执行序末位往前：若 `raw_i < 100` 则该只为 0，其金额按剩余候选权重重新分给未砍的买入（或直接弃掉并停止再分——**选定：弃掉，不回灌**，避免末位反复重算；弃掉后预算可以有残留，写入 `position_plan.unallocated_cash`）。

`est_fee`、`est_shares` 用 `_buy_fee_and_shares(amount, nav, rate)`。无净值则金额仍报、份额为空、备注「无净值」。

### 5.2 卖出

```
sell_frac = {减仓观望: 0.15, 减仓: 0.30, 卖出: 1.0}
shares = current_shares * sell_frac
```

再套第 4 节总量帽。`est_fee = _fifo_sell_fee(...)`；`est_amount = shares * nav - est_fee`。份额不足或为 0 → 0。

## 6. 接入

`screen_funds.py --with-context` 在写入 `merged_buy_ranking` 之后调用：

```python
from suggest_sizes import build_trade_suggestions

result["trade_suggestions"] = build_trade_suggestions(conn, result)
```

无 `--with-context` 不调用。不改 `decision.action`。

新文件：`aiskill/fund-screening/scripts/suggest_sizes.py`  
单测：`aiskill/fund-screening/scripts/test_suggest_sizes.py`

## 7. JSON

```json
"trade_suggestions": {
  "as_of": "2026-09-11",
  "position_plan": {
    "cash": 7894.31,
    "market_value": 129422.29,
    "total_assets": 137316.60,
    "equity": 0.9425,
    "regime": "chop",
    "crash_mode": false,
    "band": {"target": 0.70, "min": 0.55, "max": 0.80},
    "buy_budget": 0.0,
    "unallocated_cash": 0.0,
    "max_sell_mv": 53900.0,
    "notes": ["组合仓位已超 regime 上限"]
  },
  "items": [
    {
      "fundcode": "002861",
      "fundname": "工银智能制造股票A",
      "side": "buy",
      "action": "加仓",
      "amount": 0.0,
      "shares": null,
      "est_fee": 0.0,
      "est_nav": 3.074,
      "weight": 114.37,
      "reasons": ["组合仓位已超 regime 上限"]
    }
  ]
}
```

数字以脚本为准。Agent 不得改金额/份额。

## 8. 报告

在「持仓综合决策」和「买入合并排序」表增加「建议数量」列（买：金额；卖：份额 + 约估金额）。  
增加短节「仓位计划」：当前仓位、目标带、买入预算、是否暴跌模式。  
SKILL「勿做」增加：不得手写编造建议数量。

## 9. 测试

| 用例 | 期望 |
|------|------|
| 现金 8000，三只加仓 score 100/50/50，chop，市值 0 | 预算 8000；金额约 4000/2000/2000 |
| 上例第三只 raw&lt;100 | 第三只 0，预算残留进 unallocated，不回灌 |
| chop，市值 12.9 万，现金 0.8 万（仓位 94%） | buy_budget=0，所有买入 amount=0 |
| 减仓份额 1000 | 建议卖 300；减仓观望 150；卖出 1000 |
| 卖出合计会把仓位打穿下限 | 从减仓观望起缩，直到不超过 max_sell_mv |
| ice_point + rebound_trap | crash_mode，带 20/10/35，买入预算按 35% 上限 |
| 无费率 | est_fee=0，reasons 含「无费率规则」 |
| 份额 0 的减仓 | amount/shares=0，「无确认份额」 |
| 一年持有减仓观望 | 仍有 15% 份额，reasons 含锁定期 |
| 赎回 3000 | buy_budget 不因此增加 |

## 10. 实现顺序

1. `suggest_sizes.py` + 上表单测  
2. `screen_funds.py` 接入 JSON  
3. 更新 `SKILL.md` 报告模板与勿做  
4. 不改 FundCharts / `fund_position` 写路径

## 11. 审阅检查

- 与 2026-07-15 的 12% 单基目标已声明取代，避免双公式  
- 加权买入与仓位上限同时存在时，以上限预算为分子，不会超满仓  
- 暴跌不清空、强势不满仓均有硬帽  
- 赎回不回补当日现金已写死  
- 末位 &lt;100 元弃掉不回灌，避免循环重分歧义  
