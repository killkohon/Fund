---
name: fund-screening
description: >-
  Screens mutual funds from fund.db and ann/*.pth using CDF(madv), idxdv, gradient,
  SVD trend, and ANN predictions. Produces buy Top10 and holdings advice with
  message/market context, integrated decisions, and merged buy ranking
  (Top10 + holding 加仓/轻仓加仓/可考虑加仓). Use when the user asks for fund buy/sell
  recommendations, screening rankings, 基金分析, 买入推荐, 卖出建议, 持仓分析,
  消息面, 市场面, 加仓减仓, 合并排序, 规模排序, A/C份额, 行情评估, or running fund
  screening in TF8h128i.
---

# 基金筛选与买卖建议

基于项目指标策略，**优先执行脚本** `scripts/screen_funds.py`，不要手写重复计算逻辑。

## 前置条件

- 工作目录：`TF8h128i` 项目根（含 `fund.db`、`ann/`）
- 依赖：`numpy`、`scipy`、`torch`（ANN 推理）、`requests`（消息面/市场面）
- 持仓标记：`fund_meta.holding = 1` 表示已持有
- 基金池：仅含 `ann/*.pth` 与 `fund_meta` 交集（当前约 576 只），非全市场
- 孤儿模型：`ann/` 中存在但 `fund_meta` 无记录的 `.pth`（如 350008）不参与筛选

### 维护持仓

```sql
-- 标记持有（示例）
UPDATE fund_meta SET holding = 1 WHERE fundcode IN ('519606', '164402');
-- 取消持有
UPDATE fund_meta SET holding = 0 WHERE fundcode = '010796';
```

## 执行命令（推荐一条龙）

```bash
cd TF8h128i
python aiskill/fund-screening/scripts/screen_funds.py --with-context
```

`--with-context` 会追加：
1. **全局市场评估**（`market_regime`：576 池 + 沪深300/创业板指数）
2. **买入 Top10** 消息面 + 市场面 + 执行节奏标注
3. **已持仓基金** 消息面 + 市场面 + **综合决策**（技术/消息/市场/regime 四维）

**耗时预期**：576 只 + API 约 **2.5~3 分钟**。买入榜 ANN 仅对技术分前 `--ann-pool`（默认 80）名推理；**持仓每只均跑 ANN**。

可选参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--top N` | 10 | 买入推荐数量 |
| `--ann-pool N` | 80 | 买入榜：仅对技术分前 N 名跑 ANN（加速） |
| `--with-context` | 关 | 市场评估 + 买入 Top10 + 持仓 消息面/市场面/综合决策（需网络） |
| `--format json` | markdown | JSON 含 `market_regime`、`buy_context`、`holding_context` |

### Windows / JSON 输出

PowerShell **不要用 `&&`**，用 `;` 或分行执行。

管道重定向 JSON 可能引入控制字符导致解析失败。推荐用 Python 写 UTF-8 文件：

```bash
cd TF8h128i
python -c "import subprocess, pathlib; r=subprocess.run(['python','aiskill/fund-screening/scripts/screen_funds.py','--with-context','--format','json'],capture_output=True,text=True,encoding='utf-8',errors='replace'); p=pathlib.Path('aiskill/fund-screening/scripts/_screen_latest.json'); p.write_text(r.stdout[r.stdout.find('{'):], encoding='utf-8'); print('saved', p)"
```

### 分步执行

```bash
# 技术面 + 全部分析
python aiskill/fund-screening/scripts/screen_funds.py --with-context --format json

# 仅持仓消息面/市场面/综合决策
python aiskill/fund-screening/scripts/screen_funds.py --format json | python aiskill/fund-screening/scripts/analyze_buy_context.py --stdin-json --holdings-only

# 指定代码
python aiskill/fund-screening/scripts/analyze_buy_context.py 519606 164402
```

### 跑完验证清单

- [ ] JSON 含 `generated_at` 且日期为当日
- [ ] JSON 含顶层 `market_regime`（`regime`、`scenario`、`pool`）
- [ ] `holding_context.funds` 长度 = `holdings` 长度（有持仓时）
- [ ] 每只持仓在 `holding_context.funds[]` 中有 `decision.action`
- [ ] JSON 含顶层 `merged_buy_ranking`（`by_execution`、`by_tech_score`）
- [ ] JSON 含顶层 `trade_suggestions.position_plan`
- [ ] `trade_suggestions.items` 覆盖合并买入候选与减仓/减仓观望/卖出
- [ ] 报告含「买入合并排序」节（执行优先序为主）
- [ ] 报告含「仓位计划」与建议数量列
- [ ] 未跳过 `--with-context` 却输出消息面/市场面/市场评估

## JSON 输出结构

`--format json` 时顶层字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `generated_at` | string | ISO 时间 |
| `window` | int | 指标窗口（45） |
| `eligible_with_model` | int | 有 ANN 模型的基金数 |
| `trend_passed` | int | 通过趋势过滤数 |
| `filters` | object | 门槛常量（见下） |
| `buy_excluded_sample` | array | 买入榜被排除样例（最多 20） |
| `buy_top10` | array | 技术面买入推荐 |
| `holdings` | array | 持仓技术面建议 |
| `market_regime` | object | 全局市场评估（见下） |
| `buy_context` | object | 买入 Top 消息面/市场面 |
| `holding_context` | object | 持仓消息面/市场面/综合决策 |
| `merged_buy_ranking` | object | 买入 Top10 + 持仓加仓类合并排序（见下） |
| `trade_suggestions` | object | 仓位计划 + 建议买卖数量（见下） |

`market_regime` 对象（`--with-context` 时由 `market_regime.py` 生成）：

| 字段 | 说明 |
|------|------|
| `as_of` | 评估基准交易日 |
| `regime` | `ice_point` / `weak_down` / `chop` / `mild_up` / `strong_up` |
| `tags` | 附加标签，如 `rebound_trap` |
| `scenario` | `偏谨慎` / `中性` / `偏乐观` |
| `advice` | 对买卖节奏的软约束摘要 |
| `pool` | 池内指标：`n_funds`、`day_median`、`down_ratio`、`ret5_median`、`ret10_median`、`rebound_fail` 等 |
| `index` | 指数：`status`（`ok` / `unavailable`）、`hs300`、`cyb`（含 `ret_1d` 等） |

指数 API 不可用时 `index.status = "unavailable"`，**仍用池内指标判定 regime**，筛选不中断。

`buy_context` / `holding_context` 均为 **object**（非 array），结构：

```json
{
  "analyzed_at": "2026-07-11T23:05:26",
  "market_regime": { "...": "与顶层同源引用" },
  "funds": [
    {
      "fundcode", "message", "market", "decision?", "signal_regime?",
      "execution_hint?", "market_regime_note?"
    }
  ],
  "market_overview": [ { "theme", "count", "median_ret20", "sample_size" } ]
}
```

`merged_buy_ranking`（`--with-context` 时由 `merged_buy_ranking.py` 生成）：

| 字段 | 说明 |
|------|------|
| `as_of` / `regime` | 与 `market_regime` 对齐 |
| `method` | 排序口径说明 |
| `n` | 合并后样本数 |
| `by_execution` | **执行优先序**（报告主表）：持仓 `加仓` > `轻仓加仓` > `可考虑加仓` 优先，新建 `defer` 靠后；同档按 `buy_score` 降序 |
| `by_tech_score` | 纯技术序：按 `buy_score`（技术分+ANN加分）降序 |

纳入规则：
- 买入 Top10：标签 `新建候选` 或 `新建候选(defer)`（看 `execution_hint`）
- 持仓：`decision.action in {加仓, 轻仓加仓, 可考虑加仓}`（含 regime 软降档后）
- 持仓 `buy_score` 由持仓技术指标现场重算（持仓列表里原 `buy_score` 恒为 0）

买入 Top10 **技术面排名不变**；`buy_context.funds[i]` 在弱势 regime 时附加：
- `execution_hint`：`defer_or_tranche`（`ice_point`/`weak_down`）或 `normal`
- `market_regime_note`：短文案（执行节奏提示）

持仓综合决策路径：`holding_context.funds[i].decision.action`（及 `decision.reasons`、`decision.market_regime`、`decision.regime_adjusted`）。

`trade_suggestions`（`--with-context` 时由 `suggest_sizes.py` 生成）：

| 字段 | 说明 |
|------|------|
| `position_plan` | 现金、市值、当前仓位、regime 目标带、买入预算、暴跌模式 |
| `items[]` | `side=buy` 为含费申购金额；`side=sell` 为赎回份额 + 到账约估 |

仓位带与拆分规则见 `docs/superpowers/specs/2026-09-13-trade-size-suggestions-design.md`。数字必须以脚本为准。

## 策略摘要

### 关键门槛（`screen_funds.py`）

| 常量 | 值 | 用途 |
|------|-----|------|
| `MIN_TREND_GRAD` | 0.0005 | 趋势过滤；持仓加仓硬门槛 |
| `MIN_ADD_ANN_PCT` | 3.0% | 持仓「加仓」最低 ANN |
| `MIN_BUY_SCALE_YI` | 1.0 亿 | 买入榜最低规模 |
| `WINDOW` | 45 日 | CDF/均线/SVD 窗口 |

### 买入 Top10（技术面 + 前置过滤）

1. **趋势过滤**：`grad_last > MIN_TREND_GRAD`
2. **打分**：CDF(madv) + idxdv + SVD + ANN 加分（`technical_buy_score` + `ann_buy_bonus`）
3. **排除**（`buy_exclusion_reason`）：

| 条件 | 原因 |
|------|------|
| 名称/类别含 QDII | QDII |
| 名称含 ETF | 场内 ETF |
| `buyable=false` 或 API 暂停申购 | 暂停申购 |
| API 申赎受限 | 申赎受限 |
| 规模 < 1 亿 | 规模偏小 |
| `holding=1` | 已持仓 |

**说明**：名称含 ETF 的排除；**LOF 仍可能入榜**（如煤炭 LOF）。

### 持仓技术面建议（`holding_action`）

| 建议 | 条件概要 |
|------|---------|
| **卖出** | 卖信号 ≥ 3 |
| **减仓** | 卖信号 ≥ 2；或卖信号 = 1 且加信号 < 3 |
| **加仓** | 卖信号 = 0，加信号 ≥ 2，且梯度 + ANN 均达标 |
| **继续持有** | 其余；含「梯度/ANN 未达加仓门槛」「多空分歧」 |

卖/加信号见 `count_sell_signals` / `count_add_signals`。

### 消息面分析（买入 Top10 + 持仓）

| 维度 | 数据来源 | 输出 |
|------|----------|------|
| 申赎状态 | 天天基金 API + `fund_values` | 开放/暂停/限购 |
| 基金经理 | `fund_meta` vs API | 是否变更（含时间折旧） |
| 规模 | API `ENDNAV` | 偏小/偏大 |
| 分红 | `fund_values.dividend` | 近两年分红 |
| 公告 | 天天基金公告 API | 暂停/清盘/限购等 |
| 同类排名 | API 近1年排名 | 长期业绩 |

**消息面结论**：`可执行` / `关注` / `谨慎` / `中性`

**经理变更折旧**（`analyze_buy_context.py`）：
- 库内未同步（首次发现滞后）：同步 API 经理，**不计**变更风险；API 中 `manager_tier: "deprecated"` 表示此类或上任 >18 月
- 新变更（已同步后再变）：`经理变更(<6M)` → 可触发谨慎
- 上任 6–18 月：`经理变更(观察)` → 仅关注
- 上任 >18 月或无 since：折旧中性，不标变更
- 增聘（库内经理 ⊆ 现职）：视为稳定

**局限**：公告 API 可能返回较旧条目，消息面宜作辅助而非唯一依据。

### 市场面分析（买入 Top10 + 持仓）

| 维度 | 计算方式 | 输出 |
|------|----------|------|
| 相对板块 | 近20日收益 vs 主题板块中位数 | 强于/弱于 |
| 板块热度 | 全库同主题近20日中位 | 样本数 + 收益 |
| 官方收益 | API 近1月/3月/1年 | 互证 |
| 组合分布 | 买入Top10或持仓的主题聚类 | 行业集中度 |

**市场面结论**：`超跌反弹窗口` / `贴合同类` / `相对同类偏弱` / `相对板块偏强（追价性价比下降）` / `商品回调中（与技术面超跌共振）` 等

**信号模式**（`signal_regime`）：`超跌反弹` / `趋势延续` / `分歧观望` / `趋势偏弱` / `温和偏多` / `中性` 等

**ANN 口径**（重要）：
- `ann_pred_pct = (predmax - 现价) / 现价`；模型输出为未来窗 max/mean/min，真值理论上应落在 predmin–predmax 之间
- **ANN 正负不作看空/卖点判断**：predmax < 现价（指标为负）视为模型偏差或「无上行确认」，**不计卖信号、不扣买入分、不综合降档**
- ANN 仅作**上行确认/加分**：加仓门槛 `≥ MIN_ADD_ANN_PCT`；较强保护/加分 `≥ 15%`（或加分曲线对正值）

**综合决策权重**（参考）：技术 0.4 / ANN 0.35 / 消息 0.15 / 市场 0.10 / regime 0.05（说明性，不参与数值求和）；ANN≥15% 且卖信号<2 时，消息面 alone 不降级至减仓观望

### 持仓综合决策（`synthesize_holding_decision`）

**全部决策态**（按优先级理解，以脚本输出为准）：

`卖出` · `减仓` · `减仓观望` · `继续持有` · `可考虑加仓` · `轻仓加仓` · `加仓`

在技术面建议基础上，结合消息面、市场面微调：

| 技术面 | 消息面/市场面触发 | 综合决策 |
|--------|-------------------|----------|
| 加仓 | 消息面谨慎/关注 | **轻仓加仓** |
| 加仓 | 消息面可执行 + 市场超跌 | **加仓** |
| 加仓 | 市场相对板块偏强 | **继续持有**（追价性价比下降） |
| 加仓 | 相对同类偏弱 | **加仓**（理由含「宜分批」） |
| 继续持有 | 市场超跌 + 消息面尚可 | **可考虑加仓** |
| 继续持有 | 消息面谨慎 + 卖信号≥1 + ANN 不强 | **减仓观望** |
| 继续持有 | 消息面谨慎 + ANN≥15% + 卖信号<2 | **继续持有**（ANN 保护） |
| 减仓 | 市场仍超跌 | **继续持有**（暂缓减仓） |
| 卖出 | — | **卖出** |
| 减仓/卖出 | 持有期产品（名称含「N年持有」等） | 决策不变，**附加锁定期提示** |

### 全局市场 regime 软降档（`synthesize_holding_decision`）

当 `regime in {ice_point, weak_down}` 时，在消息面/市场面综合之后、锁定期处理之前应用：

| 当前综合决策 | 弱势 regime 调整后 |
|--------------|-------------------|
| **加仓** | → **轻仓加仓**（`regime_adjusted=true`） |
| **可考虑加仓** | → **继续持有**（`regime_adjusted=true`） |
| **轻仓加仓** | 保持；追加市场理由 |
| **减仓 / 卖出 / 减仓观望 / 继续持有** | **不变** |

`rebound_trap` 标签仅追加 caution 理由，**不单独触发降档**。不因弱势自动加重减仓/卖出。

## Agent 输出格式

脚本运行后（**必须含 `--with-context`**），用中文整理为：

```markdown
# 基金筛选报告（YYYY-MM-DD）

## 今日市场评估
（**必须**来自 JSON 顶层 `market_regime`，勿编造）
- 日期 / regime / scenario / advice
- 池：日中位、下跌面、近5/10日
- 指数：沪深300、创业板（`index.status=ok` 时展示；不可用时注明仅用池内）
- 对买卖含义：弱势时加仓类软降档、新建仓宜缓/分批

## 买入推荐 Top10（技术面）
（表格；排名不变；可附 execution_hint / market_regime_note）

## 买入 Top10 消息面分析
（每只：结论 + 要点 + 关注项）

## 买入 Top10 市场面分析
（行业分布 + 逐基）

## 持仓建议（技术面）
（表格：建议、卖/加、指标）

## 持仓 消息面分析
（每只：结论 + 要点 + 关注项）

## 持仓 市场面分析
（持仓行业分布 + 逐基）

## 持仓综合决策
（表格：技术建议 → 综合决策；增加「建议数量」列，数字来自 `trade_suggestions.items`）

## 仓位计划
（**必须**来自 `trade_suggestions.position_plan`：当前仓位、目标带、买入预算、是否暴跌模式）

## 买入合并排序（Top10 + 持仓加仓/轻仓加仓/可考虑加仓）
（**必须**来自 JSON 顶层 `merged_buy_ranking`，勿手算改序）
- 优先展示 **`by_execution`（执行优先序）** 表格，并带「建议数量」列
- 可附 `by_tech_score`（纯技术分序）作对照
- 综合点评中引用执行序前几名作为「今日优先买入/补仓」

## 综合点评（Agent 撰写）
- 买入与持仓行业是否重叠、是否过度集中（同主题 A/C 勿重复计入）
- 持仓中哪些「技术面加仓」被消息面/市场面/regime 降级或强化
- 需优先执行的加仓/减仓标的（与合并排序执行序对齐）

## 说明
- 指标基于 45 日窗口与 unfixedvalue
- 市场评估以脚本 `market_regime` 为准，勿主观编造
- 综合决策以脚本 `decision.action` 为准（含 regime 软降档后），Agent 勿擅自改写
- 合并排序以脚本 `merged_buy_ranking` 为准，Agent 勿擅自改写名次
- 建议数量以脚本 `trade_suggestions` 为准，Agent 勿手写金额/份额
- 不构成投资建议
```

## 扩展任务（正式脚本）

| 任务 | 脚本 | 说明 |
|------|------|------|
| 基金分析（主流程） | `screen_funds.py --with-context` | 见上文 |
| 规模排序导出 | `export_fund_scale_sorted.py` | 输出 CSV/JSON |
| 池内 A/C 配对 | `find_ac_pairs.py` | 同名 + 日收益验证；输出 `fund_ac_pairs.csv` |
| ANN 信号回测 | `ann_signal_backtest.py` | IC/分位/Precision；**非严格 walk-forward**，报告须披露局限 |
| 单基消息/市场 | `analyze_buy_context.py <codes>` | 可独立运行 |
| 全局市场评估 | `market_regime.py` → `assess_market_regime` | 由 `--with-context` 自动调用 |
| 买入合并排序 | `merged_buy_ranking.py` → `build_merged_buy_ranking` | 由 `--with-context` 自动写入 JSON |
| 建议数量 | `suggest_sizes.py` → `build_trade_suggestions` | 现金加权 + 仓位带；写入 `trade_suggestions` |

**池外推荐**（无 ANN 验证）：若用户要求 576 只外推荐，可临时编写脚本调用天天基金 API，报告中须注明「未纳入 ann 模型验证」。

**临时脚本**：`scripts/_tmp_*` 为会话 scratch，**勿作为 Skill 正式入口**。

### 自测

```bash
python aiskill/fund-screening/scripts/test_decision_regime.py
python aiskill/fund-screening/scripts/test_market_regime.py
python aiskill/fund-screening/scripts/test_manager_decay.py
python aiskill/fund-screening/scripts/test_merged_buy_ranking.py
python aiskill/fund-screening/scripts/test_suggest_sizes.py
```

## 故障处理

| 现象 | 处理 |
|------|------|
| 找不到 fund.db | 确认在 TF8h128i 根目录执行 |
| 无持仓 | `holdings` 与 `holding_context.funds` 为空，跳过持仓分析节 |
| `holding_context.funds` 少于持仓数 | 检查 API 是否部分失败；标注缺失项 |
| API 失败 | 保留 DB 技术面数据，报告中标注 |
| 指数 API 不可用 | `market_regime.index.status=unavailable`，仍用池内 regime；报告中注明 |
| ANN 慢 | 减小 `--ann-pool`（仅影响买入榜 ANN） |
| JSON 解析失败 | 改用上文 Python 写文件方式，勿用 PowerShell `Out-File` 直接管道 |

## 勿做

- 不要跳过 `--with-context` 却编造消息面/市场面/市场评估
- 不要编造 `market_regime`：必须来自脚本 JSON 顶层字段
- 不要忽略 `holding_context.funds[].decision` 中的综合决策
- 不要覆盖或改写 `decision.action`（含 regime 软降档结果）
- 不要手算或改写 `merged_buy_ranking` 名次；报告必须引用脚本结果
- 不要手写编造 `trade_suggestions` 金额/份额；不要改回 12% 单基目标公式
- 不要将 `buyable/salable` 与技术指标混淆
- 不要将 `scripts/_tmp_*` 当作正式入口
- 不要对池外推荐声称已通过 ANN 回测

## 代码索引

- 技术面：`screen_funds.py` → `technical_buy_score`、`holding_action`、`buy_exclusion_reason`
- 消息面/市场面：`analyze_buy_context.py` → `analyze_message`、`analyze_market`
- 持仓综合决策：`analyze_buy_context.py` → `synthesize_holding_decision`、`infer_signal_regime`
- 全局市场评估：`market_regime.py` → `assess_market_regime`、`classify_regime`
- 建议数量：`suggest_sizes.py` → `build_trade_suggestions`
