# 设计：基金仓位流水录入（fund_trades + 费率分档 + FundCharts UI）

日期：2026-09-12  
状态：待用户审阅  
关联：`docs/superpowers/specs/2026-07-15-position-sizing-design.md`（仓位建议前置数据；本阶段不实现现金账本与 sizing 计算）  
范围：`fund.db` 表结构、`FundNetv.py` 读写与确认逻辑、`FundCharts.py` 底部 `manipulate_panel` 手工录入与费率编辑

## 1. 背景与目标

当前仅有 `fund_meta.holding` 0/1，无法记录份额、成本、预申购/预赎回与手续费。  
目标：在 SQLite 中落地可审计的仓位流水，并在桌面 UI 中手工录入；买卖为预委托，确认日用净值回填并自动计费。

**非目标（本阶段）**
- `fund_cash_ledger` 与目标仓位 sizing
- 自动将 `holding` 置 0
- Web UI、ANN / `aiskill` 筛选逻辑改动
- 已确认流水的手续费重算工具（可后续加）

## 2. 已确认偏好

| 项 | 选择 |
|----|------|
| 表结构 | 单表 `fund_trades`，`side ∈ {buy, sell, init}` |
| 初始值 | 补录持仓快照（份额 + 成本单价），非真实买卖 |
| 多次 init | 只认最近一次；其后买卖在其上累加 |
| 买卖录入 | 预委托：买=金额+确认日；卖=份额+确认日 |
| 净值回填 | 打开 UI / Refresh 时自动确认到期 pending |
| 申购费 | 金额分档；内扣：`shares = (amount − fee) / nav` |
| 赎回费 | 持有天数分档；FIFO 匹配历史 init/buy 批次 |
| 费率维护 | FundCharts 内按基金编辑分档 |
| UI | 扩展底部 `manipulate_panel` 动态表单 |
| 方案 | A：扩展 trades + FundNetv API + 底部动态表单 |

## 3. 数据模型

### 3.1 `fund_trades`

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INTEGER PK AUTOINCREMENT | |
| `fundcode` | TEXT NOT NULL | 6 位基金代码 |
| `side` | TEXT NOT NULL | `buy` / `sell` / `init` |
| `status` | TEXT NOT NULL | `pending` / `confirmed` |
| `amount` | REAL NULL | 金额（元）。买：录入必填；卖：确认后=`shares×nav`；初始：=`shares×nav` |
| `shares` | REAL NULL | 份额。卖/初始：录入必填；买：确认后内扣计算 |
| `nav` | REAL NULL | 买/卖：确认日 `unfixedvalue`；初始：成本单价 |
| `fee` | REAL NULL | 手续费。pending 为空；confirmed 后写入；init=0 |
| `settle_date` | TEXT NOT NULL | `YYYY-MM-DD`。买卖=净值确认日；初始=录入日（可改） |
| `note` | TEXT | 可选 |
| `created_at` | TEXT NOT NULL | 录入时间 ISO |
| `confirmed_at` | TEXT NULL | 回填完成时间 |

**写入形态**
- `init`：直接 `confirmed`；必填 `shares`、`nav`（成本单价）；`amount = shares×nav`；`fee = 0`
- `buy` pending：必填 `amount`、`settle_date`；`nav`/`shares`/`fee` 空
- `sell` pending：必填 `shares`、`settle_date`；`nav`/`amount`/`fee` 空

**约束（应用层 + 建议 CHECK）**
- `side IN ('buy','sell','init')`
- `status IN ('pending','confirmed')`
- confirmed 后：`shares > 0`、`nav > 0`、`amount > 0`、`fee >= 0`
- 索引：`(fundcode, settle_date, id)`、`(status, settle_date)`

### 3.2 `fund_fee_rules`

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INTEGER PK | |
| `fundcode` | TEXT NOT NULL | |
| `side` | TEXT NOT NULL | `buy` / `sell` |
| `min_value` | REAL NOT NULL | 档位下限（含）。买=申购金额（元）；卖=持有天数 |
| `max_value` | REAL NULL | 上限（不含）；NULL=无穷 |
| `rate` | REAL NOT NULL | 费率，如 0.0015 = 0.15% |
| `note` | TEXT | |
| `updated_at` | TEXT | |

同基金同 `side` 区间不得重叠。缺规则时该笔费率按 **0**，UI 提示「无费率规则」。

### 3.3 DDL 落点

- 更新 `DB_Schema.sql` 附录本两表
- 运行时 `ensure_position_tables()`：`CREATE TABLE IF NOT EXISTS` + 索引（兼容已有 `fund.db`）

## 4. 确认与计费逻辑

### 4.1 `confirm_due_trades()`

触发：`MainUI` 启动、`Refresh` 回调开始时。

对每条 `status='pending'` 且 `settle_date <= 今天`：
1. 查 `fund_values`：`fundcode` + `vdate = settle_date` 的 `unfixedvalue`
2. 若无净值：保持 pending（可汇总提示）
3. 若有净值：写入 `nav`，按 side 算份额/金额与 `fee`，`status='confirmed'`，写 `confirmed_at`

### 4.2 申购费（内扣）

1. 用录入 `amount` 在 `fund_fee_rules`（side=buy）落档得 `rate`
2. 内扣：`fee = amount × rate / (1 + rate)`（当 rate 为外示费率且费从本金扣时的常见形式；实现时用恒等式校验 `fee = amount − shares×nav`）
3. `shares = (amount − fee) / nav`
4. 存库：`amount` 仍为用户付出总金额；`fee`、`shares`、`nav` 一并写入

### 4.3 赎回费（FIFO 持有天数）

1. 构建该基金 confirmed 批次队列：`init`/`buy` 按 `(settle_date, id)` 升序，份额为可用余额（减去更早 confirmed sell 已消耗部分）
2. 对本笔 sell 的 `shares`，从队头扣减；每段 `days = settle_date_sell − settle_date_lot`（按日差，非负）
3. 用 `days` 在 side=sell 规则落档得 `rate`；`fee_seg = shares_seg × nav × rate`
4. `fee = Σ fee_seg`；`amount = shares × nav`（毛赎回额）；净到手 = `amount − fee`（派生，可不单列）

### 4.4 当前持仓份额

仅统计 **confirmed**：
```
baseline = 最近一笔 init 的 shares（按 settle_date, id）；若无则 0
lots after = 该 init 之后的 confirmed buy/sell
current = baseline + Σ buy.shares − Σ sell.shares
```
若无任何 init：`current = Σ buy.shares − Σ sell.shares`（允许；UI 可提示建议补 init）。

`current > 0` 时：建议将 `fund_meta.holding` 置 1。  
`current == 0`：**不**自动改 holding（避免误关）；UI 可提示。

卖出份额不得超过确认后的 `current`（确认时校验；pending 阶段可只警告）。

## 5. UI（`FundCharts.py`）

扩展底部 `manipulate_panel`（保留现有 PullData / AddFund / holding 等）：

| 控件 | 行为 |
|------|------|
| 类型下拉 | 买入 / 卖出 / 初始 |
| 动态字段 | 买：金额、确认日；卖：份额、确认日；初始：份额、成本单价 |
| 备注 | 可选 |
| 保存仓位 | 校验后 `add_trade`；默认 `fundcode`=当前选中基金 |
| 费率 | 对话框编辑当前基金买/卖分档（增删改） |
| 流水（建议） | 只读列表：该基金最近流水（status / fee / 日期） |

默认确认日=今天。保存成功清空可变字段并保留类型。

## 6. 模块边界

| 模块 | 职责 |
|------|------|
| `FundNetv.py` | DDL、CRUD、确认回填、FIFO、算费、`current_shares` |
| `FundCharts.py` | 表单、费率对话框、流水只读、触发 confirm |
| `DB_Schema.sql` | 文档化 DDL |
| 可选小测 | 对 FIFO/内扣/落档的纯函数单测（不强制本阶段 GUI 测） |

不把算费写进 UI 事件里；UI 只调 `FundNetv`。

## 7. 错误与边界

- 确认日净值缺失：保持 pending，Refresh 后提示数量
- 无费率规则：fee=0 + 提示
- 卖出超额：确认失败并提示
- 费率区间重叠：保存规则时拒绝
- 已 confirmed 记录：本阶段不因改费率回算

## 8. 验收标准

1. 能录入 init / pending 买 / pending 卖并写入 `fund_trades`
2. 库中有确认日 `unfixedvalue` 时，打开或 Refresh 后自动 confirmed，且 buy/sell 的 nav、派生字段、fee 正确
3. 申购内扣份额与赎回 FIFO 分档费与手工验算一致
4. 能在 UI 为当前基金编辑费率分档并影响**之后**的新确认
5. `current_shares` 与「最近 init + 其后买卖」一致
6. 不改动 ANN 训练与筛选主路径

## 9. 实现顺序（写入 plan 前的草图）

1. DDL + `ensure_position_tables`
2. FundNetv：trades CRUD、confirm、fee、FIFO、current_shares
3. FundCharts：动态表单 + 保存
4. 费率编辑对话框
5. 流水只读（可与 3 同批）
6. 手工/脚本验收上述用例

## 10. 相对 2026-07-15 设计的差异

| 7/15 | 本设计 |
|------|--------|
| buy/sell 录入即完整 amount/shares/nav | 买卖先 pending，确认日回填 |
| 无 init / fee / status | 增加 init、status、fee、settle_date、confirmed_at |
| 另有 cash ledger | 本阶段不做 |
| 服务于 position_sizing | 本阶段只服务记账与 UI；sizing 可后续接同一 `fund_trades` |
