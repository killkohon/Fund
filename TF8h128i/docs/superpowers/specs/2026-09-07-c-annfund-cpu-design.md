# 设计：c_annfund Linux CPU（无 GPU）复刻

日期：2026-09-07  
状态：待用户审阅  
关联：`c_annfund/`（GNU C + CUDA 复刻 `annfund.py`）

## 1. 背景与目标

现有 `c_annfund` 依赖 NVIDIA GPU（`nvcc` + cuBLAS）。目标是在 **Linux CPU（无 GPU）** 环境下提供功能对齐的训练/推理主程，可读写同一套 `ann/*.bin` 权重。

**目标**：

- 新建独立目录 `c_annfund_cpu/`，与 `c_annfund` 并存
- 纯 `gcc` 构建：OpenBLAS + sqlite3 + pthread（无 CUDA）
- CLI、训练语义、`.bin` 格式与 CUDA 版对齐
- `--predict` 相对 CUDA 版误差约 `1e-3`；短 epoch loss 可下降

**非目标（首版）**：

- 不实现完整 QK^T softmax 多头注意力（保持与现有 `model.cu` 相同的简化 encoder）
- 不提供 Windows 原生构建
- 不与 `c_annfund` 共享源码树 / 统一 Makefile 后端切换
- 不做逼近 GPU 的性能调优

## 2. 已确认偏好

| 项 | 选择 |
|----|------|
| 交付形态 | **A**：独立目录 `c_annfund_cpu/` |
| 矩阵库 | **A**：OpenBLAS / CBLAS（`cblas_sgemm`） |
| 并发 | **C**：默认串行互斥；`ANN_CPU_PARALLEL=1` 时真正并行 |
| 成功标准 | **A**：对齐 `c_annfund`（同 `.bin`、predict ~1e-3） |
| 实现路径 | **方案 1**：移植 `model.cu` → `model.c` + OpenBLAS |

## 3. 目录与模块边界

```
c_annfund_cpu/
  README.md
  Makefile                 # gcc + -lopenblas -lsqlite3 -lpthread -lm
  include/
    ann.h                  # 常量与 CUDA 版一致
    model.h                # 去掉 cublas；cuda_* → cpu_*
    db.h / dataset.h / queue.h / weights.h
  src/
    main.c                 # 复制自 c_annfund，gpu_mu → train_mu + 并行开关
    model.c                # 由 model.cu 移植（核心）
    weights.c              # cudaMemcpy → memcpy；.bin 格式不变
    db.c / dataset.c / queue.c
```

不设 `tools/` 子目录：README 写明复用 `../c_annfund/tools`（避免 Windows 软链问题）。

| 层 | 策略 |
|----|------|
| CLI / 队列 / SQLite / dataset | 复制，行为对齐 CUDA 版 |
| 权重 `.bin` | 同 magic `ANNB`、version=1，与 CUDA 互通 |
| 模型算子 | 仅 `model.c` 重写（主机内存 + OpenBLAS） |
| 工具脚本 | 复用 `c_annfund/tools`（不复制、不软链） |

工作目录仍为仓库根（可读 `./fund.db`、`./ann/`）：

```bash
cd c_annfund_cpu && make
../c_annfund_cpu/annfund 6 1 20001
../c_annfund_cpu/annfund --predict 001245
```

## 4. 模型层设计

### 4.1 内存与 API

- 权重 / 梯度 / Adam 状态 / workspace：全部 `malloc`/`calloc`
- **workspace 挂在 `Model` 上**（禁止进程级 `static Work`），以便 `ANN_CPU_PARALLEL=1` 时多模型并发安全
- 前向/反向不再做 host↔device 拷贝
- `weights.c` 用 `memcpy` 读写张量；`TensorRef` 指向主机指针
- 公开 API 与现有 `model.h` 对齐：`model_create` / `destroy` / `train` / `predict` / `forward` / `forward_bwd` / `adamw_step` / `clip_grad` / `cosine_lr` 等
- `cuda_runtime_ok` → `cpu_runtime_ok()`（恒返回成功）
- `cuda_harden_reset` → `cpu_harden_reset()`（清空/标记 workspace 失效，无 device reset）

`Model` 结构去掉 `void *cublas`；其余字段布局与权重名保持一致，便于 `weights_load/save` 共用同一套 tensor 名表。

### 4.2 BLAS

`cublasSgemm` 映射为 `cblas_sgemm`，保持相同布局约定：

- Forward：`Y(B,out) = X(B,in) @ W(out,in)^T`
- Backward：`dX = dY @ W`，`dW += dY^T @ X`

依赖包：`libopenblas-dev`（或提供 CBLAS 的兼容实现）。Makefile：`-lopenblas`。

### 4.3 算子映射

现有 CUDA encoder **不是**完整 MHA：QKV 线性 → 取 V → head dropout → out proj → residual+LN → FFN(ReLU)+dropout → residual+LN。CPU 版逐算子移植，不改变算法语义。

| CUDA | CPU |
|------|-----|
| `k_add_bias` / `k_bias_grad` / `k_add` / `k_axpy` / `k_zero` / `k_scale` | C 循环 |
| `k_relu` / `k_relu_bwd` | C 循环 |
| `k_ln_fwd` / `k_ln_bwd` | C 循环（`ANN_LN_EPS`） |
| `k_dropout` / `k_head_drop`（及 bwd） | 同 seed 规则的主机 RNG |
| `k_adamw` / `k_sqsum`（用于 clip） | C 循环 |
| `k_copy_v` / `k_scatter_v` | C 循环 |
| `k_l1_bwd` + mean L1 | C 循环 |
| `k_add_pe` | C 循环（正弦 pe0） |

训练超参与停存条件与 CUDA 版一致（见 `c_annfund/README.md`）：AdamW、CosineAnnealing、L1 mean、grad clip 1.0、batch 360、loss 阈值等。

### 4.4 数值目标

同一 `.bin` 下，CPU `--predict` 相对 CUDA `--predict` 相对误差约 `1e-3`（非 bit-exact）。OpenBLAS 与 cuBLAS 的浮点归约顺序不同，允许该量级偏差。

## 5. 并发与错误处理

### 5.1 并发

- 默认：`train_mu` 互斥，同时只训一只基金（对齐 `gpu_mu`）
- `ANN_CPU_PARALLEL=1`：跳过互斥，允许多 worker 同时训不同基金
- CLI `+thread` / `-thread` 保留；并行模式下由用户自行控制线程数
- SQLite：对 `db_load_nav` 等共享访问加短临界区 mutex，避免并行读冲突；不改 schema

### 5.2 错误处理

- 无 `cudaDeviceSynchronize` / `cudaDeviceReset`
- 训练失败：`model_destroy` + 日志；必要时 `cpu_harden_reset` 仅清 workspace
- `malloc` / BLAS 调用失败返回 `ANN_ERR`
- `quit` → `interrupt` 语义保留
- UDP loss 上报保留（失败忽略）

## 6. 验证计划

1. 无 CUDA 的 Linux 上 `make` 成功（gcc + OpenBLAS + libsqlite3）
2. 同一 `.bin`：`c_annfund_cpu/annfund --predict CODE` vs `c_annfund/annfund --predict CODE`，相对误差约 `1e-3`
3. 短 epoch（数百）loss 下降；可选对照 `c_annfund/tools/compare_calculate.py`
4. 默认串行 smoke；`ANN_CPU_PARALLEL=1` 双 worker smoke（不做性能基准）

## 7. 实现顺序（供后续 plan 展开）

1. 脚手架：目录、Makefile、复制非模型源与头文件并去 CUDA 依赖
2. `weights.c` 主机版 + `model.h` API 调整
3. `model.c`：分配/初始化 → linear/GEMM → 逐算子 → encoder fwd/bwd → train/predict
4. `main.c`：`train_mu` + `ANN_CPU_PARALLEL`
5. README 与 predict 数值对照

## 8. 风险与假设

- **假设**：现有 `model.cu` 的简化 attention 是有意为之；CPU 版不“修正”为完整 Transformer
- **已决策**：去掉 CUDA 版进程级 `static Work`，workspace 一律 per-`Model`（见 §4.1）
- **风险**：CPU 训练显著慢于 GPU；首版可接受，README 注明
