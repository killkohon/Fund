# c_annfund_cpu

GNU C + OpenBLAS 复刻 `annfund.py` 训练主程的 **纯 CPU** 版本。目标环境是 **Linux**（无需 CUDA）。权重转换与对照工具复用 `../c_annfund/tools/`。

## 依赖

- gcc / pthread / `sqlite-devel`（或 `libsqlite3-dev`）
- **BLAS（二选一）**
  - **推荐**：系统 OpenBLAS（更快）。注意：OpenBLAS 头会 `#define I`（虚数单位），`model.c` 已在 include 后 `#undef I`。
    - Debian/Ubuntu：`apt-get install -y libopenblas-dev`
    - RHEL/Rocky/Alma 9：默认源常无此包，需 EPEL；或已有 `/usr/include/openblas/cblas.h` 时可直接 `make BLAS=openblas`

## 编译

```bash
cd c_annfund_cpu
make              # 默认 OpenBLAS
make BLAS=builtin # 强制内置（EL9 无 openblas-devel 时用这个）
make BLAS=auto    # 有 cblas.h 用 OpenBLAS，否则 builtin
```

运行时工作目录应是仓库根（能读到 `./fund.db` 和 `./ann/`）：

```bash
cd /path/to/TF8h128i
python3 c_annfund/tools/pth_to_bin.py --all ./ann
c_annfund_cpu/annfund 6 1 20001
```

## 权重

C 侧只读写 `ann/XXXXXX.bin`，不能直接读 PyTorch pickle。转换脚本在 CUDA 树中：

```bash
python3 c_annfund/tools/pth_to_bin.py ./ann/001245.pth
python3 c_annfund/tools/pth_to_bin.py --all ./ann
python3 c_annfund/tools/pth_to_bin.py --to-pth ./ann/001245.bin
python3 c_annfund/tools/pth_to_bin.py --check ./ann/001245.pth ./ann/001245.bin
```

`.bin`：magic `ANNB`、version=1、基金代码、形状头、按 PyTorch `state_dict` 名存储的 float32 张量。

## 命令行

与 `c_annfund` / `annfund.py` 对齐：

```bash
./annfund <taskCount> <thisTask> [epoch] [r|s]
./annfund --predict 001245
```

交互命令：`quit` `ls/list` `reverse` `shuffle` `ascend` `descend` `+thread` `-thread` `status` `refuel` `defuel` `init` `epoch` `p`，以及 `XXXXXX` / `-XXXXXX` / `<XXXXXX` / `>XXXXXX`。

## 并行训练

默认 **串行训练**（`train_mu` 互斥）：多个 `+thread` 工人会排队，同一时刻只有一个在 `model_train`。

设置环境变量可允许多线程同时训练（需足够 CPU/内存）：

```bash
ANN_CPU_PARALLEL=1 ./c_annfund_cpu/annfund 6 1 200
```

SQLite 访问（`db_open` / `db_close` / `db_load_nav`）由 `db_mu` 保护，并行模式下数据库读写仍串行。

**注意：** CPU 训练远慢于 CUDA 版；短 epoch 冒烟可用 `./c_annfund_cpu/annfund 1 1 200`。

## 推理对照

有 CUDA 二进制时，与 GPU 版对比：

```bash
bash c_annfund_cpu/tools/compare_predict.sh 001245
```

仅 CPU 机器时，对照 Python：

```bash
./c_annfund_cpu/annfund --predict 001245
python3 c_annfund/tools/compare_calculate.py 001245
```

相对误差目标约 `1e-3`（非 bit-exact）。

## 训练语义

- 输入：`unfixedvalue` 最近 800 条；窗口 126 + 2 个日期特征（相对 2020-01-01 的年数、weekday/6）
- 输出：未来 20 日 max / mean / min
- 网络：`Linear(128→256)` + 位置编码 + 简化 Transformer encoder（2 层、8 头、dropout=0.1）+ `Linear(256→3)`
- 优化：AdamW（默认 lr `1e-6`，`weight_decay=1e-4`）、CosineAnnealing、L1 mean、grad clip 1.0、batch 360
- 存盘：`ep>1000 && ep%20000==1`；loss 低于 0.0003 且 `ep>1000` 提前停
