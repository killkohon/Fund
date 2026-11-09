# c_annfund_win

GNU C + CUDA 复刻 `annfund.py` 训练主程的 **原生 Windows** 版本。目标环境是 **Windows 10/11 + NVIDIA GPU**（`nvcc` + cuBLAS）。权重转换与对照工具复用 `../c_annfund/tools/`。

## 依赖

- Visual Studio 2022 或更新（本机实测 VS 18 / MSVC），工作负载 **使用 C++ 的桌面开发**
- **CUDA Toolkit**（`nvcc`、cuBLAS）。驱动 CUDA 13.0 时，Toolkit 12.4+ 且支持 `sm_75`（GTX 1650）即可
  - 下载：https://developer.nvidia.com/cuda-downloads
  - 装完后重新打开终端，确认 `nvcc --version` 可用
- CMake 3.18+
- 转换现有权重：本机 Python + PyTorch（只需离线跑 `c_annfund/tools/pth_to_bin.py`）
- sqlite 已编进 exe（amalgamation 3.46.1），不需要系统 sqlite DLL

未安装 CUDA Toolkit 时，CMake 仍可编 `test_compat`，但会跳过 `annfund.exe`。

## 编译

在仓库根目录（推荐，VS 生成器 + MSBuild）：

```text
cmake -S c_annfund_win -B c_annfund_win/build -G "Visual Studio 17 2022" -A x64
cmake --build c_annfund_win/build --config Release
```

**不要**在 Visual Studio 的 `CMakeSettings.json` 里同时写 `"generator": "Ninja"` 和 `cmakeCommandArgs: -G "Visual Studio 17 2022"`。那会让 CMake 用 `ninja.exe` 去编 `.vcxproj`，报 `loading 'build.ninja'`。从 VS 打开本目录时，选 `x64-Release`（纯 Ninja）即可。

命令行配置前请确认 `CUDA_PATH` 指向 Toolkit（安装器通常会写好）。若 MSBuild 报 `CudaToolkitDir '' does not exist`：

```text
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
```

然后删掉 `c_annfund_win/build` 再 cmake。

若 VS CMake 配置失败，删掉 `c_annfund_win/out` 后再点一次 Configure。

产物：

- `c_annfund_win\build\Release\test_compat.exe` — 线程/目录扫描冒烟
- `c_annfund_win\build\Release\annfund.exe` — 训练/推理主程（需 Toolkit）

运行时工作目录应是仓库根（能读到 `./fund.db` 和 `./ann/`）：

```text
cd D:\path\to\TF8h128i
python c_annfund\tools\pth_to_bin.py --all .\ann
c_annfund_win\build\Release\annfund.exe 6 1 20001
```

## 权重

C 侧只读写 `ann/XXXXXX.bin`，不能直接读 PyTorch pickle。转换脚本在 CUDA Linux 树中：

```text
python c_annfund\tools\pth_to_bin.py .\ann\001245.pth
python c_annfund\tools\pth_to_bin.py --all .\ann
python c_annfund\tools\pth_to_bin.py --to-pth .\ann\001245.bin
python c_annfund\tools\pth_to_bin.py --check .\ann\001245.pth .\ann\001245.bin
```

`.bin`：magic `ANNB`、version=1、基金代码、形状头、按 PyTorch `state_dict` 名存储的 float32 张量。

## 命令行

与 Linux `c_annfund` / `annfund.py` 对齐：

```text
annfund.exe <taskCount> <thisTask> [epoch] [r|s]
annfund.exe --predict 001245
```

交互命令：`quit` `ls/list` `reverse` `shuffle` `ascend` `descend` `+thread` `-thread` `status` `refuel` `defuel` `init` `epoch` `p`，以及 `XXXXXX` / `-XXXXXX` / `<XXXXXX` / `>XXXXXX`。

同时只跑 **1 个 CUDA 训练工人**（`gpu_mu` 串行）。`+thread` 仍可排队；第二工人会打印 `waiting for GPU`。

## 推理对照

从 Linux GPU 机器导出：

```text
./c_annfund/annfund --predict 001245 > pred_linux_001245.txt
```

在本机仓库根：

```text
powershell -File c_annfund_win\tools\compare_predict.ps1 -Code 001245 -LinuxPredictFile .\pred_linux_001245.txt
```

相对误差目标约 `1e-3`（非 bit-exact）。没有 Linux 输出时，可旁证：

```text
python c_annfund\tools\compare_calculate.py 001245
```

## 训练语义

- 输入：`unfixedvalue` 最近 800 条；窗口 126 + 2 个日期特征（相对 2020-01-01 的年数、weekday/6）
- 输出：未来 20 日 max / mean / min
- 网络：`Linear(128→256)` + 位置编码 + 简化 Transformer encoder（2 层、8 头、dropout=0.1）+ `Linear(256→3)`
- 优化：AdamW（默认 lr `1e-6`，`weight_decay=1e-4`）、CosineAnnealing、L1 mean、grad clip 1.0、batch 360
- 存盘：`ep>1000 && ep%5000==1`；loss 低于 0.0003 且 `ep>1000` 提前停

训练异常后会 `cudaDeviceSynchronize` + `cudaGetLastError`，必要时 `cudaDeviceReset` 并要求下次任务重建模型。
