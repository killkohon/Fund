#!/usr/bin/env python3
"""Host-only check: .bin write/read roundtrip and SerialDataset date features."""
from __future__ import annotations

import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pth_to_bin import _np_to_torch, read_bin, write_bin  # noqa: E402


def test_bin_roundtrip(tmp: Path) -> None:
    hidden, inn, out, ff = 256, 128, 3, 2048
    tensors = {
        "embedding.weight": np.linspace(0, 1, hidden * inn, dtype=np.float32).reshape(hidden, inn),
        "embedding.bias": np.arange(hidden, dtype=np.float32),
        "fc.weight": np.ones((out, hidden), dtype=np.float32),
        "fc.bias": np.array([0.1, 0.2, 0.3], dtype=np.float32),
    }
    for i in range(2):
        p = f"transformer.layers.{i}"
        tensors[f"{p}.self_attn.in_proj_weight"] = np.full((3 * hidden, hidden), 0.01 * (i + 1), np.float32)
        tensors[f"{p}.self_attn.in_proj_bias"] = np.zeros(3 * hidden, np.float32)
        tensors[f"{p}.self_attn.out_proj.weight"] = np.eye(hidden, dtype=np.float32)
        tensors[f"{p}.self_attn.out_proj.bias"] = np.zeros(hidden, np.float32)
        tensors[f"{p}.linear1.weight"] = np.full((ff, hidden), 0.02, np.float32)
        tensors[f"{p}.linear1.bias"] = np.zeros(ff, np.float32)
        tensors[f"{p}.linear2.weight"] = np.full((hidden, ff), 0.03, np.float32)
        tensors[f"{p}.linear2.bias"] = np.zeros(hidden, np.float32)
        tensors[f"{p}.norm1.weight"] = np.ones(hidden, np.float32)
        tensors[f"{p}.norm1.bias"] = np.zeros(hidden, np.float32)
        tensors[f"{p}.norm2.weight"] = np.ones(hidden, np.float32)
        tensors[f"{p}.norm2.bias"] = np.zeros(hidden, np.float32)

    path = tmp / "001245.bin"
    write_bin(path, "001245", tensors)
    meta, got = read_bin(path)
    assert meta["code"] == "001245"
    assert meta["insize"] == 128 and meta["hidden"] == 256
    assert meta["nlayers"] == 2 and meta["d_ff"] == 2048
    assert set(got) == set(tensors)
    for k, v in tensors.items():
        if np.max(np.abs(got[k] - v)) != 0:
            raise AssertionError(k)
    print("bin roundtrip: ok", path)


def test_np_to_torch_frombuffer_view() -> None:
    import torch

    raw = np.array([[1.5, -2.0], [3.25, 0.0]], dtype=np.float32).tobytes()
    view = np.frombuffer(raw, dtype=np.float32).reshape(2, 2)
    t = _np_to_torch(view)
    assert t.shape == (2, 2) and t.dtype == torch.float32
    if float(torch.max(torch.abs(t - torch.tensor([[1.5, -2.0], [3.25, 0.0]])))) != 0.0:
        raise AssertionError("np->torch mismatch")
    print("np_to_torch frombuffer view: ok")


def test_bin_to_pth_roundtrip(tmp: Path) -> None:
    import torch
    from bin_to_pth import bin_to_pth

    tensors = {
        "embedding.weight": np.linspace(0, 1, 256 * 128, dtype=np.float32).reshape(256, 128),
        "embedding.bias": np.arange(256, dtype=np.float32),
        "fc.weight": np.ones((3, 256), dtype=np.float32),
        "fc.bias": np.array([0.1, 0.2, 0.3], dtype=np.float32),
    }
    for i in range(2):
        p = f"transformer.layers.{i}"
        tensors[f"{p}.self_attn.in_proj_weight"] = np.full((768, 256), 0.01 * (i + 1), np.float32)
        tensors[f"{p}.self_attn.in_proj_bias"] = np.zeros(768, np.float32)
        tensors[f"{p}.self_attn.out_proj.weight"] = np.eye(256, dtype=np.float32)
        tensors[f"{p}.self_attn.out_proj.bias"] = np.zeros(256, np.float32)
        tensors[f"{p}.linear1.weight"] = np.full((2048, 256), 0.02, np.float32)
        tensors[f"{p}.linear1.bias"] = np.zeros(2048, np.float32)
        tensors[f"{p}.linear2.weight"] = np.full((256, 2048), 0.03, np.float32)
        tensors[f"{p}.linear2.bias"] = np.zeros(256, np.float32)
        tensors[f"{p}.norm1.weight"] = np.ones(256, np.float32)
        tensors[f"{p}.norm1.bias"] = np.zeros(256, np.float32)
        tensors[f"{p}.norm2.weight"] = np.ones(256, np.float32)
        tensors[f"{p}.norm2.bias"] = np.zeros(256, np.float32)

    src = tmp / "001245.bin"
    write_bin(src, "001245", tensors)
    dst = bin_to_pth(src)
    assert dst == src.with_suffix(".pth")
    sd = torch.load(dst, map_location="cpu", weights_only=True)
    assert set(sd) == set(tensors)
    for k, v in tensors.items():
        got = sd[k].detach().cpu().float().numpy()
        if np.max(np.abs(got - v)) != 0:
            raise AssertionError(k)
    print("bin_to_pth roundtrip: ok", dst)


def test_date_features() -> None:
    dates = ["2020-01-01", "2020-01-06", "2024-02-29", "2026-09-04"]
    base = datetime(2020, 1, 1)
    parsed = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    deltas = np.array([(d - base).days for d in parsed], dtype=np.float32) / 365.25
    weekdays = np.array([d.weekday() for d in parsed], dtype=np.float32) / 6.0
    assert dates[0] == "2020-01-01"
    assert abs(float(deltas[0])) < 1e-7
    assert abs(float(weekdays[0]) - (2 / 6.0)) < 1e-6  # 2020-01-01 Wednesday
    assert abs(float(weekdays[1])) < 1e-6  # 2020-01-06 Monday
    print("date features:", list(zip(dates, deltas.tolist(), weekdays.tolist())))
    print("date features: ok")


def test_serial_window() -> None:
    isize, osize = 126, 20
    n = isize + osize + 3
    serial = np.arange(n, dtype=np.float32) + 1.0
    dates = [(datetime(2021, 1, 1) + __import__("datetime").timedelta(days=i)).strftime("%Y-%m-%d")
             for i in range(n)]
    base = datetime(2020, 1, 1)
    parsed = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    deltas = np.array([(d - base).days for d in parsed], dtype=np.float32) / 365.25
    weekdays = np.array([d.weekday() for d in parsed], dtype=np.float32) / 6.0
    ns = n - isize - osize + 1
    x = np.empty((ns, isize + 2), np.float32)
    y = np.empty((ns, 3), np.float32)
    for i in range(ns):
        x[i, :isize] = serial[i:i + isize]
        x[i, isize] = deltas[i + isize - 1]
        x[i, isize + 1] = weekdays[i + isize - 1]
        out = serial[i + isize:i + isize + osize]
        y[i] = [out.max(), out.mean(), out.min()]
    assert x.shape == (ns, 128)
    assert np.allclose(x[0, :isize], serial[:isize])
    assert y[0, 0] == serial[isize:isize + osize].max()
    print(f"serial window: {ns} samples ok")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="annbin_"))
    try:
        test_bin_roundtrip(tmp)
        test_np_to_torch_frombuffer_view()
        test_bin_to_pth_roundtrip(tmp)
        test_date_features()
        test_serial_window()
        print("all host checks passed")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
