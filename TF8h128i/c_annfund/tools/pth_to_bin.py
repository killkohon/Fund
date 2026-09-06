#!/usr/bin/env python3
"""Convert PyTorch ann/*.pth <-> C ann/*.bin (little-endian, float32)."""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

MAGIC = 0x424E4E41  # 'ANNB'
VERSION = 1
NAME_LEN = 64

EXPECTED = [
    ("embedding.weight", 2),
    ("embedding.bias", 1),
    ("fc.weight", 2),
    ("fc.bias", 1),
]


def _layer_names(i: int) -> list[tuple[str, int]]:
    p = f"transformer.layers.{i}"
    return [
        (f"{p}.self_attn.in_proj_weight", 2),
        (f"{p}.self_attn.in_proj_bias", 1),
        (f"{p}.self_attn.out_proj.weight", 2),
        (f"{p}.self_attn.out_proj.bias", 1),
        (f"{p}.linear1.weight", 2),
        (f"{p}.linear1.bias", 1),
        (f"{p}.linear2.weight", 2),
        (f"{p}.linear2.bias", 1),
        (f"{p}.norm1.weight", 1),
        (f"{p}.norm1.bias", 1),
        (f"{p}.norm2.weight", 1),
        (f"{p}.norm2.bias", 1),
    ]


def _code_from_path(path: Path) -> str:
    return path.stem[:6]


def write_bin(path: Path, code: str, tensors: dict, insize=128, hidden=256,
              outsize=3, nlayers=2, nhead=8, d_ff=2048) -> None:
    names = list(EXPECTED)
    for i in range(nlayers):
        names.extend(_layer_names(i))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", VERSION))
        f.write(code.encode("ascii")[:6].ljust(8, b"\0"))
        f.write(struct.pack("<6i", insize, hidden, outsize, nlayers, nhead, d_ff))
        f.write(struct.pack("<I", len(names)))
        for name, _ndim_hint in names:
            val = tensors[name]
            if hasattr(val, "detach"):
                arr = val.detach().cpu().float().contiguous().numpy()
            else:
                import numpy as np
                arr = np.asarray(val, dtype=np.float32)
            dims = list(arr.shape) + [0] * (4 - arr.ndim)
            raw = arr.tobytes(order="C")
            f.write(name.encode("ascii").ljust(NAME_LEN, b"\0"))
            f.write(struct.pack("<i", arr.ndim))
            f.write(struct.pack("<4i", *dims[:4]))
            f.write(struct.pack("<i", len(raw)))
            f.write(raw)


def read_bin(path: Path) -> tuple[dict, dict]:
    import numpy as np

    with path.open("rb") as f:
        magic, ver = struct.unpack("<II", f.read(8))
        if magic != MAGIC or ver != VERSION:
            raise ValueError(f"bad header magic={magic:#x} ver={ver}")
        code = f.read(8).split(b"\0", 1)[0].decode("ascii")
        insize, hidden, outsize, nlayers, nhead, d_ff = struct.unpack("<6i", f.read(24))
        (nt,) = struct.unpack("<I", f.read(4))
        meta = dict(code=code, insize=insize, hidden=hidden, outsize=outsize,
                    nlayers=nlayers, nhead=nhead, d_ff=d_ff)
        tensors = {}
        for _ in range(nt):
            name = f.read(NAME_LEN).split(b"\0", 1)[0].decode("ascii")
            (ndim,) = struct.unpack("<i", f.read(4))
            dims = list(struct.unpack("<4i", f.read(16)))[:ndim]
            (nbytes,) = struct.unpack("<i", f.read(4))
            raw = f.read(nbytes)
            tensors[name] = np.frombuffer(raw, dtype=np.float32).reshape(dims)
        return meta, tensors


def pth_to_bin(src: Path, dst: Path | None = None) -> Path:
    import torch

    try:
        sd = torch.load(src, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(src, map_location="cpu")
    # strip possible prefix
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    dst = dst or src.with_suffix(".bin")
    write_bin(dst, _code_from_path(src), sd)
    return dst


def _np_to_torch(arr):
    """NumPy -> Tensor without torch.from_numpy (breaks on NumPy 2 vs older torch ABI)."""
    import numpy as np
    import torch

    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return torch.frombuffer(bytearray(arr.tobytes()), dtype=torch.float32).clone().reshape(arr.shape)


def bin_to_pth(src: Path, dst: Path | None = None) -> Path:
    import torch

    _meta, tensors = read_bin(src)
    sd = {k: _np_to_torch(v) for k, v in tensors.items()}
    dst = dst or src.with_suffix(".pth")
    torch.save(sd, dst)
    return dst


def check_pair(pth: Path, bin_path: Path) -> float:
    import torch
    import numpy as np

    sd = torch.load(pth, map_location="cpu", weights_only=True)
    _, tensors = read_bin(bin_path)
    max_abs = 0.0
    for k, v in tensors.items():
        if k not in sd:
            raise KeyError(k)
        diff = np.max(np.abs(v - sd[k].detach().cpu().float().numpy()))
        max_abs = max(max_abs, float(diff))
    return max_abs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="*", help=".pth / .bin files, or a directory")
    ap.add_argument("--all", action="store_true", help="convert every .pth in dir")
    ap.add_argument("--to-pth", action="store_true", help="bin -> pth")
    ap.add_argument("--check", nargs=2, metavar=("PTH", "BIN"))
    ap.add_argument("-o", "--out", type=Path)
    args = ap.parse_args()

    if args.check:
        d = check_pair(Path(args.check[0]), Path(args.check[1]))
        print(f"max abs diff: {d}")
        return 0 if d == 0.0 else 1

    paths: list[Path] = []
    if args.all:
        root = Path(args.inputs[0]) if args.inputs else Path("./ann")
        paths = sorted(root.glob("*.pth" if not args.to_pth else "*.bin"))
    else:
        paths = [Path(p) for p in args.inputs]
    if not paths:
        ap.print_help()
        return 2

    for src in paths:
        if args.to_pth:
            dst = bin_to_pth(src, args.out if len(paths) == 1 else None)
        else:
            dst = pth_to_bin(src, args.out if len(paths) == 1 else None)
        print(f"{src} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
