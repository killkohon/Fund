#!/usr/bin/env python3
"""Convert C ann/*.bin -> PyTorch ann/*.pth (little-endian, float32)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pth_to_bin import bin_to_pth, check_pair


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="*", help=".bin files, or a directory")
    ap.add_argument("--all", action="store_true", help="convert every .bin in dir")
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
        paths = sorted(root.glob("*.bin"))
    else:
        paths = [Path(p) for p in args.inputs]
    if not paths:
        ap.print_help()
        return 2

    for src in paths:
        dst = bin_to_pth(src, args.out if len(paths) == 1 else None)
        print(f"{src} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
