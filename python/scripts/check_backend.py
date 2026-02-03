#!/usr/bin/env python3
import argparse
import os
import sys

import torch

CUDA_TOKENS = ("nvidia", "geforce", "quadro", "tesla", "rtx")
ROCM_TOKENS = ("amd", "radeon", "instinct", "mi", "gfx")


def matches_backend(name: str, backend: str) -> bool:
    lower = name.lower()
    if backend == "cuda":
        return any(token in lower for token in CUDA_TOKENS)
    return any(token in lower for token in ROCM_TOKENS) and not any(token in lower for token in CUDA_TOKENS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-index", action="store_true", help="Print only the selected device index")
    args = parser.parse_args()

    backend = os.environ.get("BACKEND", "cpu").lower()
    if backend not in ("cuda", "rocm"):
        print(f"ERROR: invalid BACKEND for smoke test: {backend}")
        return 1

    if backend == "cuda":
        if not torch.version.cuda:
            print("ERROR: torch.version.cuda is None")
            return 1
        if torch.version.hip:
            print(f"ERROR: torch.version.hip is set ({torch.version.hip})")
            return 1
    else:
        if not torch.version.hip:
            print("ERROR: torch.version.hip is None")
            return 1
        if torch.version.cuda:
            print(f"ERROR: torch.version.cuda is set ({torch.version.cuda})")
            return 1

    count = torch.cuda.device_count()
    if count == 0 or not torch.cuda.is_available():
        print("SKIP: torch.cuda.is_available() is False")
        return 2

    forced = os.environ.get("BACKEND_DEVICE_INDEX")
    indices = list(range(count))
    if forced is not None:
        try:
            indices = [int(forced)]
        except ValueError:
            print(f"ERROR: BACKEND_DEVICE_INDEX must be an integer, got: {forced}")
            return 1

    names = [torch.cuda.get_device_name(i) for i in range(count)]
    selected = None
    for idx in indices:
        if idx < 0 or idx >= count:
            continue
        if matches_backend(names[idx], backend):
            selected = idx
            break

    if selected is None:
        print(f"ERROR: no matching {backend} device found. Detected: {names}")
        return 1

    if args.print_index:
        print(selected)
    else:
        print(f"GPU available (device {selected}: {names[selected]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
