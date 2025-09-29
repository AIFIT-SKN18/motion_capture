#!/usr/bin/env python3
"""
ctvm_sanitize.py
Convert skeleton npy arrays to a clean (C, T, V, M) shape by removing any leading batch/copy axis.

Supported inputs:
- (K, C, T, V, M): keep one K (default: first) or reduce across K
- (C, T, V, M): pass-through
- (C, T, V): add M=1
- (1, C, T, V, M): squeeze leading 1 (treated as K=1)

Optional: enforce fixed T by left-aligned zero pad / front crop to match training.
"""

import argparse, sys, os
from pathlib import Path
import numpy as np

def pick_k_slice(x5: np.ndarray, strategy: str = "first") -> np.ndarray:
    """
    Select/aggregate along leading axis K for shape (K, C, T, V, M).
    Returns (C, T, V, M).
    """
    assert x5.ndim == 5, f"expected 5D, got {x5.shape}"
    K, C, T, V, M = x5.shape
    if strategy == "first":
        return x5[0]
    elif strategy == "mean":
        return x5.mean(axis=0)
    elif strategy == "max_energy":
        var_per_k = x5.reshape(K, -1).var(axis=1)
        k = int(var_per_k.argmax())
        return x5[k]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def to_ctvm(arr: np.ndarray, k_strategy: str = "first") -> np.ndarray:
    """
    Normalize array to (C,T,V,M). Copies data if needed.
    """
    if arr.ndim == 5:
        # (K,C,T,V,M) or (1,C,T,V,M) -> resolve leading K
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = pick_k_slice(arr, strategy=k_strategy)
    elif arr.ndim == 4:
        # assume already (C,T,V,M)
        pass
    elif arr.ndim == 3:
        # (C,T,V) -> add M=1
        C, T, V = arr.shape
        arr2 = np.zeros((C, T, V, 1), dtype=arr.dtype)
        arr2[:, :, :, 0] = arr
        arr = arr2
    else:
        raise ValueError(f"Unsupported shape {arr.shape}: need 3D, 4D or 5D array")
    return arr

def left_pad_or_front_crop(x: np.ndarray, target_T: int) -> np.ndarray:
    """
    Left-align zero pad when T < target_T; front-crop when T > target_T.
    x: (C,T,V,M)
    """
    assert x.ndim == 4, f"Expected (C,T,V,M), got {x.shape}"
    C, T, V, M = x.shape
    if target_T is None or target_T <= 0:
        return x
    if T == target_T:
        return x
    if T < target_T:
        y = np.zeros((C, target_T, V, M), dtype=x.dtype)
        y[:, :T] = x
        return y
    # T > target_T
    return x[:, :target_T]

def process_file(src: Path, dst: Path, k_strategy: str, target_T: int, overwrite: bool) -> str:
    try:
        if dst.exists() and not overwrite:
            return f"[SKIP] exists: {dst}"
        arr = np.load(src, allow_pickle=False)
        arr = to_ctvm(arr, k_strategy=k_strategy)
        arr = left_pad_or_front_crop(arr, target_T=target_T)
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, arr)
        return f"[OK] {src.name} -> {dst.name} shape={arr.shape}"
    except Exception as e:
        return f"[ERR] {src}: {e}"

def main():
    ap = argparse.ArgumentParser(description="Sanitize NPY to (C,T,V,M) by removing leading K/N axis")
    ap.add_argument("--in_dir", required=True, help="Root directory to scan for .npy")
    ap.add_argument("--out_dir", required=True, help="Destination root for sanitized files")
    ap.add_argument("--pattern", default="*.npy", help="Glob pattern (default: *.npy)")
    ap.add_argument("--k_strategy", default="first", choices=["first","mean","max_energy"],
                    help="How to resolve K dimension when present (default: first)")
    ap.add_argument("--window_T", type=int, default=300, help="Target T (<=0 means keep original)")
    ap.add_argument("--overwrite", type=int, default=0, help="1 to overwrite outputs if exist")
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    overwrite = bool(int(args.overwrite))

    if not in_dir.exists():
        print(f"[FATAL] in_dir not found: {in_dir}", file=sys.stderr)
        sys.exit(2)

    paths = sorted(in_dir.rglob(args.pattern))
    if not paths:
        print(f"[WARN] no files matched under {in_dir} with pattern {args.pattern}")
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        futs = []
        for src in paths:
            rel = src.relative_to(in_dir)
            dst = out_dir / rel
            futs.append(ex.submit(process_file, src, dst, args.k_strategy, args.window_T, overwrite))
        for f in as_completed(futs):
            results.append(f.result())

    ok = sum(1 for r in results if r.startswith("[OK]"))
    err = sum(1 for r in results if r.startswith("[ERR]"))
    skip = sum(1 for r in results if r.startswith("[SKIP]"))
    for r in results:
        print(r)
    print(f"\n[SUMMARY] ok={ok} skip={skip} err={err} total={len(results)}")

if __name__ == "__main__":
    main()
