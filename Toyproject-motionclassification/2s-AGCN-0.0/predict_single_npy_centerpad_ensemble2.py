#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-NPY inference with center zero-pad/crop
+ optional second modality (e.g., joint + bone) logits-weighted ensemble
+ 5D input (K,C,T,V,M) slicing support via --five_d_index/--five_d_index2.

Usage (examples):
  # joint only (5D에서 K=0을 사용)
  python predict_single_npy_centerpad_ensemble.py \
    --model_py /path/to/model/agcn.py \
    --model_class Model \
    --model_kwargs '{"num_class":6}' \
    --ckpt /path/to/joint_ckpt.pt \
    --npy  /path/to/sample_5d_joint_bone.npy \
    --five_d_index 0 \
    --device cuda --window_T 300 --do_norm 0 --print_all 1

  # joint + bone ensemble (같은 5D 파일에서 K=0,1 사용)
  python predict_single_npy_centerpad_ensemble.py \
    --model_py /path/to/model/agcn.py \
    --model_class Model \
    --model_kwargs '{"num_class":6}' \
    --ckpt  /path/to/joint_ckpt.pt \
    --npy   /path/to/sample_5d_joint_bone.npy \
    --five_d_index 0 \
    --ckpt2 /path/to/bone_ckpt.pt \
    --npy2  /path/to/sample_5d_joint_bone.npy \
    --five_d_index2 1 \
    --weight2 1.0 \
    --device cuda --window_T 300 --do_norm 0 --print_all 1
"""
import argparse
import importlib.util
import json
import os
from types import ModuleType

import numpy as np
import torch


# -----------------------------
# Utilities
# -----------------------------
def load_module_from_path(py_path: str) -> ModuleType:
    """Dynamically load a .py file as a module."""
    py_path = os.path.abspath(py_path)
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"model_py not found: {py_path}")
    spec = importlib.util.spec_from_file_location("user_model", py_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, f"Failed to load spec for {py_path}"
    spec.loader.exec_module(module)  # type: ignore
    return module


def load_model_from_py(py_path: str, class_name: str, model_kwargs: dict):
    """Instantiate model class from python file."""
    mod = load_module_from_path(py_path)
    if not hasattr(mod, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {py_path}")
    ModelClass = getattr(mod, class_name)
    model = ModelClass(**model_kwargs)
    return model


def load_ckpt(ckpt_path: str, device: torch.device):
    """Load a checkpoint file into memory (state_dict or whole checkpoint)."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location=device)
    # Allow {"state_dict": ...} or {"model_state_dict": ...}
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
    return obj  # assume it's already a state_dict


def ensure_ctvm_shape(arr: np.ndarray) -> np.ndarray:
    """Ensure array is shaped as (C, T, V, M). If (T, V, C) or (C, T, V) given, expand M=1."""
    a = np.asarray(arr)
    if a.ndim == 4:
        # Assume already (C,T,V,M)
        return a
    elif a.ndim == 3:
        # Try to infer (C,T,V) vs (T,V,C)
        t0, t1, t2 = a.shape
        if t0 in (2, 3, 4, 6) and t1 >= 1 and t2 >= 1:
            C, T, V = a.shape
            return a.reshape(C, T, V, 1)
        if t2 in (2, 3, 4, 6) and t0 >= 1 and t1 >= 1:
            T, V, C = a.shape
            return np.transpose(a, (2, 0, 1)).reshape(C, T, V, 1)
        # Fallback: assume (C,T,V)
        return a.reshape(t0, t1, t2, 1)
    else:
        raise ValueError(f"Unsupported input shape {a.shape}; expected 3D or 4D array.")


def center_zero_pad_or_center_crop(arr: np.ndarray, target_T: int) -> np.ndarray:
    """Center align in time: if T < target pad with zeros; if T > target, center-crop."""
    C, T, V, M = arr.shape
    if T == target_T:
        return arr
    if T < target_T:
        pad_total = target_T - T
        left = pad_total // 2
        right = pad_total - left
        out = np.concatenate([np.zeros((C, left, V, M), dtype=arr.dtype),
                              arr,
                              np.zeros((C, right, V, M), dtype=arr.dtype)], axis=1)
        return out
    # T > target_T: center crop
    start = (T - target_T) // 2
    end = start + target_T
    return arr[:, start:end, :, :]


def maybe_z_norm(arr: np.ndarray, do_norm: int) -> np.ndarray:
    """If do_norm==1, apply global z-normalization per-channel over (T,V,M)."""
    if not do_norm:
        return arr
    C, T, V, M = arr.shape
    flat = arr.reshape(C, -1)
    mean = flat.mean(axis=1, keepdims=True)
    std = flat.std(axis=1, keepdims=True) + 1e-8
    normed = ((flat - mean) / std).reshape(C, T, V, M)
    return normed.astype(np.float32)


def build_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert (C,T,V,M) to (N=1,C,T,V,M)."""
    x = torch.from_numpy(arr).float().unsqueeze(0)
    return x.to(device)


def topk_from_logits(logits: torch.Tensor, k: int = 5):
    """Return top-k indices and probabilities for batch size 1."""
    if logits.ndim == 2:  # (1, C)
        probs = torch.softmax(logits[0], dim=-1)
    elif logits.ndim == 1:  # (C,)
        probs = torch.softmax(logits, dim=-1)
    else:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")
    top_prob, top_idx = torch.topk(probs, k=min(k, probs.numel()))
    return top_idx.cpu().numpy().tolist(), top_prob.cpu().numpy().tolist(), probs


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Single NPY inference (center zero-pad / center crop) with optional ensemble & 5D slicing")
    ap.add_argument("--model_py", required=True, help="Path to model .py (e.g., agcn.py)")
    ap.add_argument("--model_class", default="Model", help="Model class name in model_py")
    ap.add_argument("--model_kwargs", default="{}", help="JSON for model kwargs (e.g., '{\"num_class\":6}')")
    ap.add_argument("--ckpt", required=True, help="Checkpoint for first modality (e.g., joint)")
    ap.add_argument("--npy", required=True, help="NPY path for first modality")
    # Second modality (optional)
    ap.add_argument("--ckpt2", default=None, help="Checkpoint for second modality (e.g., bone)")
    ap.add_argument("--npy2", default=None, help="NPY path for second modality")
    ap.add_argument("--weight2", type=float, default=1.0, help="Weight for second modality in logits average")
    # 5D input slicing
    ap.add_argument("--five_d_index", type=int, default=None, help="If npy is 5D (K,C,T,V,M), pick K index for the 1st modality (default 0)")
    ap.add_argument("--five_d_index2", type=int, default=None, help="If npy2 is 5D (K,C,T,V,M), pick K index for the 2nd modality (default 1)")
    # Inference options
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--window_T", type=int, default=300, help="Target temporal length")
    ap.add_argument("--num_classes", type=int, default=None, help="Override num_class/num_classes in model kwargs")
    ap.add_argument("--do_norm", type=int, default=0, help="1 to apply z-norm")
    ap.add_argument("--print_all", type=int, default=1, help="Verbose prints")
    args = ap.parse_args()

    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Parse kwargs
    try:
        model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}
        if not isinstance(model_kwargs, dict):
            raise ValueError("model_kwargs must be a JSON object")
    except Exception as e:
        raise SystemExit(f"--model_kwargs must be JSON. Error: {e}")
    # apply num_classes override if provided
    if args.num_classes is not None:
        if "num_class" in model_kwargs:
            model_kwargs["num_class"] = args.num_classes
        elif "num_classes" in model_kwargs:
            model_kwargs["num_classes"] = args.num_classes
        else:
            model_kwargs["num_class"] = args.num_classes

    # === First modality ===
    model1 = load_model_from_py(args.model_py, args.model_class, model_kwargs).to(device)
    model1.eval()
    state1 = load_ckpt(args.ckpt, device)
    missing1, unexpected1 = model1.load_state_dict(state1, strict=False)
    if args.print_all:
        print(f"[CKPT-1] loaded: {args.ckpt}")
        if missing1 or unexpected1:
            print(f"[CKPT-1] missing={len(missing1)} unexpected={len(unexpected1)}")

    arr1 = np.load(args.npy, allow_pickle=False)
    # 5D slicing for first modality
    if arr1.ndim == 5:
        default_k = 0
        k = default_k if args.five_d_index is None else args.five_d_index
        if not (0 <= k < arr1.shape[0]):
            raise ValueError(f"--five_d_index {k} out of range for arr1 with shape {arr1.shape}")
        arr1 = arr1[k]
        if args.print_all:
            print(f"[INPUT-1] 5D detected, using K={k} -> slice shape={arr1.shape}")
    arr1 = ensure_ctvm_shape(arr1)
    if args.print_all:
        C1, T1, V1, M1 = arr1.shape
        print(f"[INPUT-1] {args.npy} shape={arr1.shape} mean={arr1.mean():.6f} std={arr1.std():.6f}")
    arr1 = center_zero_pad_or_center_crop(arr1, args.window_T)
    arr1 = maybe_z_norm(arr1, args.do_norm)
    x1 = build_tensor(arr1, device)

    with torch.no_grad():
        logits1 = model1(x1)
        if isinstance(logits1, (list, tuple)):
            logits1 = logits1[-1]

    logits = logits1  # default to single-modality

    # === Second modality (optional) ===
    has_second = (args.ckpt2 is not None) and (args.npy2 is not None)
    if has_second:
        model2 = load_model_from_py(args.model_py, args.model_class, model_kwargs).to(device)
        model2.eval()
        state2 = load_ckpt(args.ckpt2, device)
        missing2, unexpected2 = model2.load_state_dict(state2, strict=False)
        if args.print_all:
            print(f"[CKPT-2] loaded: {args.ckpt2}")
            if missing2 or unexpected2:
                print(f"[CKPT-2] missing={len(missing2)} unexpected={len(unexpected2)}")

        arr2 = np.load(args.npy2, allow_pickle=False)
        if arr2.ndim == 5:
            default_k2 = 1
            k2 = default_k2 if args.five_d_index2 is None else args.five_d_index2
            if not (0 <= k2 < arr2.shape[0]):
                raise ValueError(f"--five_d_index2 {k2} out of range for arr2 with shape {arr2.shape}")
            arr2 = arr2[k2]
            if args.print_all:
                print(f"[INPUT-2] 5D detected, using K={k2} -> slice shape={arr2.shape}")
        arr2 = ensure_ctvm_shape(arr2)
        if args.print_all:
            C2, T2, V2, M2 = arr2.shape
            print(f"[INPUT-2] {args.npy2} shape={arr2.shape} mean={arr2.mean():.6f} std={arr2.std():.6f}")
        arr2 = center_zero_pad_or_center_crop(arr2, args.window_T)
        arr2 = maybe_z_norm(arr2, args.do_norm)
        x2 = build_tensor(arr2, device)

        with torch.no_grad():
            logits2 = model2(x2)
            if isinstance(logits2, (list, tuple)):
                logits2 = logits2[-1]

        # Weighted average of logits
        logits = (logits1 + args.weight2 * logits2) / (1.0 + args.weight2)

    # === Report ===
    idxs, probs, _ = topk_from_logits(logits, k=5)
    print("\n=== Prediction ===")
    print(f"Top-1: class_id={idxs[0]}, prob={probs[0]:.4f}")
    print("Top-5:")
    for i, (ci, pi) in enumerate(zip(idxs, probs), 1):
        print(f" {i:>2}. {ci:>6} : {pi:.4f}")


if __name__ == "__main__":
    main()
