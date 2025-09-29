#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------
# Class names (6-way)
# -----------------------------
CLASS_NAMES = [
    "other",
    "dumbbell benchpress",
    "dumbbell deadlift",
    "dumbbell lunges",
    "dumbbell side lateral raise",
    "dumbbell squat",
]


# -----------------------------
# Utils: dynamic import
# -----------------------------
def load_model_from_file(py_path, class_name, kwargs):
    spec = importlib.util.spec_from_file_location("user_model", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Model = getattr(mod, class_name)
    return Model(**kwargs)


# -----------------------------
# Shape helpers
# -----------------------------
def center_crop_or_pad_T(x, target_T):
    """
    x: np.ndarray of shape (C,T,V,M)
    -> returns np.ndarray (C,target_T,V,M)
    """
    C, T, V, M = x.shape
    if T == target_T:
        return x
    if T > target_T:
        # center crop
        s = (T - target_T) // 2
        e = s + target_T
        return x[:, s:e, :, :]
    else:
        # replicate-pad at the end
        pad = target_T - T
        if pad <= 0:
            return x
        last = x[:, -1:, :, :]
        tail = np.repeat(last, pad, axis=1)
        return np.concatenate([x, tail], axis=1)


def normalize_seq(x, eps=1e-6):
    """
    x: (C,T,V,M), simple z-norm per sequence & channel
    (주의) 학습 정규화와 같은 방식이 아니라면 OFF(--do_norm 0) 권장
    """
    x = x.astype(np.float32)
    mean = x.mean(axis=(1, 2, 3), keepdims=True)
    std = x.std(axis=(1, 2, 3), keepdims=True)
    return (x - mean) / (std + eps)


def ensure_ctvm_flex(x, window_size=300, do_norm=False, modality_idx=None, verbose=True):
    """
    입력을 (C,T,V,M)으로 정규화해서 반환.
    허용 입력:
      - (C,T,V,M)
      - (1,C,T,V,M) -> squeeze
      - (2,C,T,V,M) -> 2-stream(joint,bone)에서 modality_idx로 0/1 선택
      - (N,C,T,V,M) -> 배치로 보고 첫 샘플 사용
    그 외 흔한 포맷은 미지원(필요하면 여기서 변환 추가)
    """
    orig_shape = tuple(x.shape)
    if verbose:
        print(f"[ensure_ctvm_flex] raw shape={orig_shape}", flush=True)

    arr = np.asarray(x)

    # 5D -> N 또는 streams 차원 처리
    if arr.ndim == 5:
        n0 = arr.shape[0]
        # (2,C,T,V,M): 2-stream으로 가정 (joint,bone)
        if n0 == 2 and modality_idx is not None:
            if verbose:
                stream_name = "joint" if modality_idx == 0 else "bone"
                print(f"[ensure_ctvm_flex] 2-stream detected -> using stream {modality_idx} ({stream_name})", flush=True)
            arr = arr[modality_idx]  # -> (C,T,V,M)
        else:
            # 일반 배치로 보고 첫 샘플
            if verbose:
                print(f"[ensure_ctvm_flex] treat as batch -> take the first sample (idx=0)", flush=True)
            arr = arr[0]

    # 4D: (C,T,V,M)
    if arr.ndim == 4:
        C, T, V, M = arr.shape
        # windowing
        if window_size is not None and window_size > 0:
            arr = center_crop_or_pad_T(arr, window_size)

        # optional normalization
        if do_norm:
            arr = normalize_seq(arr)

        if verbose:
            print(f"[ensure_ctvm_flex] final (before torch): shape={arr.shape}, mean={float(arr.mean()):.6f}, std={float(arr.std()):.6f}", flush=True)
        return arr

    # 5D: (1,C,T,V,M)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
        return ensure_ctvm_flex(arr, window_size, do_norm, modality_idx, verbose)

    raise ValueError(f"Unsupported input shape: {orig_shape}. Expect (C,T,V,M) or 5D variants.")


def to_torch_ctvm(arr, device):
    """
    arr: np.ndarray (C,T,V,M)
    -> torch tensor (1,C,T,V,M) on device
    """
    x = torch.from_numpy(arr.astype(np.float32))
    # model expects (N,C,T,V,M), so add batch dim
    x = x.unsqueeze(0)
    x = x.to(device)
    return x


# -----------------------------
# Inference
# -----------------------------
def run_inference(
    npy_path,
    model_py,
    model_class,
    model_kwargs,
    ckpt_path,
    device="cpu",
    window_T=300,
    do_norm=False,
    modality_idx=None,
    num_classes=6,
    apply_other_gate=False,
    other_top1_min=0.8,
    precision=6,
    print_all=True,
    save_json=None,
):
    device = torch.device(device)
    print(f"[Device] using {device.type}", flush=True)

    # 1) Build model
    model = load_model_from_file(model_py, model_class, model_kwargs)
    model.to(device)
    model.eval()

    # 2) Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state", ckpt)  # support plain state_dict or {'model_state':...}
    # try strict load
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # pretty print
    num_keys = len(state_dict.keys())
    num_backbone_like = sum(k.startswith("l") or k.startswith("data_bn") for k in state_dict.keys())
    num_fc_like = sum(k.startswith("fc") for k in state_dict.keys())
    print(f"[CKPT] keys={num_keys} | backbone_like={num_backbone_like} | fc_like={num_fc_like}")
    print(f"        samples: {list(state_dict.keys())[:3]} ...", flush=True)
    print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    # 3) Load NPY
    f = str(npy_path)
    raw = np.load(f)
    # modality from env if provided
    env_idx = os.getenv("AGCN_MODALITY_IDX", None)
    if modality_idx is None and env_idx is not None and env_idx != "":
        try:
            modality_idx = int(env_idx)
        except Exception:
            modality_idx = None

    # 4) preprocess -> (1,C,T,V,M)
    arr = ensure_ctvm_flex(
        raw, window_size=window_T, do_norm=bool(do_norm),
        modality_idx=modality_idx, verbose=True
    )
    x = to_torch_ctvm(arr, device)

    # DEBUG stats
    print(f"[DEBUG] tensor before model: shape={tuple(x.shape)}, mean={float(x.mean().item()):.{precision}f}, std={float(x.std().item()):.{precision}f}", flush=True)

    # 5) forward
    with torch.no_grad():
        logits = model(x)  # (1,num_classes)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    # 6) simple gate to class 0(other)
    pred_idx = int(probs.argmax())
    pred_prob = float(probs[pred_idx])
    if apply_other_gate and pred_prob < other_top1_min:
        pred_idx = 0
        pred_prob = float(probs[pred_idx])

    # 7) print result
    s = probs.sum()
    if print_all:
        print(f"\n[Result] file={Path(f).name}")
        cname = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class {pred_idx}"
        print(f"  predicted: {pred_idx} ({cname})")
        print(f"  sum(probs) = {s:.6f}")
        # top-k sorted
        order = probs.argsort()[::-1]
        k = min(num_classes, len(order))
        print(f"  top-{k}:")
        for r, idx in enumerate(order[:k], start=1):
            name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class {idx}"
            print(f"    #{r}: class {idx} ({name})  prob={probs[idx]:.{precision}f}")
        print("", flush=True)

    # 8) save json
    if save_json:
        out = {
            "file": f,
            "pred_idx": pred_idx,
            "pred_name": CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class {pred_idx}",
            "probs": probs.tolist(),
        }
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w") as w:
            json.dump(out, w, indent=2)
        print(f"[OUT] saved JSON -> {save_json}", flush=True)


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_py", required=True, help="path to model .py (e.g., model/agcn.py)")
    p.add_argument("--model_class", required=True, help="class name in model_py (e.g., Model)")
    p.add_argument("--model_kwargs", default="{}", help="JSON for model kwargs")
    p.add_argument("--ckpt", required=True, help="checkpoint path (.pt/.pth)")
    p.add_argument("--npy", required=True, help="input npy path")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--window_T", type=int, default=300)
    p.add_argument("--num_classes", type=int, default=6)
    p.add_argument("--do_norm", type=int, default=0, help="apply normalization (1) or not (0)")
    p.add_argument("--modality", type=int, default=None, help="0: joint, 1: bone (overrides env if set)")
    p.add_argument("--apply_other_gate", type=int, default=0)
    p.add_argument("--other_top1_min", type=float, default=0.80)
    p.add_argument("--precision", type=int, default=6)
    p.add_argument("--print_all", type=int, default=1)
    p.add_argument("--save_json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}

    # modality from CLI > ENV > None
    modality_idx = args.modality
    if modality_idx is None:
        env_idx = os.getenv("AGCN_MODALITY_IDX", None)
        if env_idx is not None and env_idx != "":
            try:
                modality_idx = int(env_idx)
            except Exception:
                modality_idx = None

    run_inference(
        npy_path=args.npy,
        model_py=args.model_py,
        model_class=args.model_class,
        model_kwargs=model_kwargs,
        ckpt_path=args.ckpt,
        device=args.device,
        window_T=args.window_T,
        do_norm=bool(args.do_norm),
        modality_idx=modality_idx,
        num_classes=args.num_classes,
        apply_other_gate=bool(args.apply_other_gate),
        other_top1_min=args.other_top1_min,
        precision=args.precision,
        print_all=bool(args.print_all),
        save_json=args.save_json,
    )


if __name__ == "__main__":
    main()
