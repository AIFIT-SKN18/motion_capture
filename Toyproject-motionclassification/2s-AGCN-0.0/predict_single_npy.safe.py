# 덮어쓰기
# -*- coding: utf-8 -*-
"""
predict_single_npy.safe.py (clean)
- Load FULL (backbone+head) model checkpoint and predict a SINGLE .npy
- Class-name mapping (0..5)
- Optional OTHER gate (if confidence for 1..5 < threshold -> class 0)
- CPU/GPU safe, stateless inference
- Robust NPY loader:
    * (C,T,V,M)
    * (1,C,T,V,M) -> squeeze
    * (2,3,T,V,M) -> two-stream (AGCN_MODALITY_IDX: 0=joint, 1=bone, -1=concat->(6,T,V,M))
    * (N,C,T,V,M) -> batch로 보고 첫 샘플 사용
- Train-like preprocessing: center-to-joint, std-norm, window to T=300, M=2

Usage example:
python -u predict_single_npy.safe.py \
  --model_py "/content/drive/MyDrive/2s-AGCN-0.0/model/agcn.py" \
  --model_class "Model" \
  --model_kwargs '{"num_class":6,"in_channels":3,"num_point":25,"num_person":2,"graph":"graph.ntu_rgb_d.Graph","graph_args":{"labeling_mode":"spatial"}}' \
  --ckpt "/content/drive/MyDrive/fine-tuning_outdir/0924-3/best.pt" \
  --npy  "/content/drive/MyDrive/single_npy/S001C002P001R001A999_test6_benchpress.npy" \
  --window_T 300 --num_classes 6 --device cpu \
  --apply_other_gate 1 --other_top1_min 0.80 --other_valid_first 1 --other_valid_last 5 \
  --precision 6 --print_all 1 \
  --save_json "/content/drive/MyDrive/single_pred/pred_single_0924.json"

# 2-stream NPY에서 joint/bone 선택:
#   %env AGCN_MODALITY_IDX=0  # joint
#   %env AGCN_MODALITY_IDX=1  # bone
#   %env AGCN_MODALITY_IDX=-1 # early-fusion(6ch) -> 모델 in_channels=6 필요
"""

from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
import inspect

# ---------- Class names (fixed 0..5) ----------
CLASS_NAMES = [
    "other",
    "dumbbell benchpress",
    "dumbbell deadlift",
    "dumbbell lunges",
    "dumbbell side lateral raise",
    "dumbbell squat",
]

# ---------- dynamic import ----------
def dynamic_import(py_path: str, class_name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("__dyn_model__", py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["__dyn_model__"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {py_path}")
    return getattr(mod, class_name)

def strip_common_prefixes(sd: dict) -> OrderedDict:
    out = OrderedDict()
    for k, v in sd.items():
        if isinstance(k, str) and k.startswith("module."):
            k = k[7:]
        if isinstance(k, str) and k.startswith("model."):
            k = k[6:]
        out[k] = v
    return out

@torch.no_grad()
def apply_other_gate(logits: torch.Tensor, top1_min: float = 0.8,
                     valid_class_start: int = 1, valid_class_end: int | None = None) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    C = probs.size(1)
    end = valid_class_end if valid_class_end is not None else (C - 1)
    main = probs[:, valid_class_start:end + 1]
    maxp, _ = main.max(dim=1)
    pred = probs.argmax(dim=1)
    pred[maxp < top1_min] = 0
    return pred

def param_fingerprint(model: torch.nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        s += float(p.detach().abs().sum().cpu())
    return s

# ---------- IO helpers ----------
def ensure_ctvm(x: np.ndarray) -> np.ndarray:
    """Return 4D (C,T,V,M). Accept (C,T,V,M) or (1,C,T,V,M)."""
    x = np.asarray(x)
    if x.ndim == 5 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 4:
        raise ValueError(f"Expected (C,T,V,M) or (1,C,T,V,M), got {x.shape}")
    return x.astype(np.float32, copy=False)

def ensure_ctvm_flex(arr: np.ndarray) -> np.ndarray:
    """
    Robust 5D handler:
      - (1,C,T,V,M) -> squeeze -> 4D
      - (2,3,T,V,M) -> two-stream or concat(6ch) via env AGCN_MODALITY_IDX
      - (N,C,T,V,M) -> batch -> first sample
      - (C,T,V,M)   -> passthrough
    """
    x = np.asarray(arr)
    if x.ndim == 4:
        return ensure_ctvm(x)

    if x.ndim == 5:
        n0, c, t, v, m = x.shape
        # (1,C,T,V,M): squeeze
        if n0 == 1:
            print("[ensure_ctvm_flex] 1-sample 5D -> squeeze to 4D")
            return ensure_ctvm(x[0])

        # (2,3,T,V,M): joint/bone 2-stream
        if n0 == 2 and c in (3, 6):
            mi_str = os.environ.get("AGCN_MODALITY_IDX", "0")
            try:
                mi = int(mi_str)
            except Exception:
                mi = 0

            if mi == -1:
                # concat two 3ch streams -> 6ch
                if c == 3:
                    print("[ensure_ctvm_flex] concat 2x3ch -> 6ch")
                    x6 = x.reshape(2 * 3, t, v, m).astype(np.float32, copy=False)
                    return x6
                else:
                    print("[ensure_ctvm_flex][WARN] (2,6,...) given -> fallback to stream 0")
                    return ensure_ctvm(x[0])

            if mi not in (0, 1):
                print(f"[ensure_ctvm_flex][WARN] invalid AGCN_MODALITY_IDX={mi_str} -> 0")
                mi = 0

            print(f"[ensure_ctvm_flex] 2-stream detected -> using stream {mi} "
                  f"({'joint' if mi==0 else 'bone'})")
            return ensure_ctvm(x[mi])

        # other 5D: batch -> first sample
        print(f"[ensure_ctvm_flex][WARN] treat 5D as batch (N={n0}) -> first sample")
        return ensure_ctvm(x[0])

    raise ValueError(f"Unsupported npy shape {x.shape}. Expected (C,T,V,M) or (N,C,T,V,M).")

def preprocess_like_train(x: np.ndarray, center_joint_idx: int = 1,
                          window_size: int = 300, do_norm: bool = True) -> np.ndarray:
    """
    Train-like preprocessing:
      1) center to 'center_joint_idx'
      2) std normalization (global)
      3) temporal window to 'window_size' (pad/crop center)
      4) fix M=2 (pad or slice)
    """
    x = np.asarray(x).astype('float32')
    if x.ndim != 4:
        raise ValueError(f"preprocess_like_train expects 4D (C,T,V,M), got {x.shape}")
    C, T, V, M = x.shape

    # (1) center to joint
    ref = x[:, :, center_joint_idx, :].copy()  # (C,T,M)
    x = x - ref[:, :, None, :]

    # (2) normalization
    if do_norm:
        s = x.std()
        if not np.isfinite(s) or s < 1e-6:
            s = 1.0
        x = x / s

    # (3) window
    if window_size and window_size > 0:
        if T < window_size:
            pad = np.zeros((C, window_size - T, V, M), dtype=x.dtype)
            x = np.concatenate([x, pad], axis=1)
        elif T > window_size:
            start = (T - window_size) // 2
            x = x[:, start:start + window_size, :, :]

    # (4) ensure M=2
    target_M = 2
    if M < target_M:
        padm = np.zeros((C, x.shape[1], V, target_M - M), dtype=x.dtype)
        x = np.concatenate([x, padm], axis=3)
    elif M > target_M:
        x = x[:, :, :, :target_M]

    return x

def crop_center_T(x: np.ndarray, T: int) -> np.ndarray:
    C, t, V, M = x.shape
    if t == T:
        return x
    if t > T:
        s = max(0, (t - T) // 2)
        return x[:, s:s + T, :, :]
    y = np.zeros((C, T, V, M), dtype=x.dtype)
    s = (T - t) // 2
    y[:, s:s + t, :, :] = x
    return y

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_py", type=str, required=True)
    ap.add_argument("--model_class", type=str, required=True)
    ap.add_argument("--model_kwargs", type=str, default="")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--npy", type=str, required=True)
    ap.add_argument("--window_T", type=int, default=300)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--print_topk", type=int, default=6)
    ap.add_argument("--apply_other_gate", type=int, default=0)
    ap.add_argument("--other_top1_min", type=float, default=0.8)
    ap.add_argument("--other_valid_first", type=int, default=1)
    ap.add_argument("--other_valid_last", type=int, default=5)
    ap.add_argument("--precision", type=int, default=6)
    ap.add_argument("--print_all", type=int, default=0)
    ap.add_argument("--save_json", type=str, default="")
    ap.add_argument("--fingerprint_check", type=int, default=1)
    args = ap.parse_args()

    # device
    dev_str = str(args.device).lower()
    use_cuda = (dev_str == "cuda" and torch.cuda.is_available())
    dev = torch.device("cuda" if use_cuda else "cpu")
    if dev.type == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(f"[Device] using {dev}")

    # build model
    ModelClass = dynamic_import(args.model_py, args.model_class)
    try:
        cli_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}
    except Exception as e:
        raise ValueError(f"--model_kwargs JSON parse error: {e}")

    # harmonize num_class / num_classes
    sig = inspect.signature(ModelClass.__init__)
    param_names = set(sig.parameters.keys()) - {"self"}
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if "num_classes" in cli_kwargs and "num_class" not in cli_kwargs and "num_class" in param_names:
        cli_kwargs["num_class"] = cli_kwargs.pop("num_classes")
    if "num_class" in cli_kwargs and "num_classes" not in cli_kwargs and "num_classes" in param_names:
        cli_kwargs["num_classes"] = cli_kwargs.pop("num_class")
    if "num_class" not in cli_kwargs and "num_classes" not in cli_kwargs:
        if "num_class" in param_names:
            cli_kwargs["num_class"] = args.num_classes
        elif "num_classes" in param_names:
            cli_kwargs["num_classes"] = args.num_classes
    if not has_var_kw:
        cli_kwargs = {k: v for k, v in cli_kwargs.items() if k in param_names}

    model = ModelClass(**cli_kwargs).to(dev)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # load ckpt
    sd = torch.load(args.ckpt, map_location=dev)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = strip_common_prefixes(sd)
    ckpt_keys = list(sd.keys())
    num_fc_like = sum(1 for k in ckpt_keys if k.startswith("fc.") or k.startswith("classifier."))
    print(f"[CKPT] keys={len(ckpt_keys)} | backbone_like={len(ckpt_keys)-num_fc_like} | fc_like={num_fc_like}")
    print("        samples:", ckpt_keys[:3], "...")

    ik = model.load_state_dict(sd, strict=False)
    missing = getattr(ik, "missing_keys", ik[0] if isinstance(ik, (tuple, list)) else [])
    unexpected = getattr(ik, "unexpected_keys", ik[1] if isinstance(ik, (tuple, list)) else [])
    print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("        missing(sample):", missing[:5])
    if unexpected:
        print("        unexpected(sample):", unexpected[:5])

    # load npy -> robust -> preprocess -> crop
    f = Path(args.npy)
    assert f.exists(), f"File not found: {f}"
    x = np.load(f)
    x = ensure_ctvm_flex(x)                              # -> (C,T,V,M)
    x = preprocess_like_train(x, window_size=300, do_norm=True)
    x = crop_center_T(x, args.window_T)
    X = torch.from_numpy(x).unsqueeze(0).to(dev)         # [1,C,T,V,M]

    # inference
    fp_before = param_fingerprint(model) if args.fingerprint_check else None
    with torch.inference_mode():
        logits = model(X)                                # [1,C]
        probs  = torch.softmax(logits, dim=1)           # [1,C]
    fp_after = param_fingerprint(model) if args.fingerprint_check else None
    if args.fingerprint_check and abs(fp_after - fp_before) > 1e-6:
        print("[WARN] parameters changed during inference?! (unexpected)")

    # OTHER gate
    if args.apply_other_gate:
        pred = apply_other_gate(logits,
                                top1_min=args.other_top1_min,
                                valid_class_start=args.other_valid_first,
                                valid_class_end=args.other_valid_last)
    else:
        pred = probs.argmax(dim=1)

    pred_cls = int(pred.item())
    cls_name = CLASS_NAMES[pred_cls] if pred_cls < len(CLASS_NAMES) else f"class{pred_cls}"

    # print results
    topk = probs.size(1) if args.print_all else min(args.print_topk, probs.size(1))
    vals, idxs = torch.topk(probs, k=topk, dim=1)
    prec = max(0, int(args.precision))

    print(f"\n[Result] file={f.name}")
    print(f"  predicted: {pred_cls} ({cls_name})")
    print(f"  sum(probs) = {float(probs.sum()):.{prec}f}")
    print(f"  top-{topk}:")
    for r in range(topk):
        cls = int(idxs[0, r].item())
        pv = float(vals[0, r].item())
        cname = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"class{cls}"
        print(f"    #{r+1}: class {cls} ({cname})  prob={pv:.{prec}f}")

    # save json
    if args.save_json:
        out = {
            "file": f.name,
            "pred": pred_cls,
            "pred_name": cls_name,
            "topk": [
                {
                    "class": int(idxs[0, i].item()),
                    "name": (CLASS_NAMES[int(idxs[0, i].item())]
                             if int(idxs[0, i].item()) < len(CLASS_NAMES)
                             else f"class{int(idxs[0, i].item())}"),
                    "prob": float(vals[0, i].item())
                }
                for i in range(topk)
            ],
            "probs": [float(p) for p in probs[0].cpu().numpy().tolist()],
            "class_names": CLASS_NAMES[:],
            "apply_other_gate": int(args.apply_other_gate),
            "other_top1_min": float(args.other_top1_min),
            "other_valid_range": [int(args.other_valid_first), int(args.other_valid_last)]
        }
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as g:
            json.dump(out, g, ensure_ascii=False, indent=2)
        print(f"[OUT] saved JSON -> {args.save_json}")

if __name__ == "__main__":
    main()
