from pathlib import Path
import argparse, json, importlib.util, sys
import numpy as np
import torch
import torch.nn.functional as F

def load_model_from_py(py_path: str, class_name: str, kwargs: dict):
    spec = importlib.util.spec_from_file_location("user_model", py_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_model"] = module
    spec.loader.exec_module(module)  # type: ignore
    ModelClass = getattr(module, class_name)
    return ModelClass(**kwargs)

def strip_prefix_if_present(state_dict, prefixes=("module.", "model.")):
    out = {}
    for k, v in state_dict.items():
        new_k = k
        for p in prefixes:
            if new_k.startswith(p):
                new_k = new_k[len(p):]
        out[new_k] = v
    return out

def load_ckpt(ckpt_path: str, device: torch.device):
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict):
        for key in ["model_state", "state_dict", "model", "net"]:
            if key in obj and isinstance(obj[key], dict):
                return strip_prefix_if_present(obj[key])
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return strip_prefix_if_present(obj)
    raise RuntimeError("Unsupported checkpoint format.")

def center_zero_pad_or_center_crop(x: np.ndarray, target_T: int) -> np.ndarray:
    """학습(폴더 로더)와 동일 정책: 왼쪽 정렬 제로패딩 / 앞쪽 자르기"""
    assert x.ndim == 4, f"Expected (C,T,V,M), got shape={x.shape}"
    C, T, V, M = x.shape
    if T == target_T:
        return x
    if T < target_T:
        y = np.zeros((C, target_T, V, M), dtype=x.dtype)
        y[:, :T] = x   # 앞쪽 정렬 패딩
        return y
    # T > target_T: 앞쪽(시작) 기준 잘라내기
    return x[:, :target_T]

def ensure_ctvm_shape(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 5:  # (N,C,T,V,M) → 첫 샘플
        arr = arr[0]
    if arr.ndim == 3:  # (C,T,V) → M=1
        C, T, V = arr.shape
        arr = arr.reshape(C, T, V, 1)
    if arr.ndim != 4:
        raise ValueError(f"Unsupported npy shape {arr.shape}")
    return arr

def maybe_z_norm(x: np.ndarray, do_norm: int) -> np.ndarray:
    if not do_norm:
        return x
    C, T, V, M = x.shape
    x_ = x.reshape(C, -1)
    mean = x_.mean(axis=1, keepdims=True)
    std = x_.std(axis=1, keepdims=True) + 1e-8
    x_ = (x_ - mean) / std
    return x_.reshape(C, T, V, M)

def build_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    x = x.astype(np.float32, copy=False)
    ten = torch.from_numpy(x).unsqueeze(0).to(device)  # (1,C,T,V,M)
    return ten

def topk_from_logits(logits: torch.Tensor, k: int = 5):
    prob = F.softmax(logits, dim=1)
    px, ix = torch.topk(prob, k=min(k, prob.shape[1]), dim=1)
    return ix.squeeze(0).tolist(), px.squeeze(0).tolist(), prob.squeeze(0)

def main():
    ap = argparse.ArgumentParser(description="Single NPY inference (left pad/front crop)")
    ap.add_argument("--model_py", required=True)
    ap.add_argument("--model_class", default="Model")
    ap.add_argument("--model_kwargs", default="{}")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--window_T", type=int, default=300)
    ap.add_argument("--num_classes", type=int, default=None)
    ap.add_argument("--do_norm", type=int, default=0)
    ap.add_argument("--modality", type=int, default=0)
    ap.add_argument("--print_all", type=int, default=1)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    try:
        model_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}
    except Exception as e:
        raise SystemExit(f"--model_kwargs must be JSON. Error: {e}")

    if args.num_classes is not None:
        for key in ["num_class", "num_classes"]:
            if key in model_kwargs:
                model_kwargs[key] = args.num_classes
        if "num_class" not in model_kwargs and "num_classes" not in model_kwargs:
            model_kwargs["num_class"] = args.num_classes

    model = load_model_from_py(args.model_py, args.model_class, model_kwargs).to(device)
    model.eval()

    state = load_ckpt(args.ckpt, device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if args.print_all:
        print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}")

    arr = np.load(args.npy, allow_pickle=False)
    arr = ensure_ctvm_shape(arr)

    if args.print_all:
        C, T, V, M = arr.shape
        print(f"[INPUT] shape={arr.shape} mean={arr.mean():.6f} std={arr.std():.6f}")

    arr = center_zero_pad_or_center_crop(arr, args.window_T)  # ← 이제 왼쪽 정렬 정책
    arr = maybe_z_norm(arr, args.do_norm)
    x = build_tensor(arr, device)

    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]

    idxs, probs, _ = topk_from_logits(logits, k=5)
    print("\n=== Prediction ===")
    print(f"Top-1: class_id={idxs[0]}, prob={probs[0]:.4f}")
    print("Top-5:")
    for i, (ci, pi) in enumerate(zip(idxs, probs), 1):
        print(f" {i:>2}. {ci:>6} : {pi:.4f}")

if __name__ == "__main__":
    main()
