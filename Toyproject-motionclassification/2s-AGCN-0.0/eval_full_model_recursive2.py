# -*- coding: utf-8 -*-
"""
Evaluate a FULL model checkpoint (state_dict for backbone+head) over an entire folder tree.

Key features:
- Recursive file matching: use --pattern "**/*.npy" (comma-separated patterns allowed)
- Recursive label sidecar lookup under --label_dir
- Accurate sample-weighted mean loss
- Acc / F1 (macro/weighted), confusion matrix
- Saves predictions.csv & confusion_matrix.png
- Optional TensorBoard logging

Usage:
python -u eval_full_model_recursive.py \
  --model_py "/content/drive/MyDrive/ToyProject/2s-AGCN-0.0/model/agcn.py" \
  --model_class "Model" \
  --model_kwargs '{"num_class": 6, "in_channels": 3, "num_point": 25, "num_person": 2, "graph": "graph.ntu_rgb_d.Graph", "graph_args":{"labeling_mode":"spatial"}}' \
  --ckpt "/content/drive/MyDrive/ToyProject/fine-tuning_full_0922/best.pt" \
  --data_dir  "/content/drive/MyDrive/ToyProject/dataset/final_dataset_new/test" \
  --label_dir "/content/drive/MyDrive/ToyProject/dataset/final_dataset_new/test/label" \
  --pattern "**/*.npy" \
  --window_T 300 \
  --num_classes 6 \
  --batch_size 32 \
  --device cuda \
  --tb_logdir "/content/drive/MyDrive/ToyProject/tb_logs/eval_full_recursive" \
  --out_dir "/content/drive/MyDrive/ToyProject/eval_full_recursive_out"
"""

import argparse, json, os, re, pickle, time
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ---------------- data utils ----------------

def read_label_from_pkl(p: Path) -> int:
    with open(p, "rb") as f:
        L = pickle.load(f)
    if isinstance(L, dict):
        if "label_index" in L and len(L["label_index"])>0:
            return int(L["label_index"][0])
        for k in ("label","y","class","target","idx","id"):
            if k in L:
                v = L[k]
                return int(v[0] if isinstance(v,(list,tuple,np.ndarray)) else v)
    if isinstance(L, (list,tuple,np.ndarray)) and len(L)>0:
        return int(L[0])
    return int(L)

# ===== robust label fetch =====
import re, pickle
from pathlib import Path
from typing import Any

def _strip_at_y(stem: str) -> str:
    """파일명(stem)에서 첫 `_y<digits>` 이전까지를 기준 키로 사용."""
    m = re.search(r"_y\d+", stem, flags=re.IGNORECASE)
    return stem[:m.start()] if m else stem

def _to_int_if_possible(obj: Any):
    """dict 라벨 지원: 공통 키/단일 값/스칼라 캐스팅."""
    if isinstance(obj, bool): return True, int(obj)
    if isinstance(obj, int):  return True, obj
    if isinstance(obj, float):
        r = round(obj);  return (abs(obj - r) < 1e-6, int(r))
    try:
        import numpy as np
        if isinstance(obj, np.integer):  return True, int(obj)
        if isinstance(obj, np.floating):
            fv = float(obj); r = round(fv)
            return (abs(fv - r) < 1e-6, int(r))
    except Exception:
        pass
    if isinstance(obj, str):
        s = obj.strip()
        if s.isdigit(): return True, int(s)
        try:
            fv = float(s); r = round(fv)
            if abs(fv - r) < 1e-6: return True, int(r)
        except Exception:
            pass
    if isinstance(obj, (list, tuple)) and len(obj) == 1:
        return _to_int_if_possible(obj[0])
    if isinstance(obj, dict):
        for k in ["label","y","class","cls","target","gt","category","cat","id"]:
            if k in obj:
                ok, v = _to_int_if_possible(obj[k])
                if ok: return True, v
        for v in obj.values():
            ok, vv = _to_int_if_possible(v)
            if ok: return True, vv
    return False, None

def _load_label_as_int(pkl_path: Path) -> int:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    ok, val = _to_int_if_possible(obj)
    if not ok:
        raise ValueError(f"Unsupported label object in {pkl_path}: {type(obj)}")
    return int(val)

def get_label_for_file(npy_path: Path, label_dir: Path) -> int:
    """
    bone/joint/증강 꼬리표와 무관하게 같은 '기준 키'로 PKL을 찾는다.
    기준 키 = 파일명(stem)에서 첫 `_y<digits>` 이전까지.
    예) S001..._data_joint_y1_bone.npy -> 기준 키: S001..._data_joint
    """
    stem = npy_path.stem
    # bone 꼬리표 제거 (안전)
    stem = stem.replace("_bone", "")
    key  = _strip_at_y(stem)

    # 1) 가장 흔한 케이스: {key}.pkl
    cand = label_dir / f"{key}.pkl"
    if cand.exists():
        return _load_label_as_int(cand)

    # 2) 라벨 폴더에 다양한 이름이 있을 수 있으니, 같은 기준 키를 가진 pkl을 스캔
    for p in label_dir.rglob("*.pkl"):
        if _strip_at_y(p.stem) == key:
            return _load_label_as_int(p)

    # 못 찾으면 에러
    raise FileNotFoundError(f"Label not found for: {npy_path.name} (base key='{key}')")

def ensure_ctvm(arr: np.ndarray) -> np.ndarray:
    x = arr
    if x.ndim==5 and x.shape[0]==1: x = x[0]
    if x.ndim!=4: raise ValueError(f"Expected (C,T,V,M) or (1,C,T,V,M), got {x.shape}")
    return x.astype(np.float32, copy=False)

def crop_center_T(x: np.ndarray, T: int) -> np.ndarray:
    C,t,V,M = x.shape
    if t==T: return x
    if t>T:
        s = max(0,(t-T)//2)
        return x[:, s:s+T]
    y = np.zeros((C,T,V,M), dtype=x.dtype)
    s = (T-t)//2
    y[:, s:s+t] = x
    return y

class NpySidecarDataset(Dataset):
    def __init__(self, data_dir: str, label_dir: str|None, pattern="*.npy", window_T=300):
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir) if label_dir else None

        # comma-separated patterns, recursive rglob for patterns containing "**"
        patterns = [p.strip() for p in str(pattern).split(",") if p.strip()]
        files = []
        for pat in patterns:
            if "**" in pat:
                files.extend(self.data_dir.rglob(pat))
            else:
                files.extend(self.data_dir.glob(pat))
        # unique + keep files only
        self.files = sorted({f for f in files if getattr(f, "is_file", lambda: False)()})
        if not self.files:
            raise FileNotFoundError(f"No files matched: {self.data_dir}/{pattern}")
        print(f"[Files] matched {len(self.files)} files (pattern='{pattern}')")
        for s in map(str, self.files[:10]):
            print("   -", s)
        if len(self.files) > 10:
            print("   ...")

        self.T = window_T

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        x = ensure_ctvm(np.load(f))
        x = crop_center_T(x, self.T)
        y = get_label_for_file(f, self.label_dir)
        return torch.from_numpy(x), int(y), f.name

# -------------- model helpers --------------

def dynamic_import(py_path: str, class_name: str):
    import importlib.util, sys
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
    for k,v in sd.items():
        if isinstance(k, str) and k.startswith("module."): k = k[7:]
        if isinstance(k, str) and k.startswith("model."):  k = k[6:]
        out[k] = v
    return out

def confusion_figure(cm: np.ndarray, num_classes: int, title="Confusion"):
    fig = plt.figure(figsize=(4,3)); ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest'); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title); ax.set_xlabel("Pred"); ax.set_ylabel("True")
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    fig.tight_layout()
    return fig

@torch.no_grad()
def apply_other_gate(logits: torch.Tensor, top1_min: float = 0.8,
                     valid_class_start: int = 1, valid_class_end: int | None = None) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    C = probs.size(1)
    end = valid_class_end if valid_class_end is not None else (C - 1)
    main = probs[:, valid_class_start:end+1]
    maxp, _ = main.max(dim=1)
    pred = probs.argmax(dim=1)
    pred[maxp < top1_min] = 0
    return pred

# -------------- main --------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_py", type=str, required=True)
    ap.add_argument("--model_class", type=str, required=True)
    ap.add_argument("--model_kwargs", type=str, default="")

    ap.add_argument("--ckpt", type=str, required=True, help="FULL model checkpoint (.pt with state_dict)")

    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--label_dir", type=str, default="")
    ap.add_argument("--pattern", type=str, default="*.npy")
    ap.add_argument("--window_T", type=int, default=300)

    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--apply_other_gate", type=int, default=0)
    ap.add_argument("--other_top1_min", type=float, default=0.8)
    ap.add_argument("--other_valid_first", type=int, default=1)
    ap.add_argument("--other_valid_last", type=int, default=5)

    ap.add_argument("--tb_logdir", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)

    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # dataset
    ds = NpySidecarDataset(args.data_dir, args.label_dir if args.label_dir else None,
                           pattern=args.pattern, window_T=args.window_T)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # model build
    ModelClass = dynamic_import(args.model_py, args.model_class)

    # parse kwargs + harmonize num_class/num_classes + drop unknowns if needed
    import inspect
    try:
        cli_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}
    except Exception as e:
        raise ValueError(f"--model_kwargs JSON parse error: {e}")
    sig = inspect.signature(ModelClass.__init__)
    param_names = set(sig.parameters.keys()) - {"self"}
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    # ensure class count
    if "num_classes" in cli_kwargs and "num_class" not in cli_kwargs and "num_class" in param_names:
        cli_kwargs["num_class"] = cli_kwargs.pop("num_classes")
    if "num_class" in cli_kwargs and "num_classes" not in cli_kwargs and "num_classes" in param_names:
        cli_kwargs["num_classes"] = cli_kwargs.pop("num_class")
    if "num_class" not in cli_kwargs and "num_classes" not in cli_kwargs:
        if "num_class" in param_names: cli_kwargs["num_class"] = args.num_classes
        elif "num_classes" in param_names: cli_kwargs["num_classes"] = args.num_classes

    if not has_var_kw:
        cli_kwargs = {k: v for k, v in cli_kwargs.items() if k in param_names}

    dev = torch.device(args.device if (args.device=="cuda" and torch.cuda.is_available()) else "cpu")
    model = ModelClass(**cli_kwargs).to(dev)
    model.eval()

    # load FULL checkpoint
    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    sd = strip_common_prefixes(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)} (ok if buffers differ)")

    # eval
    crit = nn.CrossEntropyLoss(reduction='mean')  # batch-mean
    total_loss_sum, total_n = 0.0, 0
    all_true, all_pred, all_names = [], [], []

    with torch.no_grad():
        for X, y, names in dl:
            X = X.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True, dtype=torch.long)
            logits = model(X)
            loss = crit(logits, y).detach()
            bs = y.size(0)
            total_loss_sum += float(loss.item()) * bs
            total_n += int(bs)

            if args.apply_other_gate:
                pred = apply_other_gate(logits, top1_min=args.other_top1_min,
                                        valid_class_start=args.other_valid_first,
                                        valid_class_end=args.other_valid_last)
            else:
                pred = logits.argmax(1)

            all_true.append(y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
            all_names += list(names)

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    loss_mean = total_loss_sum / max(1, total_n)

    acc = float((y_true == y_pred).mean())
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    print(f"[TEST] N={len(y_true)} | acc={acc:.4f} | loss={loss_mean:.4f} | f1_macro={f1_macro:.4f} | f1_weighted={f1_weighted:.4f}")

    # Enhanced reports
    present = sorted(np.unique(y_true))
    print("[Info] present labels in y_true:", present)
    print("\n[Report | present labels only]")
    print(classification_report(y_true, y_pred, labels=present, digits=4, zero_division=0))
    print("\n[Report | full labels 0..num_classes-1]")
    print(classification_report(y_true, y_pred, labels=list(range(args.num_classes)), digits=4, zero_division=0))
    f1_macro_all = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro_trueonly = float(f1_score(y_true, y_pred, average="macro", labels=present, zero_division=0))
    print(f"[F1 macro] all={f1_macro_all:.4f}  |  true-only={f1_macro_trueonly:.4f}")

    # confusion, save artifacts
    cm = confusion_matrix(y_true, y_pred, labels=list(range(args.num_classes)))
    import csv
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir/"predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file","true","pred"])
        for nm, t, p in zip(all_names, y_true.tolist(), y_pred.tolist()):
            w.writerow([nm, t, p])
    print(f"[OUT] predictions -> {out_dir/'predictions.csv'}")

    fig = confusion_figure(cm, args.num_classes, title="Confusion")
    fig.savefig(out_dir/"confusion_matrix.png", dpi=160); plt.close(fig)
    print(f"[OUT] confusion matrix -> {out_dir/'confusion_matrix.png'}")

    if args.tb_logdir:
        tb = Path(args.tb_logdir); tb.mkdir(parents=True, exist_ok=True)
        run = f"eval_full_recursive_{time.strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(str(tb/run))
        writer.add_scalar("metrics/acc", acc, 0)
        writer.add_scalar("metrics/loss", loss_mean, 0)
        writer.add_scalar("metrics/f1_macro", f1_macro, 0)
        writer.add_scalar("metrics/f1_macro_all", f1_macro_all, 0)
        writer.add_scalar("metrics/f1_macro_trueonly", f1_macro_trueonly, 0)
        fig2 = confusion_figure(cm, args.num_classes, title="Confusion")
        writer.add_figure("confusion_matrix", fig2, 0); plt.close(fig2)
        writer.add_hparams({"ckpt": os.path.basename(args.ckpt)}, {"hparam/acc": acc, "hparam/loss": loss_mean})
        writer.flush(); writer.close()
        print(f"[TB] logs saved to: {tb/run}")

if __name__ == "__main__":
    main()
