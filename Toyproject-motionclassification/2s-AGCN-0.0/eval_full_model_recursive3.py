#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch evaluation (recursive) to mirror your CLI with extras:
 - Args: --data_dir, --label_dir, --pattern, --window_T, --num_classes, --batch_size,
         --device, --tb_logdir, --out_dir, --model_py, --model_class, --model_kwargs, --ckpt
 - Left-aligned window policy (pad right if T<win; take first win if T>win)
 - Head re-attachment if checkpoint used head_replacement.* (Cosine/MLP)
 - Class-wise bias calibration (class_bias.json) applied before prediction
 - OTHER gate option after bias (defaults ON)
 - Per-sample class scores printing & CSV dump
 - TensorBoard logging (acc/macro_f1/loss), CSV export of predictions, confusion matrix save
 - Debug helpers: echo args to file, preview matched files
"""

import os, re, json, argparse, inspect, csv
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# ---------------------- Heads (same as training) ----------------------
class CosineMarginFC(nn.Module):
    def __init__(self, in_feat: int, num_classes: int, s: float = 30.0,
                 margin: float = 0.0, per_class_margin: Optional[List[float]] = None, arcface: bool=False):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_feat, num_classes) * 0.01)
        self.s = float(s)
        self.m = float(margin)
        if per_class_margin is not None:
            assert len(per_class_margin)==num_classes
            self.register_buffer("m_c", torch.tensor(per_class_margin, dtype=torch.float32))
        else:
            self.register_buffer("m_c", None)
        self.arcface = bool(arcface)
    def forward(self, x):
        x_n = torch.nn.functional.normalize(x, dim=1)
        W_n = torch.nn.functional.normalize(self.W, dim=0)
        cos = x_n @ W_n
        m = self.m_c.view(1,-1) if self.m_c is not None else self.m
        logits = self.s * (cos - m)
        return logits

class MLPHead(nn.Module):
    def __init__(self, in_feat: int, num_classes: int, hidden: int = 512, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feat, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class Identity(nn.Module):
    def forward(self, x):
        return x

# ---------------------- Utils ----------------------

def dynamic_import(py_path: str, class_name: str):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("__dyn_model__", str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["__dyn_model__"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {py_path}")
    return getattr(mod, class_name)


def _find_last_linear_qualname(model: nn.Module, num_classes: int) -> Optional[Tuple[str, nn.Linear]]:
    last_name, last_mod = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.out_features == num_classes:
            last_name, last_mod = name, mod
    return (last_name, last_mod)


def _set_by_qualname(root: nn.Module, qualname: str, new_mod: nn.Module):
    parts = qualname.split('.')
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


def attach_head_like_training(model: nn.Module, sd: dict, train_args: Optional[dict], num_classes: int) -> nn.Module:
    has_head = any(k.startswith('head_replacement.') for k in sd.keys())
    if not has_head:
        return model
    head_type = 'cosine'
    mlp_hidden, mlp_dropout = 512, 0.2
    cos_s, cos_margin = 30.0, 0.0
    per_class_margin = None
    if train_args:
        head_type = train_args.get('head_type', head_type)
        mlp_hidden = int(train_args.get('mlp_hidden', mlp_hidden))
        mlp_dropout = float(train_args.get('mlp_dropout', mlp_dropout))
        cos_s = float(train_args.get('cos_s', cos_s))
        cos_margin = float(train_args.get('cos_margin', cos_margin))
        pcm = train_args.get('per_class_margin', '')
        if isinstance(pcm, str) and pcm:
            try:
                per_class_margin = (json.loads(pcm) if pcm.strip().startswith('[')
                                    else [float(x) for x in pcm.split(',')])
            except Exception:
                per_class_margin = None
        elif isinstance(pcm, (list, tuple)):
            per_class_margin = list(map(float, pcm))

    qual, last = _find_last_linear_qualname(model, num_classes)
    if last is None:
        return model
    in_feat = last.in_features
    _set_by_qualname(model, qual, Identity())

    head = MLPHead(in_feat, num_classes, hidden=mlp_hidden, p=mlp_dropout) if head_type=='mlp' \
           else CosineMarginFC(in_feat, num_classes, s=cos_s, margin=cos_margin, per_class_margin=per_class_margin)

    try:
        ref = next(model.parameters())
        head = head.to(device=ref.device, dtype=ref.dtype)
    except StopIteration:
        pass

    model.head_replacement = head
    orig_forward = model.forward
    def new_forward(x):
        feats = orig_forward(x)
        logits = model.head_replacement(feats)
        return logits
    model.forward = new_forward
    return model


def ensure_leftpad(x: np.ndarray, target_T: int) -> np.ndarray:
    assert x.ndim == 4, f"Expected (C,T,V,M), got {x.shape}"
    C,T,V,M = x.shape
    if T == target_T:
        return x
    if T < target_T:
        y = np.zeros((C, target_T, V, M), dtype=x.dtype)
        y[:, :T] = x
        return y
    else:
        return x[:, :target_T]


def load_bias_tensor(bias_json_path: Optional[Path], num_classes: int, device: str):
    if bias_json_path and bias_json_path.is_file():
        with open(bias_json_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        b = obj.get('bias', None)
        if isinstance(b, list) and len(b) == num_classes:
            return torch.tensor(b, dtype=torch.float32, device=device)
    return None


def apply_other_gate(logits: torch.Tensor, top1_min: float, valid_first: int, valid_last: int) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    C = probs.size(1)
    end = min(valid_last, C-1)
    main = probs[:, valid_first:end+1]
    maxp, _ = main.max(dim=1)
    pred = probs.argmax(dim=1)
    pred[maxp < top1_min] = 0
    return pred

# ---------------------- Dataset ----------------------

class EvalFolder(Dataset):
    def __init__(self, data_dir: str, label_dir: Optional[str], pattern: str, window_T: int):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.pattern = pattern
        self.window_T = int(window_T)
        self.files = sorted(Path(data_dir).rglob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files matched: {data_dir} / {pattern}")

    @staticmethod
    def parse_label_pkl(p: Path) -> Optional[int]:
        import pickle
        try:
            with open(p, 'rb') as f:
                lab = pickle.load(f)
        except Exception:
            return None
        if isinstance(lab, dict):
            for k in ("label_index","label","y","class","target","idx","id"):
                if k in lab:
                    lab = lab[k]; break
        if isinstance(lab, (list, tuple, np.ndarray)):
            if len(lab)==0: return None
            lab = lab[0]
        try:
            return int(lab)
        except Exception:
            return None

    def _find_label_for(self, npy_path: Path) -> Optional[int]:
        stem_no_ext = npy_path.stem
        stem = stem_no_ext.replace("_data_joint", "")
        candidates = [f"{stem}_label.pkl", f"{stem_no_ext}.pkl"]
        # in label_dir first
        if self.label_dir:
            L = Path(self.label_dir)
            for name in candidates:
                cand = L / name
                if cand.is_file():
                    v = self.parse_label_pkl(cand)
                    if v is not None: return v
            for name in candidates:
                hits = list(L.rglob(name))
                if hits:
                    v = self.parse_label_pkl(hits[0])
                    if v is not None: return v
        # fallback: alongside npy
        for name in candidates:
            cand = npy_path.parent / name
            if cand.is_file():
                v = self.parse_label_pkl(cand)
                if v is not None: return v
        # filename pattern
        m = re.search(r"_y(\d+)\.npy$", npy_path.name)
        if m:
            return int(m.group(1))
        return None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        x = np.load(p)
        if x.ndim == 5 and x.shape[0] == 1:
            x = x[0]
        x = ensure_leftpad(x, self.window_T).astype('float32', copy=False)
        y = self._find_label_for(p)
        y = -1 if y is None else int(y)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), str(p)

# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_py', type=str, required=True)
    ap.add_argument('--model_class', type=str, required=True)
    ap.add_argument('--model_kwargs', type=str, default='')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--label_dir', type=str, default='')
    ap.add_argument('--pattern', type=str, default='*.npy')
    ap.add_argument('--window_T', type=int, default=300)
    ap.add_argument('--num_classes', type=int, default=6)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--device', type=str, choices=['auto','cuda','cpu'], default='auto')
    ap.add_argument('--tb_logdir', type=str, default='')
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--bias_json', type=str, default='')
    ap.add_argument('--apply_other_gate', type=int, default=1)
    ap.add_argument('--other_top1_min', type=float, default=0.8)
    ap.add_argument('--other_valid_first', type=int, default=1)
    ap.add_argument('--other_valid_last', type=int, default=5)
    ap.add_argument('--num_workers', type=int, default=2)
    # scores / debug helpers
    ap.add_argument('--print_scores', type=int, default=0, help='Print per-sample class scores to stdout (0/1)')
    ap.add_argument('--dump_scores', type=str, default='', help='If set, save per-sample class scores to this CSV path')
    ap.add_argument('--score_precision', type=int, default=4, help='Decimals for printing/saving scores')
    ap.add_argument('--debug_list', type=int, default=0, help='Print/save up to N matched files before running')
    args = ap.parse_args()

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ==== Echo effective options & save to files ====
    _args_dict = {k: (str(v) if isinstance(v, Path) else v) for k,v in vars(args).items()}
    print('[ARGS] Effective options:')
    for k in sorted(_args_dict.keys()):
        print(f'  - {k}: {_args_dict[k]}', flush=True)
    with open(out_dir / 'run_args.json', 'w', encoding='utf-8') as f:
        json.dump(_args_dict, f, ensure_ascii=False, indent=2)
    with open(out_dir / 'run_args.txt', 'w', encoding='utf-8') as f:
        for k in sorted(_args_dict.keys()):
            f.write(f'{k}: {_args_dict[k]}\n')
    print(f"[OK] saved: {out_dir/'run_args.json'}", flush=True)

    # Model
    Model = dynamic_import(args.model_py, args.model_class)
    cli_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}
    sig = inspect.signature(Model.__init__)
    param_names = set(sig.parameters.keys()) - {'self'}
    if 'num_classes' in param_names:
        cli_kwargs.setdefault('num_classes', args.num_classes)
    elif 'num_class' in param_names:
        cli_kwargs.setdefault('num_class', args.num_classes)
    model = Model(**{k:v for k,v in cli_kwargs.items() if k in param_names}).to(device)

    # Load checkpoint & head attach
    obj = torch.load(args.ckpt, map_location=device)
    sd = obj['state_dict'] if isinstance(obj, dict) and 'state_dict' in obj else obj
    train_args = obj.get('args', {}) if isinstance(obj, dict) else {}
    model = attach_head_like_training(model, sd, train_args, args.num_classes)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Bias
    bias_path = Path(args.bias_json) if args.bias_json else (Path(args.ckpt).parent / 'class_bias.json')
    bias = load_bias_tensor(bias_path, args.num_classes, device)

    # Optional: preview matched files before loading dataset
    if args.debug_list and args.debug_list > 0:
        matches = sorted(Path(args.data_dir).rglob(args.pattern))
        print(f"[DEBUG] Found {len(matches)} files for pattern '{args.pattern}' under {args.data_dir}")
        preview = matches[:args.debug_list]
        for i, p in enumerate(preview):
            print(f"  [{i+1:02d}] {p}")
        with open(out_dir / 'file_list_preview.txt', 'w', encoding='utf-8') as f:
            for p in preview:
                f.write(str(p) + '\n')
        print(f"[OK] saved: {out_dir/'file_list_preview.txt'}")

    # Data
    ds = EvalFolder(args.data_dir, args.label_dir or None, args.pattern, args.window_T)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # TB
    writer = SummaryWriter(log_dir=args.tb_logdir) if args.tb_logdir else None

    # Optional scores CSV writer
    scores_writer = None
    scores_fh = None
    if args.dump_scores:
        scores_path = Path(args.dump_scores)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores_fh = open(scores_path, 'w', newline='', encoding='utf-8')
        scores_writer = csv.writer(scores_fh)
        prob_headers = [f'prob_{i}' for i in range(args.num_classes)]
        scores_writer.writerow(['path','label','pred', *prob_headers])
        print(f"[OK] will dump per-sample scores to: {scores_path}")

    cm = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    n_known = 0
    total_loss = 0.0

    rows = []
    ce = nn.CrossEntropyLoss(reduction='sum')  # labels==-1 skipped

    with torch.no_grad():
        for X, y, paths in dl:
            X = X.to(device)
            logits = model(X)
            if bias is not None:
                logits = logits + bias.view(1,-1)
            # OTHER gate after bias, only for prediction selection
            if args.apply_other_gate:
                pred = apply_other_gate(logits, args.other_top1_min, args.other_valid_first, args.other_valid_last)
            else:
                pred = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)
            conf = probs[torch.arange(probs.size(0)), pred].cpu().numpy()

            # Optional: print/dump per-sample class scores
            if args.print_scores or args.dump_scores:
                probs_np = probs.cpu().numpy()
                for i, path_i in enumerate(paths):
                    label_i = int(y[i].item())
                    pred_i = int(pred[i].item())
                    if args.print_scores:
                        vals = ' '.join([f"c{j}:{probs_np[i,j]:.{args.score_precision}f}" for j in range(probs_np.shape[1])])
                        print(f"[SCORES] {path_i} | y={label_i} pred={pred_i} | {vals}")
                    if scores_writer is not None:
                        row = [path_i, label_i, pred_i] + [round(float(p), args.score_precision) for p in probs_np[i].tolist()]
                        scores_writer.writerow(row)

            # loss/metrics with known labels
            mask = (y >= 0) & (y < args.num_classes)
            if mask.any():
                yy = y[mask].to(device)
                ll = logits[mask]
                total_loss += ce(ll, yy).item()
                n_known += int(mask.sum().item())
                pp = pred[mask].cpu().numpy()
                yy_np = yy.cpu().numpy()
                for a,b in zip(yy_np, pp):
                    cm[int(a), int(b)] += 1

            # collect rows
            for i in range(len(paths)):
                rows.append((paths[i], int(y[i].item()), int(pred[i].item()), float(conf[i])))

    acc = (np.trace(cm) / max(1, n_known)) if n_known else float('nan')
    def macro_f1(cm: np.ndarray) -> float:
        C = cm.shape[0]
        f1s = []
        for c in range(C):
            tp = cm[c,c]
            fp = cm[:,c].sum() - tp
            fn = cm[c,:].sum() - tp
            prec = tp / (tp+fp+1e-9)
            rec  = tp / (tp+fn+1e-9)
            f1 = 2*prec*rec / (prec+rec+1e-9)
            f1s.append(f1)
        return float(np.mean(f1s))
    mf1 = macro_f1(cm) if n_known else float('nan')

    avg_loss = (total_loss / max(1, n_known)) if n_known else float('nan')

    print(f"[SUMMARY] files={len(rows)} known_labels={n_known} acc={acc:.4f} f1={mf1:.4f} loss={avg_loss:.4f}")

    # Save CSV
    csv_path = out_dir / 'predictions.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['path','label','pred','conf'])
        for r in rows:
            w.writerow(r)
    print(f"[OK] saved: {csv_path}")

    # Save confusion matrix
    np.save(out_dir / 'confusion_matrix.npy', cm)
    with open(out_dir / 'confusion_matrix.txt', 'w', encoding='utf-8') as f:
        f.write("Confusion Matrix (rows=true, cols=pred)\n")
        f.write(np.array2string(cm, max_line_width=200))
    print(f"[OK] saved: confusion_matrix.npy / confusion_matrix.txt")

    # Close scores CSV if opened
    if 'scores_fh' in locals() and scores_fh is not None:
        scores_fh.close()

    # TB
    if writer:
        writer.add_scalar('eval/acc', acc, 0)
        writer.add_scalar('eval/macro_f1', mf1, 0)
        writer.add_scalar('eval/loss', avg_loss, 0)
        writer.flush(); writer.close()

if __name__ == '__main__':
    main()
