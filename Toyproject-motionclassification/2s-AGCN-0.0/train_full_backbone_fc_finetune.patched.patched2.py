# -*- coding: utf-8 -*-
"""
Full-model (2s-AGCN backbone) training with FC-only fine-tuning.

- Keeps the original script's CLI/layout (folder feeder, early stopping, range-save, TB logging, gating).
- Builds the FULL model from a user-specified Python file/class (e.g., model/agcn.py::Model).
- Optionally loads a base FULL checkpoint before training (--base_ckpt).
- Freezes all params EXCEPT those whose names start with any of --ft_train_prefixes
  (default: "classifier.,fc.,head.").
- Optimizer sees only trainable params.
- Saves checkpoints (full state_dict).

Example (Colab):
python -u train_full_backbone_fc_finetune.py \
  --config "/content/drive/MyDrive/ToyProject/train_nomerge_new_toy.yaml" \
  --use_folder_feeder 1 \
  --data_dir  "/content/drive/MyDrive/ToyProject/dataset/final_dataset_new/train/joint" \
  --label_dir "/content/drive/MyDrive/ToyProject/dataset/final_dataset_new/train/label" \
  --pattern "*.npy" \
  --label_from sidecar \
  --folder_window_size 300 \
  --device cuda \
  --k_folds 1 \
  --single_split_train_ratio 0.85 \
  --early_metric acc \
  --patience 7 \
  --min_delta 0.0 \
  --num_epoch 20 \
  --save_acc_lower 0.80 \
  --save_acc_upper 0.98 \
  --save_dir "/content/drive/MyDrive/ToyProject/fine-tuning_full_0922" \
  --apply_other_gate 0 \
  --tb_logdir "/content/drive/MyDrive/ToyProject/tb_logs/train_full_fc" \
  --model_py "/content/drive/MyDrive/ToyProject/2s-AGCN-0.0/model/agcn.py" \
  --model_class "Model" \
  --model_kwargs '{"num_class": 6, "in_channels": 3, "graph_args": {"layout":"ntu-rgb+d","strategy":"spatial"}, "edge_importance_weighting": true}' \
  --base_ckpt "/content/drive/MyDrive/ToyProject/pretrained/agcn_full.pt"

"""

import os, re, glob, csv, json, math, time, argparse, random, pickle, yaml
from typing import Optional, List, Tuple, Iterable, Dict
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter  # TensorBoard

# ---------------------- Utilities ----------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def accuracy_from_pred(pred_label: torch.Tensor, target: torch.Tensor) -> float:
    return (pred_label == target).float().mean().item()

def f1_score_from_pred(pred_label: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    tp = sum(((pred_label==c)&(target==c)).sum().item() for c in range(num_classes))
    fp = sum(((pred_label==c)&(target!=c)).sum().item() for c in range(num_classes))
    fn = sum(((pred_label!=c)&(target==c)).sum().item() for c in range(num_classes))
    precision = tp/(tp+fp+1e-9); recall = tp/(tp+fn+1e-9)
    return 2*precision*recall/(precision+recall+1e-9)

def f1_score(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    pred_label = logits.argmax(dim=1)
    return f1_score_from_pred(pred_label, target, num_classes)

# ---------------------- Dataset ----------------------

class FolderFeeder(Dataset):
    def __init__(self, data_dir: str, pattern: str = "*_data_joint*.npy",
                 label_from: str = "sidecar",
                 master_label_path: Optional[str] = None,
                 window_size: int = 300,
                 use_mmap: bool = True,
                 label_dir: Optional[str] = None,
                 filename_label_regex: str = r"_y(\d+)\.npy$"):
        self.data_dir = data_dir
        self.pattern = pattern
        self.label_from = label_from
        self.master_label_path = master_label_path
        self.window_size = int(window_size)
        self.use_mmap = use_mmap
        self.label_dir = label_dir
        self.filename_label_regex = filename_label_regex
        self._fname_re = re.compile(self.filename_label_regex)
        self.files = sorted(glob.glob(os.path.join(data_dir, pattern), recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No NPY matched: {os.path.join(data_dir, pattern)}")
        self.labels = {}
        if self.label_from == "master":
            if not self.master_label_path or not os.path.isfile(self.master_label_path):
                raise FileNotFoundError(f"master_label_path not found: {self.master_label_path}")
            if self.master_label_path.lower().endswith(".csv"):
                with open(self.master_label_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        self.labels[r["filename"]] = int(r["label"])
            else:
                raise ValueError("master_label_path must be a CSV with columns [filename,label]")

    def __len__(self): return len(self.files)

    def _load_label_sidecar(self, npy_path: str) -> int:
        base = os.path.basename(npy_path)
        stem_no_ext = os.path.splitext(base)[0]
        stem = stem_no_ext.replace("_data_joint", "")
        candidates = [f"{stem}_label.pkl", f"{stem_no_ext}.pkl"]

        if self.label_dir:
            for name in candidates:
                cand = os.path.join(self.label_dir, name)
                if os.path.isfile(cand):
                    return self._parse_label_pkl(cand)
            for name in candidates:
                hits = glob.glob(os.path.join(self.label_dir, "**", name), recursive=True)
                if hits:
                    return self._parse_label_pkl(hits[0])
        for name in candidates:
            cand = os.path.join(os.path.dirname(npy_path), name)
            if os.path.isfile(cand):
                return self._parse_label_pkl(cand)
        raise FileNotFoundError(f"Sidecar label not found for {npy_path}")

    @staticmethod
    def _parse_label_pkl(pkl_path: str) -> int:
        with open(pkl_path, "rb") as f:
            lab = pickle.load(f)
        if isinstance(lab, dict):
            if "label_index" in lab:
                lab = lab["label_index"]
            else:
                for k in ("label","y","class","target","idx","id"):
                    if k in lab:
                        lab = lab[k]; break
        if isinstance(lab, (list, tuple, np.ndarray)):
            if len(lab)==0: raise ValueError(f"Empty label in {pkl_path}")
            lab = lab[0]
        return int(lab)

    def _label_from_filename(self, npy_path: str) -> int:
        m = self._fname_re.search(os.path.basename(npy_path))
        if not m:
            raise ValueError(f"Cannot extract label from filename {os.path.basename(npy_path)} with {self.filename_label_regex}")
        return int(m.group(1))

    def __getitem__(self, idx: int):
        path = self.files[idx]
        x = np.load(path, mmap_mode="r" if self.use_mmap else None)
        if x.ndim == 5 and x.shape[0] == 1: x = x[0]
        if x.ndim != 4: raise ValueError(f"Expected (C,T,V,M) or (1,C,T,V,M), got {x.shape} from {path}")
        C,T,V,M = x.shape
        if self.window_size > 0:
            if T < self.window_size:
                pad = np.zeros((C, self.window_size, V, M), dtype=x.dtype)
                pad[:, :T] = x; x = pad
            elif T > self.window_size:
                x = x[:, :self.window_size]
        if self.label_from == "sidecar":
            y = self._load_label_sidecar(path)
        elif self.label_from == "master":
            fname = os.path.basename(path)
            if fname not in self.labels: raise KeyError(f"{fname} not in master label map")
            y = self.labels[fname]
        elif self.label_from == "filename":
            y = self._label_from_filename(path)
        else:
            raise ValueError(f"Unknown label_from: {self.label_from}")
        x = np.asarray(x, dtype=np.float32).copy()
        return torch.from_numpy(x), torch.tensor(int(y), dtype=torch.long), idx

# ---------------------- Dynamic model import ----------------------

def dynamic_import(py_path: str, class_name: str):
    import importlib.util, sys
    py_path = str(py_path)
    spec = importlib.util.spec_from_file_location("__dyn_model__", py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["__dyn_model__"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {py_path}")
    return getattr(mod, class_name)

def strip_common_prefixes(sd: dict):
    out = {}
    for k,v in sd.items():
        k2 = k
        if isinstance(k2, str) and k2.startswith("module."): k2 = k2[7:]
        if isinstance(k2, str) and k2.startswith("model."):  k2 = k2[6:]
        out[k2] = v
    return out

# ---------------------- Train utils ----------------------

@torch.no_grad()
def apply_other_gate(logits: torch.Tensor, top1_min: float = 0.8,
                     valid_class_start: int = 1, valid_class_end: Optional[int] = None) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    C = probs.size(1)
    end = valid_class_end if valid_class_end is not None else (C - 1)
    main = probs[:, valid_class_start:end+1]
    maxp, _ = main.max(dim=1)
    pred = probs.argmax(dim=1)
    pred[maxp < top1_min] = 0
    return pred

def run_epoch(loader, model, criterion, optimizer=None, device="cuda", num_classes=6,
              quiet=False, apply_gate=False, top1_min=0.8, valid_class_span=(1,5)):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, total_acc, total_f1, n = 0.0, 0.0, 0.0, 0
    pbar = loader if quiet else tqdm(loader, leave=False)
    vstart, vend = valid_class_span

    for data, label, _ in pbar:
        data, label = data.to(device), label.to(device)
        if train_mode: optimizer.zero_grad()
        logits = model(data)  # FULL forward (backbone + head)
        loss = criterion(logits, label)
        if train_mode:
            loss.backward()
            optimizer.step()

        if apply_gate:
            pred_label = apply_other_gate(logits.detach(), top1_min=top1_min,
                                          valid_class_start=vstart, valid_class_end=vend)
        else:
            pred_label = logits.detach().argmax(dim=1)

        total_loss += loss.item() * len(label)
        total_acc  += (pred_label == label).float().sum().item()
        total_f1   += f1_score_from_pred(pred_label, label.detach(), num_classes) * len(label)
        n += len(label)
        if not quiet:
            pbar.set_description(f"{'Train' if train_mode else 'Valid'} "
                                 f"loss {total_loss/max(n,1):.4f} acc {total_acc/max(n,1):.4f}")
    return total_loss/max(n,1), total_acc/max(n,1), total_f1/max(n,1)

def _save_ckpt_if_in_range(args, model, optimizer, epoch, metric_value, val_loss=None, save_dir=None, fold_idx=None):
    try:
        lower = float(getattr(args, "save_acc_lower", 0.0))
        upper = float(getattr(args, "save_acc_upper", 1.0))
    except Exception:
        lower, upper = 0.0, 1.0

    if metric_value is None:
        print(f"[RANGE-SAVE] metric=None -> skip"); return None
    try: mv = float(metric_value)
    except Exception: print(f"[RANGE-SAVE] metric cast fail -> skip"); return None

    print(f"[RANGE-SAVE] epoch={epoch} metric={mv:.4f} (range {lower}~{upper})")
    if lower <= mv <= upper:
        os.makedirs(save_dir, exist_ok=True)
        fold_tag = f"_f{fold_idx}" if fold_idx is not None else ""
        ckpt_name = f"ckpt_e{int(epoch):03d}_metric{mv:.4f}{fold_tag}.pt"
        path = os.path.join(save_dir, ckpt_name)
        torch.save({
            "epoch": int(epoch),
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "metric": mv,
            "val_loss": float(val_loss) if val_loss is not None else None,
            "args": vars(args) if hasattr(args, "__dict__") else None
        }, path)
        with open(path.replace(".pt", ".json"), "w", encoding="utf-8") as f:
            json.dump({"epoch": int(epoch), "metric": mv,
                       "val_loss": float(val_loss) if val_loss is not None else None,
                       "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, ensure_ascii=False, indent=2)
        print(f"[RANGE-SAVE] saved -> {path}")
        return path
    else:
        print(f"[RANGE-SAVE] out-of-range -> skip")
        return None

def _pick_metric_for_save(args, scope: dict):
    em = getattr(args, "early_metric", "acc").lower()
    if em in ("acc", "accuracy"):
        prefs = ["va_acc", "valid_acc", "val_acc", "acc"]
    elif em in ("f1", "macro_f1", "f1_macro"):
        prefs = ["val_f1", "macro_f1", "f1"]
    else:
        prefs = ["va_loss", "valid_loss", "val_loss", "loss"]
    for name in prefs:
        if name in scope and scope[name] is not None:
            return scope[name], name
    for k, v in scope.items():
        try:
            if isinstance(v, (int, float)): return float(v), k
        except Exception:
            pass
    return None, None

# ---------------------- Trainable selection ----------------------

def set_trainable_by_prefix(model: nn.Module, prefixes: Iterable[str]) -> Dict[str, int]:
    """
    Freeze all params, then unfreeze those whose names start with any of prefixes.
    Returns count dict: {"total":.., "trainable":.., "frozen":..}
    """
    prefixes = [p.strip() for p in prefixes if p.strip()]
    total = 0; trainable = 0
    for name, p in model.named_parameters():
        total += 1
        p.requires_grad = False
        for pref in prefixes:
            if name.startswith(pref):
                p.requires_grad = True
                trainable += 1
                break
    return {"total": total, "trainable": trainable, "frozen": total - trainable}

def trainable_params(model: nn.Module) -> Iterable[nn.Parameter]:
    for p in model.parameters():
        if p.requires_grad:
            yield p

# ---------------------- Main ----------------------

def main():
    parser = argparse.ArgumentParser()
    # Original CLI args kept
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--k_folds', type=int, default=3)
    parser.add_argument('--single_split_train_ratio', type=float, default=0.85)
    parser.add_argument('--early_metric', type=str, choices=['acc','loss','f1'], default='acc')
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--min_delta', type=float, default=0.0)
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, choices=['auto','cuda','cpu'], default='auto')
    parser.add_argument('--tb_logdir', type=str, default=None)

    parser.add_argument('--save_acc_lower', type=float, default=0.85)
    parser.add_argument('--save_acc_upper', type=float, default=0.96)

    parser.add_argument('--use_folder_feeder', type=int, default=1)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--pattern', type=str, default="*.npy")
    parser.add_argument('--label_from', type=str, default='sidecar', choices=['filename','sidecar','master'])
    parser.add_argument('--master_label_path', type=str, default=None)
    parser.add_argument('--label_dir', type=str, default=None)
    parser.add_argument('--filename_label_regex', type=str, default=r"_y(\d+)\.npy$")
    parser.add_argument('--folder_window_size', type=int, default=300)
    parser.add_argument('--quiet_console', type=int, default=1)

    parser.add_argument('--apply_other_gate', type=int, default=1)
    parser.add_argument('--other_top1_min', type=float, default=0.8)
    parser.add_argument('--other_valid_first', type=int, default=1)
    parser.add_argument('--other_valid_last', type=int, default=5)

    # New: full model import & FT-only settings
    parser.add_argument('--model_py', type=str, required=True, help="Python file path containing FULL model class (e.g., model/agcn.py)")
    parser.add_argument('--model_class', type=str, required=True, help="Class name in --model_py (e.g., Model)")
    parser.add_argument('--model_kwargs', type=str, default="", help="JSON string for model ctor kwargs (overrides cfg.model_args)")
    parser.add_argument('--base_ckpt', type=str, default="", help="(Optional) base FULL checkpoint to initialize before FT")
    parser.add_argument('--ft_train_prefixes', type=str, default="classifier.,fc.,head.",
                        help="Comma-separated prefixes to keep trainable (others frozen)")

    args = parser.parse_args()
    set_seed(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    model_args_from_cfg = dict(cfg.get('model_args', {}))
    # num_class fallback -> num_classes
    if 'num_classes' not in model_args_from_cfg and 'num_class' in model_args_from_cfg:
        model_args_from_cfg['num_classes'] = model_args_from_cfg['num_class']

    # Data
    full = FolderFeeder(
        data_dir=args.data_dir, pattern=args.pattern,
        label_from=args.label_from, master_label_path=args.master_label_path,
        window_size=args.folder_window_size, use_mmap=True,
        label_dir=args.label_dir, filename_label_regex=args.filename_label_regex
    )
    print(f"Total samples: {len(full)}")

    # Save root
    save_root = args.save_dir or os.path.join(os.path.dirname(args.config), "runs_full_fc_ft")
    os.makedirs(save_root, exist_ok=True)

    # TensorBoard writer (root). For k-fold, child writers will be created per fold.
    root_writer = SummaryWriter(log_dir=args.tb_logdir) if args.tb_logdir else None

    # Determine num_classes from cfg or model kwargs
    num_classes = int(model_args_from_cfg.get('num_classes', 6))

    # Build full model
    ModelClass = dynamic_import(args.model_py, args.model_class)

    # Merge model_kwargs: cfg.model_args < args.model_kwargs (JSON)
    if args.model_kwargs:
        try:
            cli_kwargs = json.loads(args.model_kwargs)
        except Exception as e:
            raise ValueError(f"--model_kwargs JSON parse error: {e}")
    else:
        cli_kwargs = {}
    model_kwargs = dict(model_args_from_cfg)
    model_kwargs.update(cli_kwargs)
    # Ensure num_classes/num_class present
    if 'num_classes' in model_kwargs:
        pass
    elif 'num_class' in model_kwargs:
        model_kwargs['num_classes'] = model_kwargs['num_class']
    else:
        model_kwargs['num_classes'] = num_classes

    # Instantiate
    
    # --- Instantiate with signature-aware kwargs (handle num_class vs num_classes, drop unknowns) ---
    import inspect
    sig = inspect.signature(ModelClass.__init__)
    param_names = set(sig.parameters.keys()) - {"self"}
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    # Harmonize class count key
    if "num_classes" in model_kwargs and "num_class" not in model_kwargs and "num_class" in param_names:
        model_kwargs["num_class"] = model_kwargs.pop("num_classes")
    if "num_class" in model_kwargs and "num_classes" not in model_kwargs and "num_classes" in param_names:
        model_kwargs["num_classes"] = model_kwargs.pop("num_class")

    # If constructor doesn't accept arbitrary kwargs, drop unknown keys
    if not has_var_kw:
        model_kwargs = {k: v for k, v in model_kwargs.items() if k in param_names}

    model = ModelClass(**model_kwargs).to(device)


    # (Optional) Load base FULL checkpoint (drop head & mismatched shapes)
    if args.base_ckpt:
        sd = torch.load(args.base_ckpt, map_location='cpu')
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = strip_common_prefixes(sd)
        model_sd = model.state_dict()
        drop_prefixes = ("classifier.", "fc.", "head.")
        kept, dropped_head, dropped_shape = 0, 0, 0
        filtered = {}
        for k, v in sd.items():
            # 1) 클래스 수 달라지는 헤드 키는 제외
            if any(k.startswith(p) for p in drop_prefixes):
                dropped_head += 1
                continue
            # 2) 모양 다른 텐서도 제외
            if k in model_sd and model_sd[k].shape != v.shape:
                dropped_shape += 1
                continue
            filtered[k] = v
            kept += 1
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"[BASE] loaded(backbone): {args.base_ckpt} | kept={kept} dropped_head={dropped_head} dropped_shape={dropped_shape} | missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[BASE] (none) random init for backbone+head")

    # Freeze backbone and keep only FC (by prefixes) trainable
    prefixes = [s.strip() for s in args.ft_train_prefixes.split(",") if s.strip()]
    counts = set_trainable_by_prefix(model, prefixes)
    print(f"[FT] trainable selection: total={counts['total']} trainable={counts['trainable']} frozen={counts['frozen']}")
    if counts['trainable'] == 0:
        print("[WARN] 0 trainable tensors. Check --ft_train_prefixes to match your model's head (e.g., classifier.,fc.,head.)")

    # Prepare loaders
    batch_size = int(cfg.get('batch_size', 32))
    test_bs   = int(cfg.get('test_batch_size', batch_size))
    crit = nn.CrossEntropyLoss()

    if args.k_folds <= 1:
        # Single split
        n_total = len(full)
        tr_ratio = max(0.5, min(0.99, float(args.single_split_train_ratio)))
        n_tr = int(round(n_total * tr_ratio))
        n_va = n_total - n_tr
        if n_tr == 0 or n_va == 0:
            n_tr = max(1, min(n_total-1, n_tr if n_tr>0 else 1))
            n_va = n_total - n_tr
        gen = torch.Generator().manual_seed(args.seed)
        train_ds, valid_ds = random_split(full, [n_tr, n_va], generator=gen)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=test_bs, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

        optimizer = torch.optim.Adam(list(trainable_params(model)),
                                     lr=float(cfg.get('base_lr',1e-3)),
                                     weight_decay=float(cfg.get('weight_decay',1e-4)))

        best_metric = -1e9 if args.early_metric=='acc' else (1e9 if args.early_metric=='loss' else -1e9)
        bad_count = 0
        log_path = os.path.join(save_root, "train_single.log")
        with open(log_path, "w") as lf:
            for epoch in range(1, args.num_epoch+1):
                tr_loss, tr_acc, tr_f1 = run_epoch(
                    train_loader, model, crit, optimizer, device, num_classes,
                    quiet=bool(args.quiet_console),
                    apply_gate=bool(args.apply_other_gate), top1_min=args.other_top1_min,
                    valid_class_span=(args.other_valid_first, args.other_valid_last)
                )
                va_loss, va_acc, va_f1 = run_epoch(
                    valid_loader, model, crit, None, device, num_classes,
                    quiet=bool(args.quiet_console),
                    apply_gate=bool(args.apply_other_gate), top1_min=args.other_top1_min,
                    valid_class_span=(args.other_valid_first, args.other_valid_last)
                )
                line = (f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} "
                        f"|| valid loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f}")
                if args.apply_other_gate:
                    line += f" | OTHER_GATE top1_min={args.other_top1_min}"
                print(line); lf.write(line+"\n"); lf.flush()

                # TensorBoard scalars
                if root_writer:
                    root_writer.add_scalar("train/loss", tr_loss, epoch)
                    root_writer.add_scalar("train/acc",  tr_acc,  epoch)
                    root_writer.add_scalar("train/f1",   tr_f1,   epoch)
                    root_writer.add_scalar("valid/loss", va_loss, epoch)
                    root_writer.add_scalar("valid/acc",  va_acc,  epoch)
                    root_writer.add_scalar("valid/f1",   va_f1,   epoch)
                    # Since only head is trainable, LR comes from the optimizer on head
                    lr0 = optimizer.param_groups[0]["lr"]
                    root_writer.add_scalar("opt/lr", lr0, epoch)

                mv, mv_name = _pick_metric_for_save(args, locals())
                _ = _save_ckpt_if_in_range(args, model, optimizer, epoch, mv,
                                           val_loss=va_loss, save_dir=save_root, fold_idx=None)

                if args.early_metric == 'acc':
                    cur_metric = va_acc; is_better = (cur_metric > best_metric)
                elif args.early_metric == 'f1':
                    cur_metric = va_f1; is_better = (cur_metric > best_metric)
                else:
                    cur_metric = va_loss; is_better = (cur_metric < best_metric)

                if is_better:
                    best_metric = cur_metric; bad_count = 0
                    torch.save(model.state_dict(), os.path.join(save_root, "best.pt"))
                else:
                    bad_count += 1
                    if bad_count >= args.patience and epoch >= 5:
                        print("Early stop (single split).")
                        break

        if root_writer: root_writer.flush(); root_writer.close()
        return

    # K-fold
    idxs = np.arange(len(full))
    np.random.shuffle(idxs)
    folds = np.array_split(idxs, args.k_folds)

    for k in range(args.k_folds):
        valid_idx = set(folds[k].tolist())
        train_idx = [i for i in idxs if i not in valid_idx]

        train_ds = Subset(full, train_idx)
        valid_ds = Subset(full, list(valid_idx))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=test_bs, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True)

        optimizer = torch.optim.Adam(list(trainable_params(model)),
                                     lr=float(cfg.get('base_lr',1e-3)),
                                     weight_decay=float(cfg.get('weight_decay',1e-4)))

        best_metric = -1e9 if args.early_metric=='acc' else (1e9 if args.early_metric=='loss' else -1e9)
        bad_count = 0

        fold_dir = os.path.join(save_root, f"fold_{k+1}")
        os.makedirs(fold_dir, exist_ok=True)
        # fold별 TensorBoard 서브 디렉토리
        fold_writer = SummaryWriter(log_dir=os.path.join(args.tb_logdir, f"fold_{k+1}")) if args.tb_logdir else None

        log_path = os.path.join(fold_dir, "train.log")
        with open(log_path, "w") as lf:
            for epoch in range(1, args.num_epoch+1):
                tr_loss, tr_acc, tr_f1 = run_epoch(
                    train_loader, model, crit, optimizer, device, num_classes,
                    quiet=bool(args.quiet_console),
                    apply_gate=bool(args.apply_other_gate), top1_min=args.other_top1_min,
                    valid_class_span=(args.other_valid_first, args.other_valid_last)
                )
                va_loss, va_acc, va_f1 = run_epoch(
                    valid_loader, model, crit, None, device, num_classes,
                    quiet=bool(args.quiet_console),
                    apply_gate=bool(args.apply_other_gate), top1_min=args.other_top1_min,
                    valid_class_span=(args.other_valid_first, args.other_valid_last)
                )

                line = (f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} "
                        f"|| valid loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f}")
                if args.apply_other_gate:
                    line += f" | OTHER_GATE top1_min={args.other_top1_min}"
                print(line); lf.write(line+"\n"); lf.flush()

                if fold_writer:
                    fold_writer.add_scalar("train/loss", tr_loss, epoch)
                    fold_writer.add_scalar("train/acc",  tr_acc,  epoch)
                    fold_writer.add_scalar("train/f1",   tr_f1,   epoch)
                    fold_writer.add_scalar("valid/loss", va_loss, epoch)
                    fold_writer.add_scalar("valid/acc",  va_acc,  epoch)
                    fold_writer.add_scalar("valid/f1",   va_f1,   epoch)
                    lr0 = optimizer.param_groups[0]["lr"]
                    fold_writer.add_scalar("opt/lr", lr0, epoch)

                mv, mv_name = _pick_metric_for_save(args, locals())
                _ = _save_ckpt_if_in_range(args, model, optimizer, epoch, mv,
                                           val_loss=va_loss, save_dir=fold_dir, fold_idx=k+1)

                if args.early_metric == 'acc':
                    cur_metric = va_acc; is_better = (cur_metric > best_metric)
                elif args.early_metric == 'f1':
                    cur_metric = va_f1; is_better = (cur_metric > best_metric)
                else:
                    cur_metric = va_loss; is_better = (cur_metric < best_metric)

                if is_better:
                    best_metric = cur_metric; bad_count = 0
                    torch.save(model.state_dict(), os.path.join(fold_dir, "best.pt"))
                else:
                    bad_count += 1
                    if bad_count >= args.patience and epoch >= 5:
                        print(f"Early stop on fold {k+1} at epoch {epoch}")
                        break

        if fold_writer:
            fold_writer.flush(); fold_writer.close()

    if root_writer:
        root_writer.flush(); root_writer.close()

if __name__ == "__main__":
    main()
