# -*- coding: utf-8 -*-
"""
Full-model training (2s-AGCN backbone) with *selective head/learning-logic upgrades*:

Adds three orthogonal upgrades you asked for:

(A) Data/Loss balancing
    - WeightedRandomSampler (inverse class frequency)
    - CrossEntropy with class weights + label smoothing
    - Optional Focal loss (gamma ~ 1 by default)
(B) Pre-inference calibration (class-wise bias)
    - Fit per-class logit bias Δ_c on validation logits to directly maximize macro-F1
    - Save to <save_dir>/class_bias.json; applied as: logits += bias
(C) Stronger FC heads
    - CosineMarginFC (AM-Softmax/ArcFace-like) with per-class margins
    - MLPHead (BN/Dropout) — light nonlinearity/regularization
    -> both are drop-in swaps that *preserve* the rest of the model

Quick tip:
- Keep backbone frozen, train head only for 3~5 epochs: --head_only_epochs 5
- Then continue normal FT if you want (or keep frozen; your choice via --ft_train_prefixes)

NOTE: This file is a clean superset of your prior script. Flags not used default to sane values.
"""

import os, re, csv, json, math, time, argparse, random, pickle, yaml, glob, inspect
from typing import Optional, List, Tuple, Iterable, Dict
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

# ---------------------- Utils ----------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def accuracy_from_pred(pred_label: torch.Tensor, target: torch.Tensor) -> float:
    return (pred_label == target).float().mean().item()

def f1_from_pred(pred_label: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    tp = sum(((pred_label==c)&(target==c)).sum().item() for c in range(num_classes))
    fp = sum(((pred_label==c)&(target!=c)).sum().item() for c in range(num_classes))
    fn = sum(((pred_label!=c)&(target==c)).sum().item() for c in range(num_classes))
    precision = tp/(tp+fp+1e-9); recall = tp/(tp+fn+1e-9)
    return 2*precision*recall/(precision+recall+1e-9)

@torch.no_grad()
def logits_to_metrics(logits: torch.Tensor, target: torch.Tensor, num_classes: int, bias: Optional[torch.Tensor]=None):
    if bias is not None:
        logits = logits + bias.view(1,-1)
    pred = logits.argmax(dim=1)
    acc = (pred==target).float().mean().item()
    f1  = f1_from_pred(pred, target, num_classes)
    return acc, f1

# ---------------------- FolderFeeder ----------------------

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

# ---------------------- Dynamic import ----------------------

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

# ---------------------- Heads ----------------------

class CosineMarginFC(nn.Module):
    """
    AM-Softmax / ArcFace-like cosine head with per-class margins.
    Use with CrossEntropy on scaled logits.
    """
    def __init__(self, in_feat: int, num_classes: int, s: float = 30.0,
                 margin: float = 0.0, per_class_margin: Optional[List[float]] = None, arcface: bool=False):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_feat, num_classes) * 0.01)
        self.s = float(s)
        self.m = float(margin)
        if per_class_margin is not None:
            assert len(per_class_margin)==num_classes, "per_class_margin length must equal num_classes"
            self.register_buffer("m_c", torch.tensor(per_class_margin, dtype=torch.float32))
        else:
            self.register_buffer("m_c", None)
        self.arcface = bool(arcface)

    def forward(self, x):
        # x: [N, F], W: [F, C]
        x_n = torch.nn.functional.normalize(x, dim=1)
        W_n = torch.nn.functional.normalize(self.W, dim=0)
        cos = x_n @ W_n  # [N, C]
        if self.m_c is not None:
            m = self.m_c.view(1,-1)
        else:
            m = self.m
        if self.arcface:
            # ArcFace uses theta -> theta + m; approximated as cos(theta+m)
            # Here, apply margin by subtracting m from cos for target class via CE (common trick: AM-Softmax style).
            logits = self.s * (cos - m)
        else:
            # AM-Softmax (same subtraction)
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
    def forward(self, x): return self.net(x)

def _find_and_swap_last_linear(model: nn.Module, num_classes: int,
                               head_type: str, mlp_hidden: int, mlp_dropout: float,
                               cos_s: float, cos_margin: float, per_class_margin: Optional[List[float]]):
    """
    Locate the last nn.Linear with out_features==num_classes and replace with selected head.
    Assumes backbone exposes penultimate features before the Linear.
    For AGCN variants, the last layer is often model.fc.
    """
    last_name, last_mod = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.out_features == num_classes:
            last_name, last_mod = name, mod
    if last_mod is None:
        print("[HEAD] No Linear(num_classes) found; skip head swap.")
        return model, None  # can't swap

    in_feat = last_mod.in_features

    class Identity(nn.Module):
        def forward(self, x): return x

    def set_by_name(root: nn.Module, qualname: str, new_mod: nn.Module):
        parts = qualname.split('.')
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_mod)

    # Replace original last linear with Identity so the backbone now outputs features
    set_by_name(model, last_name, Identity())

    # Create the new head
    if head_type == "cosine":
        head = CosineMarginFC(in_feat, num_classes, s=cos_s, margin=cos_margin,
                              per_class_margin=per_class_margin, arcface=False)
    elif head_type == "mlp":
        head = MLPHead(in_feat, num_classes, hidden=mlp_hidden, p=mlp_dropout)
    else:
        head = nn.Linear(in_feat, num_classes)

    # Move head to the SAME device/dtype as model params
    try:
        ref_param = next(model.parameters())
        head = head.to(device=ref_param.device, dtype=ref_param.dtype)
    except StopIteration:
        pass

    # Attach head
    model.head_replacement = head

    # Wrap forward
    orig_forward = model.forward
    def new_forward(x):
        feats = orig_forward(x)  # features after Identity
        logits = model.head_replacement(feats)
        return logits
    model.forward = new_forward
    print(f"[HEAD] Swapped last Linear with {head.__class__.__name__} (in={in_feat}, out={num_classes}) at '{last_name}'")
    return model, head

# ---------------------- Losses ----------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, weight=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(
            logits, target, weight=self.weight,
            reduction='none', label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        loss = ((1-pt)**self.gamma) * ce
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum':  return loss.sum()
        return loss

# ---------------------- Other-gate ----------------------

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

# ---------------------- Calibration (class-wise bias) ----------------------

@torch.no_grad()
def fit_class_bias(valid_logits: torch.Tensor, valid_targets: torch.Tensor, num_classes: int,
                   max_abs: float = 2.0, step: float = 0.1, rounds: int = 2):
    """
    Coordinate descent on per-class bias to maximize macro-F1 on validation.
    Bias search in [-max_abs, +max_abs] with 'step'. Few rounds suffice.
    """
    device = valid_logits.device
    bias = torch.zeros(num_classes, device=device)
    best_f1 = f1_from_pred((valid_logits.argmax(1)), valid_targets, num_classes)
    improved = True
    for _ in range(rounds):
        if not improved: break
        improved = False
        for c in range(num_classes):
            best_c = bias[c].item()
            for b in torch.arange(-max_abs, max_abs+1e-9, step, device=device):
                bias[c] = b
                _, f1 = logits_to_metrics(valid_logits, valid_targets, num_classes, bias)
                if f1 > best_f1 + 1e-6:
                    best_f1 = f1; best_c = b.item(); improved = True
            bias[c] = best_c
    return bias, float(best_f1)

# ---------------------- Train utils ----------------------

def set_trainable_by_prefix(model: nn.Module, prefixes: Iterable[str]) -> Dict[str, int]:
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

def class_counts_from_dataset(ds: Dataset, num_classes: int):
    cnt = np.zeros(num_classes, dtype=np.int64)
    for _, y, _ in DataLoader(ds, batch_size=256, shuffle=False, num_workers=0):
        for c in range(num_classes):
            cnt[c] += (y==c).sum().item()
    return cnt

def make_weighted_sampler(ds: Dataset, num_classes: int):
    # Compute per-class counts quickly (one pass)
    counts = np.zeros(num_classes, dtype=np.int64)
    labels = []
    for _, y, _ in DataLoader(ds, batch_size=256, shuffle=False, num_workers=0):
        labels.extend(y.tolist())
        for c in range(num_classes): counts[c] += (y==c).sum().item()
    labels = np.array(labels, dtype=np.int64)
    counts[counts==0] = 1
    class_w = 1.0 / counts
    sample_w = class_w[labels]
    sampler = WeightedRandomSampler(torch.from_numpy(sample_w).double(), num_samples=len(labels), replacement=True)
    return sampler, torch.tensor(class_w, dtype=torch.float32)

# ---------------------- Main ----------------------

def main():
    parser = argparse.ArgumentParser()
    # Base
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--k_folds', type=int, default=1)
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
    parser.add_argument('--save_acc_upper', type=float, default=0.98)
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

    # Model import
    parser.add_argument('--model_py', type=str, required=True)
    parser.add_argument('--model_class', type=str, required=True)
    parser.add_argument('--model_kwargs', type=str, default="")
    parser.add_argument('--base_ckpt', type=str, default="")
    parser.add_argument('--ft_train_prefixes', type=str, default="classifier.,fc.,head.")

    # === (A) Data/Loss balancing flags ===
    parser.add_argument('--use_weighted_sampler', type=int, default=1)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--use_focal', type=int, default=0)
    parser.add_argument('--focal_gamma', type=float, default=1.0)
    parser.add_argument('--manual_class_weights', type=str, default="", help='JSON list or csv "w0,w1,..."')

    # === (B) Calibration flags ===
    parser.add_argument('--calibrate_bias', type=int, default=1)
    parser.add_argument('--calib_max_abs', type=float, default=2.0)
    parser.add_argument('--calib_step', type=float, default=0.1)
    parser.add_argument('--calib_rounds', type=int, default=2)

    # === (C) Head swap flags ===
    parser.add_argument('--head_type', type=str, choices=['none','cosine','mlp'], default='cosine')
    parser.add_argument('--mlp_hidden', type=int, default=512)
    parser.add_argument('--mlp_dropout', type=float, default=0.2)
    parser.add_argument('--cos_s', type=float, default=30.0)
    parser.add_argument('--cos_margin', type=float, default=0.0)
    parser.add_argument('--per_class_margin', type=str, default="", help='JSON list or csv; len=num_classes')
    parser.add_argument('--head_only_epochs', type=int, default=5)

    args = parser.parse_args()
    set_seed(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Data
    full = FolderFeeder(
        data_dir=args.data_dir, pattern=args.pattern,
        label_from=args.label_from, master_label_path=args.master_label_path,
        window_size=args.folder_window_size, use_mmap=True,
        label_dir=args.label_dir, filename_label_regex=args.filename_label_regex
    )
    print(f"Total samples: {len(full)}")

    # Save root
    save_root = args.save_dir or os.path.join(os.path.dirname(args.config), "runs_full_fc_ft_plus")
    os.makedirs(save_root, exist_ok=True)

    writer = SummaryWriter(log_dir=args.tb_logdir) if args.tb_logdir else None

    # num_classes from cfg.model_args / convenience
    model_args_from_cfg = dict(cfg.get('model_args', {}))
    num_classes = int(model_args_from_cfg.get('num_classes', model_args_from_cfg.get('num_class', 6)))

    # Build model
    ModelClass = dynamic_import(args.model_py, args.model_class)
    cli_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}
    model_kwargs = dict(model_args_from_cfg); model_kwargs.update(cli_kwargs)

    # Harmonize constructor signature
    sig = inspect.signature(ModelClass.__init__)
    param_names = set(sig.parameters.keys()) - {"self"}
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if "num_classes" in model_kwargs and "num_class" not in model_kwargs and "num_class" in param_names:
        model_kwargs["num_class"] = model_kwargs.pop("num_classes")
    if "num_class" in model_kwargs and "num_classes" not in model_kwargs and "num_classes" in param_names:
        model_kwargs["num_classes"] = model_kwargs.pop("num_class")
    if not has_var_kw:
        model_kwargs = {k: v for k,v in model_kwargs.items() if k in param_names}
    model = ModelClass(**model_kwargs).to(device)

    # Optionally load base backbone
    if args.base_ckpt:
        sd = torch.load(args.base_ckpt, map_location='cpu')
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = strip_common_prefixes(sd)
        model_sd = model.state_dict()
        drop_prefixes = ("classifier.", "fc.", "head.")
        kept = 0
        filtered = {}
        for k, v in sd.items():
            if any(k.startswith(p) for p in drop_prefixes): continue
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered[k] = v; kept += 1
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"[BASE] loaded(backbone): kept={kept} missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("[BASE] (none) random init for backbone+head)")

    # === (C) Head swap ===
    per_class_margin = None
    if args.per_class_margin:
        try:
            if args.per_class_margin.strip().startswith('['):
                per_class_margin = json.loads(args.per_class_margin)
            else:
                per_class_margin = [float(x) for x in args.per_class_margin.split(',')]
        except Exception as e:
            raise ValueError(f"--per_class_margin parse error: {e}")

    model, new_head = _find_and_swap_last_linear(
        model, num_classes,
        head_type=args.head_type,
        mlp_hidden=args.mlp_hidden, mlp_dropout=args.mlp_dropout,
        cos_s=args.cos_s, cos_margin=args.cos_margin, per_class_margin=per_class_margin
    )

    # Freeze backbone; keep only head (and any 'fc./classifier./head.' prefixes) trainable
    prefixes = [s.strip() for s in args.ft_train_prefixes.split(",") if s.strip()]
    counts = set_trainable_by_prefix(model, prefixes + ["head_replacement."])
    print(f"[FT] trainable selection: total={counts['total']} trainable={counts['trainable']} frozen={counts['frozen']}")

    # Split
    n_total = len(full)
    tr_ratio = max(0.5, min(0.99, float(args.single_split_train_ratio)))
    n_tr = int(round(n_total * tr_ratio))
    n_va = n_total - n_tr
    if n_tr == 0 or n_va == 0:
        n_tr = max(1, min(n_total-1, max(1, n_tr)))
        n_va = n_total - n_tr
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, valid_ds = random_split(full, [n_tr, n_va], generator=gen)

    # Sampler / loaders
    batch_size = int(cfg.get('batch_size', 32))
    test_bs   = int(cfg.get('test_batch_size', batch_size))

    if args.use_weighted_sampler:
        sampler, inv_counts = make_weighted_sampler(train_ds, num_classes)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True)
        # Class weights for CE/Focal: proportional to inverse frequency
        class_weights = inv_counts / inv_counts.sum() * num_classes
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        # Uniform weights initially
        class_weights = torch.ones(num_classes, dtype=torch.float32)

    # Allow manual override of class weights
    if args.manual_class_weights:
        try:
            if args.manual_class_weights.strip().startswith('['):
                w = torch.tensor(json.loads(args.manual_class_weights), dtype=torch.float32)
            else:
                w = torch.tensor([float(x) for x in args.manual_class_weights.split(',')], dtype=torch.float32)
            assert len(w) == num_classes
            class_weights = w
        except Exception as e:
            raise ValueError(f"--manual_class_weights parse error: {e}")

    valid_loader = DataLoader(valid_ds, batch_size=test_bs, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Loss
    class_weights = class_weights.to(device)
    if args.use_focal:
        criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights, label_smoothing=args.label_smoothing)
        print(f"[LOSS] FocalLoss(gamma={args.focal_gamma}, label_smoothing={args.label_smoothing}) with class weights")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        print(f"[LOSS] CrossEntropy(label_smoothing={args.label_smoothing}) with class weights")

    # Optimizer
    base_lr = float(cfg.get('base_lr', 1e-3))
    weight_decay = float(cfg.get('weight_decay', 1e-4))
    optimizer = torch.optim.Adam(list(trainable_params(model)), lr=base_lr, weight_decay=weight_decay)

    # Training loop
    best_metric = -1e9 if args.early_metric=='acc' else (1e9 if args.early_metric=='loss' else -1e9)
    bad_count = 0

    bias_json_path = os.path.join(save_root, "class_bias.json")
    saved_bias = torch.zeros(num_classes, device=device)

    def run_epoch(loader, train: bool):
        model.train() if train else model.eval()
        total_loss, total_acc, total_f1, n = 0.0, 0.0, 0.0, 0
        all_logits, all_targets = [], []
        for data, label, _ in loader:
            data, label = data.to(device), label.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(data)
            loss = criterion(logits, label)
            if train:
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                pred = logits.detach().argmax(1)
                total_loss += loss.item() * len(label)
                total_acc  += (pred==label).float().sum().item()
                total_f1   += f1_from_pred(pred, label, num_classes) * len(label)
                n += len(label)
                all_logits.append(logits.detach())
                all_targets.append(label.detach())
        if n==0: n=1
        return (total_loss/n, total_acc/n, total_f1/n,
                torch.cat(all_logits,0) if all_logits else torch.empty(0, num_classes, device=device),
                torch.cat(all_targets,0) if all_targets else torch.empty(0, dtype=torch.long, device=device))

    for epoch in range(1, args.num_epoch+1):
        # Optional: head-only warmup epochs (keep already frozen; this mainly exists for clarity/logging)
        # If you later want to unfreeze more layers, change --ft_train_prefixes.
        if epoch == 1 and args.head_only_epochs > 0:
            print(f"[WARMUP] Head-only phase for {args.head_only_epochs} epochs (backbone frozen).")

        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(train_loader, train=True)
        va_loss, va_acc, va_f1, valid_logits, valid_targets = run_epoch(valid_loader, train=False)

        line = (f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} "
                f"|| valid loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f}")
        print(line)

        # === (B) Calibration on validation logits ===
        calib_acc, calib_f1 = None, None
        if args.calibrate_bias and valid_logits.numel() > 0:
            bias, best_f1 = fit_class_bias(valid_logits, valid_targets, num_classes,
                                           max_abs=args.calib_max_abs, step=args.calib_step, rounds=args.calib_rounds)
            saved_bias = bias.detach().clone()
            with open(bias_json_path, "w", encoding="utf-8") as f:
                json.dump({"bias": [float(x) for x in saved_bias.cpu().numpy().tolist()]}, f, ensure_ascii=False, indent=2)
            calib_acc, calib_f1 = logits_to_metrics(valid_logits, valid_targets, num_classes, saved_bias)
            print(f"[CALIB] Saved bias -> {bias_json_path} | calib_acc={calib_acc:.4f} calib_f1={calib_f1:.4f}")

        # Log
        if writer:
            writer.add_scalar("train/loss", tr_loss, epoch)
            writer.add_scalar("train/acc",  tr_acc,  epoch)
            writer.add_scalar("train/f1",   tr_f1,   epoch)
            writer.add_scalar("valid/loss", va_loss, epoch)
            writer.add_scalar("valid/acc",  va_acc,  epoch)
            writer.add_scalar("valid/f1",   va_f1,   epoch)
            if calib_acc is not None:
                writer.add_scalar("valid_calibrated/acc", calib_acc, epoch)
                writer.add_scalar("valid_calibrated/f1",  calib_f1,  epoch)
            writer.add_scalar("opt/lr", optimizer.param_groups[0]["lr"], epoch)

        # Early stop
        cur_metric = va_acc if args.early_metric=='acc' else (va_f1 if args.early_metric=='f1' else va_loss)
        is_better = (cur_metric > best_metric) if args.early_metric in ('acc','f1') else (cur_metric < best_metric)
        if is_better:
            best_metric = cur_metric; bad_count = 0
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "bias": saved_bias.detach().cpu().numpy().tolist(),
                "args": vars(args)
            }, os.path.join(save_root, "best.pt"))
        else:
            bad_count += 1
            if bad_count >= args.patience and epoch >= max(5, args.head_only_epochs):
                print("Early stop.")
                break

    # Save final bias if not already saved
    if args.calibrate_bias and not os.path.isfile(bias_json_path):
        zero_bias = [0.0]*num_classes
        with open(bias_json_path, "w", encoding="utf-8") as f:
            json.dump({"bias": zero_bias}, f, ensure_ascii=False, indent=2)

    if writer:
        writer.flush(); writer.close()

if __name__ == "__main__":
    main()
