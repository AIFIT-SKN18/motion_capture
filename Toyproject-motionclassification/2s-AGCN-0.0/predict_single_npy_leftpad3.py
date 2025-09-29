# =====================
# File 1/2: predict_single_npy_leftpad_patched.py
# =====================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single NPY inference with:
 - Head re-attachment (Cosine/MLP) if the training script swapped the last Linear
 - Class-wise bias calibration (class_bias.json)
 - Left-aligned window policy (pad/crop to T)
 - OTHER gate always applied after bias (pred <- 0 if max valid prob < threshold)
"""

import os, re, json, argparse, inspect
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import torch
from torch import nn

# ---------------------- Heads (mirror of training script) ----------------------
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

class Identity(nn.Module):
    def forward(self, x):
        return x

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


def attach_head_like_training(model: nn.Module, sd: dict, train_args: Optional[dict],
                              num_classes: int) -> nn.Module:
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

    if head_type == 'mlp':
        head = MLPHead(in_feat, num_classes, hidden=mlp_hidden, p=mlp_dropout)
    else:
        head = CosineMarginFC(in_feat, num_classes, s=cos_s, margin=cos_margin, per_class_margin=per_class_margin)

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


def load_bias_tensor(bias_json_path: Optional[Path], num_classes: int, device: str):
    if bias_json_path and bias_json_path.is_file():
        with open(bias_json_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        b = obj.get('bias', None)
        if isinstance(b, list) and len(b) == num_classes:
            return torch.tensor(b, dtype=torch.float32, device=device)
    return None


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


def apply_other_gate(logits: torch.Tensor, top1_min: float, valid_first: int, valid_last: int) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    C = probs.size(1)
    end = min(valid_last, C-1)
    main = probs[:, valid_first:end+1]
    maxp, _ = main.max(dim=1)
    pred = probs.argmax(dim=1)
    pred[maxp < top1_min] = 0
    return pred

# ---------------------- Main ----------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npy', type=str, required=True)
    p.add_argument('--model_py', type=str, required=True)
    p.add_argument('--model_class', type=str, required=True)
    p.add_argument('--model_kwargs', type=str, default='')
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--num_classes', type=int, default=6)
    p.add_argument('--device', type=str, choices=['auto','cuda','cpu'], default='auto')
    p.add_argument('--window_T', type=int, default=300)
    p.add_argument('--bias_json', type=str, default='')
    # force apply_other_gate always enabled
    p.add_argument('--other_top1_min', type=float, default=0.8)
    p.add_argument('--other_valid_first', type=int, default=1)
    p.add_argument('--other_valid_last', type=int, default=5)
    args = p.parse_args()

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device

    Model = dynamic_import(args.model_py, args.model_class)
    cli_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}
    sig = inspect.signature(Model.__init__)
    param_names = set(sig.parameters.keys()) - {'self'}
    if 'num_classes' in param_names:
        cli_kwargs.setdefault('num_classes', args.num_classes)
    elif 'num_class' in param_names:
        cli_kwargs.setdefault('num_class', args.num_classes)
    model = Model(**{k:v for k,v in cli_kwargs.items() if k in param_names}).to(device)

    obj = torch.load(args.ckpt, map_location=device)
    sd = obj['state_dict'] if isinstance(obj, dict) and 'state_dict' in obj else obj
    train_args = obj.get('args', {}) if isinstance(obj, dict) else {}

    model = attach_head_like_training(model, sd, train_args, args.num_classes)
    model.load_state_dict(sd, strict=False)
    model.eval()

    bias_path = Path(args.bias_json) if args.bias_json else (Path(args.ckpt).parent / 'class_bias.json')
    bias = load_bias_tensor(bias_path, args.num_classes, device)

    x = np.load(args.npy)
    if x.ndim == 5 and x.shape[0] == 1:
        x = x[0]
    x = ensure_leftpad(x, args.window_T).astype('float32', copy=False)
    X = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(X)
        if bias is not None:
            logits = logits + bias.view(1,-1)
        # always apply other_gate
        pred = apply_other_gate(logits, args.other_top1_min, args.other_valid_first, args.other_valid_last)
        probs = torch.softmax(logits, dim=1)
        topk = torch.topk(probs, k=min(5, probs.shape[1]), dim=1)

    print(f"Prediction: class_id={pred.item()}, prob={probs[0,pred.item()].item():.4f}")
    print("Top-5:")
    for i in range(topk.indices.shape[1]):
        cid = topk.indices[0,i].item()
        pv = topk.values[0,i].item()
        print(f" {i+1:>2}. {cid:>6} : {pv:.4f}")

if __name__ == '__main__':
    main()





