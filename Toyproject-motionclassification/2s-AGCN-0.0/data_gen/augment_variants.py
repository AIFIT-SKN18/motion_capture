#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, pickle, random, math
from pathlib import Path
import numpy as np

def _find_axes(arr: np.ndarray):
    shape = arr.shape
    if 3 not in shape:
        raise ValueError(f"Cannot find coordinate channel axis (size 3) in shape {shape}")
    chan_axis = int(np.where(np.array(shape)==3)[0][0])
    remaining = [i for i in range(arr.ndim) if i != chan_axis]
    sizes = [(i, shape[i]) for i in remaining]
    time_axis = max(sizes, key=lambda x: x[1])[0]
    joint_axis = [i for i in remaining if i != time_axis][0]
    return time_axis, joint_axis, chan_axis

def _to_TVC(arr: np.ndarray):
    time_axis, joint_axis, chan_axis = _find_axes(arr)
    tvc = np.moveaxis(arr, (time_axis, joint_axis, chan_axis), (0,1,2))
    tvc = np.ascontiguousarray(tvc)
    def _restore(x_tvc: np.ndarray):
        x = np.moveaxis(x_tvc, (0,1,2), (time_axis, joint_axis, chan_axis))
        return x
    return tvc, _restore

def _deg2rad(d): 
    return d * math.pi / 180.0

def rotate_yaw_pitch_roll_tvc(seq_tvc: np.ndarray, yaw_deg: float, pitch_deg: float, roll_deg: float):
    if yaw_deg == pitch_deg == roll_deg == 0.0:
        return seq_tvc
    yaw, pitch, roll = map(_deg2rad, (yaw_deg, pitch_deg, roll_deg))
    cz, sz = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[ cz,-sz, 0],[ sz, cz, 0],[  0,  0, 1]], dtype=seq_tvc.dtype)
    cy, sy = math.cos(pitch), math.sin(pitch)
    Ry = np.array([[ cy, 0, sy],[  0, 1,  0],[-sy, 0, cy]], dtype=seq_tvc.dtype)
    cx, sx = math.cos(roll), math.sin(roll)
    Rx = np.array([[1,  0,  0],[0, cx,-sx],[0, sx, cx]], dtype=seq_tvc.dtype)
    R = Rz @ Ry @ Rx
    out = np.tensordot(seq_tvc, R, axes=([2],[0]))  # (T,V,3) x (3,3) -> (T,V,3)
    return out

def pad_or_trim_T_tvc(seq_tvc: np.ndarray, T: int):
    curT = seq_tvc.shape[0]
    if curT == T:
        return seq_tvc
    if curT > T:
        return seq_tvc[:T, :, :]
    last = seq_tvc[-1:, :, :]
    reps = T - curT
    pad = np.repeat(last, reps, axis=0)
    return np.concatenate([seq_tvc, pad], axis=0)

def save_outputs(out_dir: Path, stem: str, variant: str, copy_idx: int, joint_arr: np.ndarray, label_obj):
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{variant}_copy{copy_idx:02d}"
    joint_path = out_dir / f"{stem}{suffix}_data_joint.npy"
    np.save(joint_path, joint_arr)
    if label_obj is not None:
        label_path = out_dir / f"{stem}{suffix}_label.pkl"
        with open(label_path, "wb") as f:
            pickle.dump(label_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def apply_variant_tvc(seq_tvc: np.ndarray, variant: str, yaw_deg: float, pitch_deg: float, roll_deg: float):
    v = variant.strip().lower()
    rng = random.Random()
    rng.seed()
    def u(a): return rng.uniform(-a, a)
    if v in ("y","yaw"):   return rotate_yaw_pitch_roll_tvc(seq_tvc, u(yaw_deg), 0.0, 0.0)
    if v in ("p","pitch"): return rotate_yaw_pitch_roll_tvc(seq_tvc, 0.0, u(pitch_deg), 0.0)
    if v in ("r","roll"):  return rotate_yaw_pitch_roll_tvc(seq_tvc, 0.0, 0.0, u(roll_deg))
    if v in ("ypr","rpy","ypr_combo"): return rotate_yaw_pitch_roll_tvc(seq_tvc, u(yaw_deg), u(pitch_deg), u(roll_deg))
    return seq_tvc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_joint", required=True, type=str)
    ap.add_argument("--out_dir",  required=True, type=str)
    ap.add_argument("--dataset",  default="ntu/xsub")
    ap.add_argument("--variants", required=True, type=str)  # e.g. "ypr"
    ap.add_argument("--copies",   required=True, type=int)
    ap.add_argument("--T",        required=True, type=int)
    ap.add_argument("--yaw_deg",   type=float, default=10.0)
    ap.add_argument("--pitch_deg", type=float, default=5.0)
    ap.add_argument("--roll_deg",  type=float, default=5.0)
    ap.add_argument("--in_label",  type=str, default=None)
    args = ap.parse_args()

    in_joint = Path(args.in_joint)
    out_dir  = Path(args.out_dir)
    in_label = Path(args.in_label) if args.in_label else None

    joint_raw = np.load(in_joint, allow_pickle=True)
    label_obj = None
    if in_label and in_label.exists():
        try:
            with open(in_label, "rb") as f:
                label_obj = pickle.load(f)
        except Exception:
            label_obj = None

    seq_tvc, restore = _to_TVC(joint_raw)
    print('SEQ_TVC shape:', seq_tvc.shape)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    stem = in_joint.name
    if stem.endswith("_data_joint.npy"):
        stem = stem[:-len("_data_joint.npy")]

    base_rng = random.getstate()
    for v in variants:
        for k in range(1, args.copies + 1):
            random.seed(hash((stem, v, k)) & 0xFFFFFFFF)
            aug = apply_variant_tvc(seq_tvc, v, args.yaw_deg, args.pitch_deg, args.roll_deg)
            aug = pad_or_trim_T_tvc(aug, args.T)
            aug_out = restore(aug)
            save_outputs(out_dir, stem, v, k, aug_out, label_obj)
    random.setstate(base_rng)

if __name__ == "__main__":
    main()
