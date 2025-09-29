# data_gen/single_ntu_to_npy.py
import os
import argparse
import glob
import re
import numpy as np
import pickle

# 같은 패키지의 전처리 사용
from .preprocess import pre_normalization

# 기본 설정
max_body_true    = 1
max_body_kinect  = 1
num_joint        = 25
max_frame        = 300

def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {'numFrame': int(f.readline()), 'frameInfo': []}
        for _ in range(skeleton_sequence['numFrame']):
            frame_info = {'numBody': int(f.readline()), 'bodyInfo': []}
            for _ in range(frame_info['numBody']):
                body_info_key = [
                    'bodyID','clipedEdges','handLeftConfidence','handLeftState',
                    'handRightConfidence','handRightState','isResticted','leanX','leanY','trackingState'
                ]
                body_info = {k: float(v) for k, v in zip(body_info_key, f.readline().split())}
                body_info['numJoint'] = int(f.readline())
                joint_info_key = [
                    'x','y','z','depthX','depthY','colorX','colorY',
                    'orientationW','orientationX','orientationY','orientationZ','trackingState'
                ]
                joints = []
                for _ in range(body_info['numJoint']):
                    joints.append({k: float(v) for k, v in zip(joint_info_key, f.readline().split())})
                body_info['jointInfo'] = joints
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence

def get_nonzero_std(s):
    idx = s.sum(-1).sum(-1) != 0
    s = s[idx]
    if len(s) != 0:
        return s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()
    return 0

def read_xyz(file, max_body=max_body_kinect, num_joint=num_joint):
    seq = read_skeleton_filter(file)
    data = np.zeros((max_body, seq['numFrame'], num_joint, 3), dtype=np.float32)
    for n, f in enumerate(seq['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
    # 상위 body 선택
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]
    # (M,T,V,C) -> (C,T,V,M)
    return data.transpose(3, 1, 2, 0)

def make_sample_npy(skeleton_file, out_npy_path, T_max=max_frame):
    """단일 .skeleton -> (1,3,T,V,M) npy 저장 (pre_normalization 포함)"""
    data = read_xyz(skeleton_file)                 # (C,T_raw,V,M)
    T = min(T_max, data.shape[1])
    fp = np.zeros((1, 3, T_max, num_joint, max_body_true), dtype=np.float32)
    fp[0, :, 0:T, :, :] = data[:, 0:T, :, :]
    fp = pre_normalization(fp)                     # (1,3,T_max,V,M)
    os.makedirs(os.path.dirname(out_npy_path), exist_ok=True)
    np.save(out_npy_path, fp)
    return T, fp.shape

def infer_label_from_fname(fname):
    """파일명 SxxxCxxxPxxxRxxxAxxx.skeleton 에서 A### → 0-based 라벨"""
    m = re.search(r'A(\d{3})', os.path.basename(fname))
    if not m:
        return None
    return int(m.group(1)) - 1

def expand_input_paths(p):
    """파일/디렉터리/글롭 패턴을 .skeleton 리스트로 확장"""
    files = []
    if os.path.isdir(p):
        for root, _, fnames in os.walk(p):
            for fn in fnames:
                if fn.lower().endswith('.skeleton'):
                    files.append(os.path.join(root, fn))
    else:
        # 파일 또는 글롭
        for x in glob.glob(p):
            if os.path.isdir(x):
                for root, _, fnames in os.walk(x):
                    for fn in fnames:
                        if fn.lower().endswith('.skeleton'):
                            files.append(os.path.join(root, fn))
            else:
                if x.lower().endswith('.skeleton'):
                    files.append(x)
    return sorted(set(files))

def main():
    ap = argparse.ArgumentParser("NTU .skeleton -> joint.npy (single or batch)")
    # 단일 모드 (이전과 호환)
    ap.add_argument("--skeleton_file", help="단일 .skeleton 파일 경로")
    ap.add_argument("--out_npy",       help="단일 출력 joint.npy 경로")
    ap.add_argument("--out_label",     default=None,  help="(선택) 단일 레이블 pkl 경로")
    ap.add_argument("--label",         type=int, default=0, help="(선택) 단일 모드 라벨(0-based)")

    # 배치 모드 (디렉터리/글롭)
    ap.add_argument("--input",   help="파일/디렉터리/글롭 패턴 (배치 모드)")
    ap.add_argument("--out_dir", help="배치 모드 출력 폴더 (개별 파일 저장)")
    ap.add_argument("--label_mode", choices=['fixed','from_fname'], default='fixed',
                    help="배치 라벨 부여 방식: fixed=--label 사용, from_fname=A### 파싱")
    ap.add_argument("--save_label_per_file", action="store_true",
                    help="배치 모드에서 각 파일 옆에 *_label.pkl 저장")
    ap.add_argument("--aggregate_label", default=None,
                    help="배치 모드에서 전체 (names, labels)를 한 pkl로 저장할 경로")

    args = ap.parse_args()

    # -------- 단일 모드 --------
    if args.skeleton_file and args.out_npy:
        T, shape = make_sample_npy(args.skeleton_file, args.out_npy)
        print(f"[single] saved npy: {args.out_npy}  shape={shape}  filled_frames={T}")
        if args.out_label:
            with open(args.out_label, "wb") as f:
                pickle.dump(([os.path.basename(args.skeleton_file)], [args.label]), f)
            print(f"[single] saved label: {args.out_label} (label={args.label})")
        return

    # -------- 배치 모드 --------
    if not args.input or not args.out_dir:
        raise SystemExit(
            "사용법:\n"
            "  단일: --skeleton_file <file> --out_npy <file.npy> [--out_label <file.pkl>] [--label K]\n"
            "  배치: --input <dir_or_glob> --out_dir <out_folder> [--label_mode fixed|from_fname] "
            "[--label K] [--save_label_per_file] [--aggregate_label <all.pkl>]\n"
        )

    files = expand_input_paths(args.input)
    if not files:
        raise SystemExit(f"[ERROR] no .skeleton files found under: {args.input}")

    os.makedirs(args.out_dir, exist_ok=True)

    all_names, all_labels = [], []
    for fpath in files:
        base = os.path.splitext(os.path.basename(fpath))[0]
        out_npy_path = os.path.join(args.out_dir, f"{base}_data_joint.npy")
        T, shape = make_sample_npy(fpath, out_npy_path)
        print(f"[batch] saved: {out_npy_path} shape={shape} frames_used={T}")

        # 라벨 결정
        if args.label_mode == 'fixed':
            lab = args.label
        else:
            lab = infer_label_from_fname(fpath)
            if lab is None:
                lab = 0
        all_names.append(os.path.basename(fpath))
        all_labels.append(int(lab))

        # 파일별 개별 label.pkl 저장(선택)
        if args.save_label_per_file:
            out_lbl_path = os.path.join(args.out_dir, f"{base}_label.pkl")
            with open(out_lbl_path, "wb") as f:
                pickle.dump(([os.path.basename(fpath)], [int(lab)]), f)
            print(f"[batch] saved label: {out_lbl_path} (label={lab})")

    # 전체 라벨 집계 저장(선택)
    if args.aggregate_label:
        with open(args.aggregate_label, "wb") as f:
            pickle.dump((all_names, all_labels), f)
        print(f"[batch] saved aggregate label: {args.aggregate_label} (N={len(all_names)})")

if __name__ == "__main__":
    main()
