# data_gen/single_ntu_to_npy.py
import os
import argparse
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
    data = np.zeros((max_body, seq['numFrame'], num_joint, 3))
    for n, f in enumerate(seq['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
    # 상위 2개 body 선택
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]
    # (M,T,V,C) -> (C,T,V,M)
    return data.transpose(3, 1, 2, 0)

def main():
    ap = argparse.ArgumentParser("Single NTU .skeleton -> joint.npy")
    ap.add_argument("--skeleton_file", required=True, help="입력 .skeleton 파일 경로")
    ap.add_argument("--out_npy",       required=True, help="출력 joint.npy 경로")
    ap.add_argument("--out_label",     default=None,  help="(선택) 레이블 pkl 경로")
    ap.add_argument("--label",         type=int, default=0, help="(선택) out_label에 저장할 라벨")
    args = ap.parse_args()

    data = read_xyz(args.skeleton_file)          # (C,T,V,M)
    T = min(max_frame, data.shape[1])            # 안전하게 클리핑
    fp = np.zeros((1, 3, max_frame, num_joint, max_body_true), dtype=np.float32)
    fp[0, :, 0:T, :, :] = data[:, 0:T, :, :]     # 패딩/클리핑 후 복사

    # 원본 파이프라인과 동일 전처리
    fp = pre_normalization(fp)

    os.makedirs(os.path.dirname(args.out_npy), exist_ok=True)
    np.save(args.out_npy, fp)
    print(f"[single] saved npy: {args.out_npy}  shape={fp.shape}  filled_frames={T}")

    if args.out_label:
        with open(args.out_label, "wb") as f:
            pickle.dump(([os.path.basename(args.skeleton_file)], [args.label]), f)
        print(f"[single] saved label: {args.out_label} (label={args.label})")

if __name__ == "__main__":
    main()
