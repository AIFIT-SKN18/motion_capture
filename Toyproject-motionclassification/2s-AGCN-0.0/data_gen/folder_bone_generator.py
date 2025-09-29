#!/usr/bin/env python3
# folder_bone_generator.py
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

# ----- joint 연결쌍(Paris) -----
PARIS = {
    'ntu/xview': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7),
        (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
        (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21),
        (23, 8), (24, 25), (25, 12)
    ),
    'ntu/xsub': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7),
        (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
        (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21),
        (23, 8), (24, 25), (25, 12)
    ),
    'kinetics': (
        (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6),
        (8, 2), (9, 8), (10, 9), (11, 5), (12, 11), (13, 12), (14, 0),
        (15, 0), (16, 14), (17, 15)
    ),
}

def compute_bone(joint: np.ndarray, mapping, zero_based: bool) -> np.ndarray:
    """
    joint: (C, T, V, M)
    mapping: list of (v1, v2)
    return: (C, T, V, M) bone 데이터 (v1 := v1 - v2)
    """
    C, T, V, M = joint.shape
    bone = np.copy(joint)
    for (v1, v2) in mapping:
        if not zero_based:  # NTU는 1-based → 0-based 변환
            v1 -= 1
            v2 -= 1
        # 경계 체크 (예외적으로 V보다 크면 skip)
        if not (0 <= v1 < V and 0 <= v2 < V):
            continue
        bone[:, :, v1, :] = joint[:, :, v1, :] - joint[:, :, v2, :]
    return bone

def main():
    ap = argparse.ArgumentParser(description="Convert folder of joint .npy files to bone .npy files (mirror folders).")
    ap.add_argument("--dataset", required=True, choices=list(PARIS.keys()),
                    help="키(ntu/xview | ntu/xsub | kinetics)")
    ap.add_argument("--in_dir", required=True, type=Path,
                    help="입력 루트 폴더 (폴더 안의 모든 .npy 대상)")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="출력 루트 폴더 (미러 구조로 저장)")
    ap.add_argument("--suffix", default="_bone", help="출력 파일명에 붙일 접미사 (기본: _bone)")
    ap.add_argument("--glob", default="**/*.npy", help="검색 패턴 (기본: **/*.npy)")
    ap.add_argument("--overwrite", action="store_true", help="이미 존재해도 덮어쓰기")
    args = ap.parse_args()

    mapping = PARIS[args.dataset]
    zero_based = (args.dataset == "kinetics")  # kinetics는 이미 0-based 인덱스

    in_files = sorted(args.in_dir.glob(args.glob))
    if not in_files:
        print(f"[WARN] 입력 파일이 없습니다: {args.in_dir} ({args.glob})")
        return

    for f in tqdm(in_files, desc="Converting to bone"):
        # 출력 경로: in_dir 기준 상대경로 유지
        rel = f.relative_to(args.in_dir)
        out_path = args.out_dir / rel

        # 파일명에 suffix 부착 (ex: foo.npy → foo_bone.npy)
        if out_path.suffix.lower() == ".npy":
            out_path = out_path.with_name(out_path.stem + args.suffix + out_path.suffix)

        # 스킵/디렉토리 생성
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.overwrite:
            continue

        try:
            joint = np.load(f)  # (C, T, V, M) 또는 (N,C,T,V,M)일 수도 있음
            if joint.ndim == 5:
                # 배치 형태(N, C, T, V, M)면 파일 단위로 처리: 각 샘플별로 저장하는 대신
                # 전체를 한 번에 변환(원 코드 호환). 원하면 per-sample로 분리 저장하도록 변경 가능.
                N, C, T, V, M = joint.shape
                bone = np.empty_like(joint)
                for i in range(N):
                    bone[i] = compute_bone(joint[i], mapping, zero_based)
            elif joint.ndim == 4:
                # (C,T,V,M)
                bone = compute_bone(joint, mapping, zero_based)
            else:
                print(f"[SKIP] 지원하지 않는 shape {joint.shape} @ {f}")
                continue

            np.save(out_path, bone)
        except Exception as e:
            print(f"[ERROR] {f} 처리 실패: {e}")

    print(f"[DONE] 출력 루트: {args.out_dir}")

if __name__ == "__main__":
    main()
