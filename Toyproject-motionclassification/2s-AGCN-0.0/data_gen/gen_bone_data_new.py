# === single_joint_to_bone_ntu.py ===
import numpy as np
import os, argparse

# NTU(25) 관절 페어 (1-based) → bone[v1] = joint[v1] - joint[v2]
PARIS_NTU = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7),
    (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21),
    (23, 8), (24, 25), (25, 12)
)

def joints_to_bones_ntu(joint: np.ndarray) -> np.ndarray:
    """
    joint: (N, C=3, T, V=25, M)  ->  bone: same shape
    """
    bone = np.zeros_like(joint, dtype=joint.dtype)
    for v1, v2 in PARIS_NTU:
        v1 -= 1; v2 -= 1
        bone[:, :, :, v1, :] = joint[:, :, :, v1, :] - joint[:, :, :, v2, :]
    return bone

def main():
    ap = argparse.ArgumentParser("Single NTU joint.npy -> bone.npy")
    ap.add_argument("--in_joint",  required=True, help="입력 joint.npy (N,3,300,25,M)")
    ap.add_argument("--out_bone",  required=True, help="출력 bone.npy 경로")
    args = ap.parse_args()

    X = np.load(args.in_joint, mmap_mode="r")
    print(f"[info] load {args.in_joint}, shape={X.shape}")

    bone = joints_to_bones_ntu(np.asarray(X))
    os.makedirs(os.path.dirname(args.out_bone), exist_ok=True)
    np.save(args.out_bone, bone)
    print(f"[done] save {args.out_bone}, shape={bone.shape}")

if __name__ == "__main__":
    main()
