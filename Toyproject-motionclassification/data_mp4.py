#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MP4 to Skeleton Data Converter
- MediaPipe Pose로 포즈 추출
- NTU RGB+D 25관절로 매핑
- 전형적인 NTU 파일 저장:
    * skeleton 텍스트(.skeleton): NTU RGB+D 포맷
    * (옵션) joint NPY: (C,T,V,M)=(3,T,25,1) 형태 *_data_joint.npy
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
from pathlib import Path

# 조용 모드(선택): TensorFlow/XLA 경고 억제
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

def _fmt_float(x):
    # NTU .skeleton은 공백 구분 텍스트. 소수 6자리로 정규화.
    try:
        return f"{float(x):.6f}"
    except Exception:
        return "0.000000"


class PoseExtractor:
    """MediaPipe를 사용한 포즈 추출기"""

    def __init__(self,
                model_complexity: int = 2,
                min_det_conf: float = 0.5,
                min_track_conf: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf
        )

        # NTU RGB+D 25개 관절점 순서 (0-based index)
        self.ntu_joint_names = [
            'spine_base',      # 0  - hip center
            'spine_mid',       # 1  - spine
            'neck',            # 2  - neck
            'head',            # 3  - head
            'left_shoulder',   # 4
            'left_elbow',      # 5
            'left_wrist',      # 6
            'left_hand',       # 7
            'right_shoulder',  # 8
            'right_elbow',     # 9
            'right_wrist',     # 10
            'right_hand',      # 11
            'left_hip',        # 12
            'left_knee',       # 13
            'left_ankle',      # 14
            'left_foot',       # 15
            'right_hip',       # 16
            'right_knee',      # 17
            'right_ankle',     # 18
            'right_foot',      # 19
            'spine_shoulder',  # 20 - shoulder center
            'left_hand_tip',   # 21
            'left_thumb',      # 22
            'right_hand_tip',  # 23
            'right_thumb'      # 24
        ]

    def __del__(self):
        try:
            self.pose.close()
        except Exception:
            pass

    def mediapipe_to_ntu(self, pose_landmarks):
        """MediaPipe 포즈를 NTU RGB+D 25개 관절점으로 변환"""
        joints = np.zeros((25, 3), dtype=np.float32)
        landmarks = pose_landmarks.landmark

        # 0: spine_base (hip center) - 양쪽 힙 중점
        joints[0] = [
            (landmarks[23].x + landmarks[24].x) / 2,
            (landmarks[23].y + landmarks[24].y) / 2,
            (landmarks[23].z + landmarks[24].z) / 2
        ]

        hip_center = joints[0]
        shoulder_center = [
            (landmarks[11].x + landmarks[12].x) / 2,
            (landmarks[11].y + landmarks[12].y) / 2,
            (landmarks[11].z + landmarks[12].z) / 2
        ]

        # 1: spine_mid
        joints[1] = [
            (hip_center[0] + shoulder_center[0]) / 2,
            (hip_center[1] + shoulder_center[1]) / 2,
            (hip_center[2] + shoulder_center[2]) / 2
        ]

        # 2: neck (어깨 중점에서 약간 위)
        joints[2] = [
            (landmarks[11].x + landmarks[12].x) / 2,
            (landmarks[11].y + landmarks[12].y) / 2 - 0.05,
            (landmarks[11].z + landmarks[12].z) / 2
        ]

        # 3: head - nose
        joints[3] = [landmarks[0].x, landmarks[0].y, landmarks[0].z]

        # 4-11: 팔
        joints[4]  = [landmarks[11].x, landmarks[11].y, landmarks[11].z]  # left_shoulder
        joints[5]  = [landmarks[13].x, landmarks[13].y, landmarks[13].z]  # left_elbow
        joints[6]  = [landmarks[15].x, landmarks[15].y, landmarks[15].z]  # left_wrist
        joints[7]  = [landmarks[15].x, landmarks[15].y, landmarks[15].z]  # left_hand(동일)
        joints[8]  = [landmarks[12].x, landmarks[12].y, landmarks[12].z]  # right_shoulder
        joints[9]  = [landmarks[14].x, landmarks[14].y, landmarks[14].z]  # right_elbow
        joints[10] = [landmarks[16].x, landmarks[16].y, landmarks[16].z]  # right_wrist
        joints[11] = [landmarks[16].x, landmarks[16].y, landmarks[16].z]  # right_hand(동일)

        # 12-19: 다리
        joints[12] = [landmarks[23].x, landmarks[23].y, landmarks[23].z]  # left_hip
        joints[13] = [landmarks[25].x, landmarks[25].y, landmarks[25].z]  # left_knee
        joints[14] = [landmarks[27].x, landmarks[27].y, landmarks[27].z]  # left_ankle
        joints[15] = [landmarks[31].x, landmarks[31].y, landmarks[31].z]  # left_foot_index
        joints[16] = [landmarks[24].x, landmarks[24].y, landmarks[24].z]  # right_hip
        joints[17] = [landmarks[26].x, landmarks[26].y, landmarks[26].z]  # right_knee
        joints[18] = [landmarks[28].x, landmarks[28].y, landmarks[28].z]  # right_ankle
        joints[19] = [landmarks[32].x, landmarks[32].y, landmarks[32].z]  # right_foot_index

        # 20-24: 기타
        joints[20] = shoulder_center                                   # spine_shoulder
        joints[21] = [landmarks[17].x, landmarks[17].y, landmarks[17].z]  # left_hand_tip
        joints[22] = [landmarks[21].x, landmarks[21].y, landmarks[21].z]  # left_thumb
        joints[23] = [landmarks[18].x, landmarks[18].y, landmarks[18].z]  # right_hand_tip
        joints[24] = [landmarks[22].x, landmarks[22].y, landmarks[22].z]  # right_thumb

        return joints.astype(np.float32)

    @staticmethod
    def _ensure_T(arr_T25x3: np.ndarray, flags: list, T: int = 300, pad_mode: str = "repeat_last"):
        """(T0,25,3) -> (T,25,3) 로 패딩/자르기 + flags(탐지여부)도 동일 길이로 보정"""
        t0 = arr_T25x3.shape[0]
        base = arr_T25x3 if t0 > 0 else np.zeros((1, 25, 3), dtype=np.float32)
        if t0 == 0:
            flags = [False]

        if base.shape[0] >= T:
            return base[:T].astype(np.float32, copy=False), flags[:T]
        # pad
        if pad_mode == "zeros":
            pad = np.zeros((T - base.shape[0], 25, 3), dtype=np.float32)
            flags_pad = [False] * (T - base.shape[0])
        else:  # repeat_last
            pad = np.repeat(base[-1][None, ...], T - base.shape[0], axis=0).astype(np.float32)
            flags_pad = [flags[-1] if len(flags) > 0 else False] * (T - base.shape[0])
        return np.concatenate([base, pad], axis=0), (flags + flags_pad)

    @staticmethod
    def to_ctvm(arr_T25x3: np.ndarray) -> np.ndarray:
        """(T,25,3) -> (3,T,25,1)"""
        ctvm = np.transpose(arr_T25x3, (2, 0, 1))[:, :, :, None]
        return ctvm.astype(np.float32, copy=False)

    @staticmethod
    def save_ntu_npy(ctvm: np.ndarray, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), ctvm)

    @staticmethod
    def save_ntu_skeleton(arr_T25x3: np.ndarray, has_pose_flags: list, out_path: Path, img_w: int, img_h: int):
        """NTU .skeleton 텍스트로 저장. 프레임당 body=1로 고정(누락 프레임도 joint=0)."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        T = arr_T25x3.shape[0]
        with open(out_path, "w", encoding="utf-8") as f:
            # 1) number of frames
            f.write(f"{T}\n")
            for t in range(T):
                # 2) number of bodies in this frame (고정 1)
                f.write("1\n")
                # 3) body info (10개 필드)
                # bodyID clippedEdges handLeftConfidence handRightConfidence handLeftState handRightState isRestricted leanX leanY trackingState
                tracking_state = 2 if (has_pose_flags[t] if t < len(has_pose_flags) else True) else 1
                f.write(f"1 0 0 0 0 0 0 0.000000 0.000000 {tracking_state}\n")
                # 4) number of joints
                f.write("25\n")
                # 5) 25 joints lines
                for j in range(25):
                    x, y, z = arr_T25x3[t, j].tolist()
                    # 픽셀 좌표: color/depth 모두 이미지 기준 픽셀로 기록
                    colorX = x * img_w
                    colorY = y * img_h
                    depthX = colorX
                    depthY = colorY
                    # 쿼터니언: 항등 회전
                    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
                    jt_state = tracking_state
                    f.write(
                        f"{_fmt_float(x)} {_fmt_float(y)} {_fmt_float(z)} "
                        f"{_fmt_float(depthX)} {_fmt_float(depthY)} "
                        f"{_fmt_float(colorX)} {_fmt_float(colorY)} "
                        f"{_fmt_float(qw)} {_fmt_float(qx)} {_fmt_float(qy)} {_fmt_float(qz)} "
                        f"{jt_state}\n"
                    )

    def extract_from_video(self, video_path: str, max_frames: int = 300):
        """비디오에서 NTU 형식의 스켈레톤 시퀀스 추출 (T,25,3), flags, (w,h)"""
        cap = cv2.VideoCapture(video_path)
        skeleton_sequence = []
        has_pose = []
        frame_count = 0
        img_w, img_h = 0, 0

        # 전체 프레임수(진행바용)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        total_frames = min(total_frames, max_frames) if total_frames > 0 else max_frames

        with tqdm(total=total_frames, desc=f"Extracting poses: {Path(video_path).name}") as pbar:
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if img_w == 0 or img_h == 0:
                    img_h, img_w = frame.shape[:2]

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)

                if results.pose_landmarks:
                    joints = self.mediapipe_to_ntu(results.pose_landmarks)
                    skeleton_sequence.append(joints)
                    has_pose.append(True)
                else:
                    # 미검출: 이전 프레임 복사, 첫 프레임이면 zero
                    if skeleton_sequence:
                        skeleton_sequence.append(skeleton_sequence[-1].copy())
                    else:
                        skeleton_sequence.append(np.zeros((25, 3), dtype=np.float32))
                    has_pose.append(False)

                frame_count += 1
                pbar.update(1)

        cap.release()

        if not skeleton_sequence:
            # 전혀 검출 안 되면 최소 한 프레임 zero
            return np.zeros((1, 25, 3), dtype=np.float32), [False], (img_w or 1920), (img_h or 1080)

        skeleton_array = np.asarray(skeleton_sequence, dtype=np.float32)
        return skeleton_array, has_pose, (img_w or 1920), (img_h or 1080)


def parse_args():
    p = argparse.ArgumentParser(description="MP4->NTU RGB+D skeleton(.skeleton) / joint NPY converter")
    p.add_argument("--videos", type=str, nargs="+", required=True,
                help="입력 비디오 경로(들). 단일파일/디렉터리/글롭('/path/*.mp4', '/path/**/*.mp4') 지원")
    p.add_argument("--out_dir", type=str, required=True, help="저장 폴더")
    p.add_argument("--format", type=str, default="skeleton", choices=["skeleton", "npy", "both"],
                help="저장 포맷 선택")
    p.add_argument("--exts", type=str, default="mp4,mov,avi,mkv",
                help="허용 비디오 확장자(쉼표 구분). 예: mp4,mov,avi,mkv")
    p.add_argument("--T", type=int, default=300, help="타겟 프레임 길이(T)")
    p.add_argument("--pad_mode", type=str, default="repeat_last", choices=["repeat_last", "zeros"],
                help="부족한 프레임 패딩 방식")
    p.add_argument("--model_complexity", type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--min_det_conf", type=float, default=0.5)
    p.add_argument("--min_track_conf", type=float, default=0.5)
    return p.parse_args()


def expand_inputs(videos, exts):
    """글롭/디렉터리/단일파일을 모두 확장하여 허용 확장자만 반환"""
    allowed = tuple("." + e.lower().lstrip(".") for e in exts)
    out = []
    for v in videos:
        v = os.path.expanduser(str(v))
        if any(ch in v for ch in "*?[]"):  # glob patterns
            out.extend(sorted(glob.glob(v, recursive=True)))
        elif os.path.isdir(v):             # directory -> scan for each ext
            for e in allowed:
                out.extend(sorted(glob.glob(os.path.join(v, f"*{e}"))))
        else:                              # single file
            out.append(v)
    # keep only files that exist and match allowed extensions
    out = [p for p in out if os.path.isfile(p) and p.lower().endswith(allowed)]
    return sorted(set(out))


def main():
    args = parse_args()
    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]

    expanded = expand_inputs(args.videos, exts)
    if not expanded:
        raise FileNotFoundError(
            "입력 비디오가 없습니다.\n"
            f" - 인자: {args.videos}\n"
            f" - 허용 확장자: {exts}\n"
            " - 팁: *.mp4 같은 비디오 패턴을 사용하거나 --exts 로 확장자를 지정하세요.\n"
            "   예) --videos '/content/.../*.mp4'  또는  --exts mp4,mov --videos '/content/.../*/*.*'"
        )

    extractor = PoseExtractor(model_complexity=args.model_complexity,
                            min_det_conf=args.min_det_conf,
                            min_track_conf=args.min_track_conf)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for vp in tqdm(expanded, desc="Processing videos"):
        try:
            arr, flags, img_w, img_h = extractor.extract_from_video(vp, max_frames=args.T)
            arr, flags = extractor._ensure_T(arr, flags, T=args.T, pad_mode=args.pad_mode)

            stem = Path(vp).stem
            if args.format in ("skeleton", "both"):
                skel_path = Path(args.out_dir) / f"{stem}.skeleton"
                extractor.save_ntu_skeleton(arr, flags, skel_path, img_w, img_h)
            if args.format in ("npy", "both"):
                ctvm = extractor.to_ctvm(arr)
                npy_path = Path(args.out_dir) / f"{stem}_data_joint.npy"
                extractor.save_ntu_npy(ctvm, npy_path)

        except Exception as e:
            print(f"❌ Error processing {vp}: {e}")

    print(f"✅ Done. Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
