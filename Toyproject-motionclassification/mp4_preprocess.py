#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MP4 to Skeleton Data Converter
- MediaPipe Pose로 포즈 추출
- NTU RGB+D 25관절로 매핑
- 전형적인 NTU 파일( C,T,V,M = 3, T, 25, 1 )로 저장: *_data_joint.npy
"""

import os
import argparse
import numpy as np
import cv2
import mediapipe as mp
from tqdm import tqdm
from pathlib import Path


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
        joints[7]  = [landmarks[15].x, landmarks[15].y, landmarks[15].z]  # left_hand (동일 좌표)
        joints[8]  = [landmarks[12].x, landmarks[12].y, landmarks[12].z]  # right_shoulder
        joints[9]  = [landmarks[14].x, landmarks[14].y, landmarks[14].z]  # right_elbow
        joints[10] = [landmarks[16].x, landmarks[16].y, landmarks[16].z]  # right_wrist
        joints[11] = [landmarks[16].x, landmarks[16].y, landmarks[16].z]  # right_hand (동일 좌표)

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
    def _ensure_T(arr_T25x3: np.ndarray, T: int = 300, pad_mode: str = "repeat_last") -> np.ndarray:
        """(T0,25,3) -> (T,25,3) 로 패딩/자르기"""
        t0 = arr_T25x3.shape[0]
        if t0 == 0:
            base = np.zeros((1, 25, 3), dtype=np.float32)
            t0 = 1
        else:
            base = arr_T25x3

        if t0 >= T:
            return base[:T].astype(np.float32, copy=False)

        # pad
        if pad_mode == "zeros":
            pad = np.zeros((T - t0, 25, 3), dtype=np.float32)
        else:  # repeat_last
            pad = np.repeat(base[-1][None, ...], T - t0, axis=0).astype(np.float32)
        return np.concatenate([base, pad], axis=0)

    @staticmethod
    def to_ctvm(arr_T25x3: np.ndarray) -> np.ndarray:
        """(T,25,3) -> (3,T,25,1)"""
        ctvm = np.transpose(arr_T25x3, (2, 0, 1))[:, :, :, None]
        return ctvm.astype(np.float32, copy=False)

    @staticmethod
    def save_ntu_npy(ctvm: np.ndarray, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), ctvm)

    def extract_from_video(self, video_path: str, max_frames: int = 300) -> np.ndarray:
        """비디오에서 NTU 형식의 스켈레톤 시퀀스 추출 (T,25,3)"""
        cap = cv2.VideoCapture(video_path)
        skeleton_sequence = []
        frame_count = 0

        # 전체 프레임수(진행바용)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        total_frames = min(total_frames, max_frames) if total_frames > 0 else max_frames

        with tqdm(total=total_frames, desc=f"Extracting poses: {Path(video_path).name}") as pbar:
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)

                if results.pose_landmarks:
                    joints = self.mediapipe_to_ntu(results.pose_landmarks)
                    skeleton_sequence.append(joints)
                else:
                    # 미검출: 이전 프레임 복사, 첫 프레임이면 zero
                    if skeleton_sequence:
                        skeleton_sequence.append(skeleton_sequence[-1].copy())
                    else:
                        skeleton_sequence.append(np.zeros((25, 3), dtype=np.float32))

                frame_count += 1
                pbar.update(1)

        cap.release()

        if not skeleton_sequence:
            # 전혀 검출 안 되면 최소 한 프레임 zero
            return np.zeros((1, 25, 3), dtype=np.float32)

        skeleton_array = np.asarray(skeleton_sequence, dtype=np.float32)
        return skeleton_array

    def extract_and_save_single(self,
                                video_path: str,
                                out_dir: str,
                                target_T: int = 300,
                                pad_mode: str = "repeat_last",
                                suffix: str = "_data_joint.npy") -> Path:
        """단일 비디오 -> (3,T,25,1) 저장하고 경로 반환"""
        arr = self.extract_from_video(video_path, max_frames=target_T)
        arr = self._ensure_T(arr, T=target_T, pad_mode=pad_mode)
        ctvm = self.to_ctvm(arr)
        out_path = Path(out_dir) / (Path(video_path).stem + suffix)
        self.save_ntu_npy(ctvm, out_path)
        return out_path

    def extract_from_video_batch(self,
                                video_paths,
                                target_T: int = 300,
                                pad_mode: str = "repeat_last",
                                save: bool = False,
                                out_dir: str | None = None,
                                suffix: str = "_data_joint.npy"):
        """여러 비디오에서 스켈레톤 데이터 일괄 추출 및(옵션) 저장"""
        results = {}
        for vp in tqdm(video_paths, desc="Processing videos"):
            try:
                arr = self.extract_from_video(vp, max_frames=target_T)
                arr = self._ensure_T(arr, T=target_T, pad_mode=pad_mode)
                ctvm = self.to_ctvm(arr)
                results[vp] = ctvm  # (3,T,25,1)

                if save:
                    assert out_dir is not None, "save=True이면 out_dir을 지정하세요."
                    out_path = Path(out_dir) / (Path(vp).stem + suffix)
                    self.save_ntu_npy(ctvm, out_path)
            except Exception as e:
                print(f"❌ Error processing {vp}: {e}")
                results[vp] = None
        return results


def parse_args():
    p = argparse.ArgumentParser(description="MP4->NTU RGB+D (3,T,25,1) converter")
    p.add_argument("--videos", type=str, nargs="+", required=True,
                help="입력 비디오 경로(들). 글롭 패턴 가능: '/path/*.mp4'")
    p.add_argument("--out_dir", type=str, required=True,
                help="저장 폴더")
    p.add_argument("--T", type=int, default=300, help="타겟 프레임 길이(T)")
    p.add_argument("--pad_mode", type=str, default="repeat_last", choices=["repeat_last", "zeros"],
                help="부족한 프레임 패딩 방식")
    p.add_argument("--model_complexity", type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--min_det_conf", type=float, default=0.5)
    p.add_argument("--min_track_conf", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    # 글롭 확장
    expanded = []
    for v in args.videos:
        if any(ch in v for ch in "*?[]"):
            expanded.extend([str(p) for p in Path().glob(v)])
        else:
            expanded.append(v)
    if not expanded:
        raise FileNotFoundError("입력 비디오가 없습니다.")

    extractor = PoseExtractor(model_complexity=args.model_complexity,
                            min_det_conf=args.min_det_conf,
                            min_track_conf=args.min_track_conf)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    extractor.extract_from_video_batch(
        expanded,
        target_T=args.T,
        pad_mode=args.pad_mode,
        save=True,
        out_dir=args.out_dir,
        suffix="_data_joint.npy",
    )
    print(f"✅ Done. Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
