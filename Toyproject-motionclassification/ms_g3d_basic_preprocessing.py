#!/usr/bin/env python
"""
MS-G3D 기반 기본 전처리 시스템
- MediaPipe를 사용한 포즈 추출
- NTU RGB+D 25개 관절점으로 변환
- 기본 Train/Val 데이터 분할
"""

import os
import sys
import numpy as np
import pickle
import torch
import cv2
import mediapipe as mp
import random
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    """모든 랜덤 시드를 고정하여 재현 가능한 결과를 보장"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"시드 {seed}로 고정 완료")

class PoseExtractor:
    """MediaPipe를 사용한 포즈 추출기"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
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

    def mediapipe_to_ntu(self, pose_landmarks):
        """MediaPipe 포즈를 NTU RGB+D 25개 관절점으로 변환"""
        joints = np.zeros((25, 3))
        landmarks = pose_landmarks.landmark

        # 0: spine_base (hip center) - 양쪽 힙의 중점
        joints[0] = [(landmarks[23].x + landmarks[24].x) / 2,
                    (landmarks[23].y + landmarks[24].y) / 2,
                    (landmarks[23].z + landmarks[24].z) / 2]

        hip_center = joints[0]
        shoulder_center = [(landmarks[11].x + landmarks[12].x) / 2,
                        (landmarks[11].y + landmarks[12].y) / 2,
                        (landmarks[11].z + landmarks[12].z) / 2]
        
        # 1: spine_mid
        joints[1] = [(hip_center[0] + shoulder_center[0]) / 2,
                    (hip_center[1] + shoulder_center[1]) / 2,
                    (hip_center[2] + shoulder_center[2]) / 2]

        # 2: neck
        joints[2] = [(landmarks[11].x + landmarks[12].x) / 2,
                    (landmarks[11].y + landmarks[12].y) / 2 - 0.05,  # 약간 위로
                    (landmarks[11].z + landmarks[12].z) / 2]

        # 3: head - nose
        joints[3] = [landmarks[0].x, landmarks[0].y, landmarks[0].z]

        # 4-11: 팔 관절
        joints[4] = [landmarks[11].x, landmarks[11].y, landmarks[11].z]  # left_shoulder
        joints[5] = [landmarks[13].x, landmarks[13].y, landmarks[13].z]  # left_elbow
        joints[6] = [landmarks[15].x, landmarks[15].y, landmarks[15].z]  # left_wrist
        joints[7] = [landmarks[15].x, landmarks[15].y, landmarks[15].z]  # left_hand
        joints[8] = [landmarks[12].x, landmarks[12].y, landmarks[12].z]  # right_shoulder
        joints[9] = [landmarks[14].x, landmarks[14].y, landmarks[14].z]  # right_elbow
        joints[10] = [landmarks[16].x, landmarks[16].y, landmarks[16].z]  # right_wrist
        joints[11] = [landmarks[16].x, landmarks[16].y, landmarks[16].z]  # right_hand

        # 12-19: 다리 관절
        joints[12] = [landmarks[23].x, landmarks[23].y, landmarks[23].z]  # left_hip
        joints[13] = [landmarks[25].x, landmarks[25].y, landmarks[25].z]  # left_knee
        joints[14] = [landmarks[27].x, landmarks[27].y, landmarks[27].z]  # left_ankle
        joints[15] = [landmarks[31].x, landmarks[31].y, landmarks[31].z]  # left_foot
        joints[16] = [landmarks[24].x, landmarks[24].y, landmarks[24].z]  # right_hip
        joints[17] = [landmarks[26].x, landmarks[26].y, landmarks[26].z]  # right_knee
        joints[18] = [landmarks[28].x, landmarks[28].y, landmarks[28].z]  # right_ankle
        joints[19] = [landmarks[32].x, landmarks[32].y, landmarks[32].z]  # right_foot

        # 20-24: 기타 관절
        joints[20] = shoulder_center  # spine_shoulder
        joints[21] = [landmarks[17].x, landmarks[17].y, landmarks[17].z]  # left_hand_tip
        joints[22] = [landmarks[21].x, landmarks[21].y, landmarks[21].z]  # left_thumb
        joints[23] = [landmarks[18].x, landmarks[18].y, landmarks[18].z]  # right_hand_tip
        joints[24] = [landmarks[22].x, landmarks[22].y, landmarks[22].z]  # right_thumb

        return joints

    def extract_from_video(self, video_path, max_frames=300):
        """비디오에서 NTU 형식의 스켈레톤 시퀀스 추출"""
        cap = cv2.VideoCapture(video_path)
        skeleton_sequence = []
        frame_count = 0

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
                # 포즈를 찾지 못한 경우 이전 프레임 복사
                if skeleton_sequence:
                    skeleton_sequence.append(skeleton_sequence[-1].copy())
                else:
                    skeleton_sequence.append(np.zeros((25, 3)))

            frame_count += 1

        cap.release()

        if not skeleton_sequence:
            return np.zeros((1, 25, 3))

        return np.array(skeleton_sequence)

def generate_original_exercise_data_with_others(data_dir, output_dir, pose_extractor):
    """원본 비디오에서 스켈레톤 데이터 추출 (Others 자동 분류 포함)"""
    os.makedirs(output_dir, exist_ok=True)

    # 정의된 클래스들
    known_classes = [
        'benchpress',       # 0
        'deadlift',         # 1
        'lunges',           # 2
        'side_lateral_raise', # 3
        'squat',            # 4
    ]

    # 전체 클래스 (others 포함)
    class_names = known_classes + ['others']  # 5
    class_to_label = {name: idx for idx, name in enumerate(class_names)}

    print(f"정의된 클래스: {known_classes}")
    print(f"전체 클래스: {class_names}")

    # MS-G3D 형식의 데이터 구조
    sample_names = []
    sample_labels = []
    all_skeleton_data = {}
    sample_id = 0

    # 데이터 디렉토리의 모든 폴더 스캔
    all_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    print(f"\n발견된 폴더들: {all_folders}")

    # 알려진 클래스와 알려지지 않은 클래스 분리
    known_folders = [f for f in all_folders if f in known_classes]
    unknown_folders = [f for f in all_folders if f not in known_classes]

    print(f"알려진 클래스 폴더: {known_folders}")
    print(f"알려지지 않은 폴더 (Others로 분류): {unknown_folders}")

    # 1. 알려진 클래스들 처리
    for class_name in known_folders:
        class_dir = os.path.join(data_dir, class_name)
        video_files = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]
        print(f"\nProcessing {len(video_files)} videos from {class_name}")

        for video_file in tqdm(video_files, desc=f"Processing {class_name}"):
            video_path = os.path.join(class_dir, video_file)

            try:
                # 스켈레톤 데이터 추출
                skeleton_seq = pose_extractor.extract_from_video(video_path)

                if skeleton_seq.shape[0] > 5:  # 최소 5프레임 이상
                    # MS-G3D 형식으로 변환: (C, T, V, M)
                    skeleton_data = skeleton_seq.transpose(2, 0, 1)  # (3, T, 25)
                    skeleton_data = np.expand_dims(skeleton_data, axis=-1)  # (3, T, 25, 1)

                    # 샘플 이름 생성
                    sample_name = f"E{sample_id:06d}"  # E for Exercise

                    # 데이터 저장
                    all_skeleton_data[sample_name] = {
                        'skel_body0': skeleton_data,
                        'label': class_to_label[class_name],
                        'has_skeleton': True,
                        'video_file': video_file,
                        'class_name': class_name,
                        'source_folder': class_name
                    }

                    sample_names.append(sample_name)
                    sample_labels.append(class_to_label[class_name])
                    sample_id += 1

            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue

    # 2. 알려지지 않은 폴더들을 Others로 처리
    if unknown_folders:
        print(f"\n🔍 알려지지 않은 폴더들을 Others 클래스로 분류 중...")

        for unknown_folder in unknown_folders:
            unknown_dir = os.path.join(data_dir, unknown_folder)
            video_files = [f for f in os.listdir(unknown_dir) if f.endswith('.mp4')]
            print(f"  📁 {unknown_folder}: {len(video_files)}개 비디오 → Others 클래스")

            for video_file in tqdm(video_files, desc=f"Processing {unknown_folder} as Others"):
                video_path = os.path.join(unknown_dir, video_file)

                try:
                    # 스켈레톤 데이터 추출
                    skeleton_seq = pose_extractor.extract_from_video(video_path)

                    if skeleton_seq.shape[0] > 5:  # 최소 5프레임 이상
                        # MS-G3D 형식으로 변환
                        skeleton_data = skeleton_seq.transpose(2, 0, 1)
                        skeleton_data = np.expand_dims(skeleton_data, axis=-1)

                        # 샘플 이름 생성
                        sample_name = f"O{sample_id:06d}"  # O for Others

                        # Others 클래스로 라벨링
                        all_skeleton_data[sample_name] = {
                            'skel_body0': skeleton_data,
                            'label': class_to_label['others'],
                            'has_skeleton': True,
                            'video_file': video_file,
                            'class_name': 'others',
                            'source_folder': unknown_folder,
                            'original_type': unknown_folder
                        }

                        sample_names.append(sample_name)
                        sample_labels.append(class_to_label['others'])
                        sample_id += 1

                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                    continue

    # 원본 데이터 저장
    original_file = os.path.join(output_dir, "exercise_data_joint.npy")
    np.save(original_file, all_skeleton_data)

    original_label_file = os.path.join(output_dir, "exercise_train_label.pkl")
    with open(original_label_file, 'wb') as f:
        pickle.dump((sample_names, sample_labels), f)

    # 클래스 정보 저장
    class_info = {
        'class_names': class_names,
        'class_to_label': class_to_label,
        'num_classes': len(class_names),
        'known_classes': known_classes,
        'unknown_folders_processed': unknown_folders
    }

    with open(os.path.join(output_dir, 'exercise_class_info.pkl'), 'wb') as f:
        pickle.dump(class_info, f)

    print(f"\n=== 원본 데이터 생성 완료 ===")
    print(f"총 {len(sample_names)}개 원본 샘플 생성")
    print(f"\n클래스별 분포:")
    for i, class_name in enumerate(class_names):
        count = sum(1 for label in sample_labels if label == i)
        if count > 0:
            print(f"  {class_name}: {count}개")

    print(f"\n저장된 파일:")
    print(f"- 원본 스켈레톤: {original_file}")
    print(f"- 원본 라벨: {original_label_file}")

    return output_dir

#def split_train_val_data(data_path, label_path, output_dir, val_split_ratio=0.2, stratify_split=True, random_seed=42):
    """데이터를 train/val로 분할
    print("\n" + "="*50)
    print("📊 데이터 분할 중...")
    print("="*50)

    # 시드 설정
    set_seed(random_seed)

    # 원본 데이터 로드
    original_data = np.load(data_path, allow_pickle=True).item()
    with open(label_path, 'rb') as f:
        sample_names, labels = pickle.load(f)

    print(f"원본 데이터: {len(sample_names)}개")

    # 분할 수행
    stratify = labels if stratify_split else None

    train_names, val_names, train_labels, val_labels = train_test_split(
        sample_names,
        labels,
        test_size=val_split_ratio,
        random_state=random_seed,
        stratify=stratify
    )

    # 분할된 데이터 저장
    datasets = {
        'train': (train_names, train_labels),
        'val': (val_names, val_labels)
    }

    for split, (names, labels_list) in datasets.items():
        # 해당 분할의 데이터만 추출
        split_data = {name: original_data[name] for name in names}

        # 파일 저장
        data_file = os.path.join(output_dir, f"exercise_{split}_data_joint.npy")
        label_file = os.path.join(output_dir, f"exercise_{split}_label.pkl")

        np.save(data_file, split_data)
        with open(label_file, 'wb') as f:
            pickle.dump((names, labels_list), f)

        print(f"✅ {split} 데이터: {len(names)}개 샘플 저장")
        print(f"   - 데이터: {data_file}")
        print(f"   - 라벨: {label_file}")

    # 분할 통계
    print(f"\n📈 분할 결과:")
    print(f"Train: {len(train_names)}개 ({len(train_names)/(len(train_names)+len(val_names)):.1%})")
    print(f"Val: {len(val_names)}개 ({len(val_names)/(len(train_names)+len(val_names)):.1%})")

    # 클래스별 분할 통계
    if stratify_split:
        print(f"\n📊 클래스별 분할 통계:")
        class_names = ['benchpress', 'deadlift', 'lunges', 'side_lateral_raise', 'squat', 'others']

        for i, class_name in enumerate(class_names):
            train_count = train_labels.count(i)
            val_count = val_labels.count(i)
            total_count = train_count + val_count

            if total_count > 0:
                train_ratio = train_count / total_count * 100
                val_ratio = val_count / total_count * 100
                print(f"  {class_name}: Train {train_count}개 ({train_ratio:.1f}%), Val {val_count}개 ({val_ratio:.1f}%)")

    return output_dir"""

def main():
    """메인 실행 함수"""
    # 시드 고정
    set_seed(42)
    
    # 입출력 디렉토리 설정
    data_dir = "your_video_dataset_path"  # 비디오 데이터셋 경로로 변경
    output_dir = "./processed_data"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("기본 전처리 시작...")
    
    # 1. 포즈 추출기 초기화
    pose_extractor = PoseExtractor()
    print("포즈 추출기 초기화 완료")
    
    # 2. 원본 데이터 생성
    print("\n1단계: 원본 비디오에서 스켈레톤 데이터 추출 중...")
    data_output_dir = generate_original_exercise_data_with_others(
        data_dir, output_dir, pose_extractor
    )
    print("1단계 완료!")
    
    # 3. Train/Val 분할
    """print("\n2단계: 데이터 분할 시작...")
    data_path = os.path.join(output_dir, "exercise_data_joint.npy")
    label_path = os.path.join(output_dir, "exercise_train_label.pkl")
    
    result_dir = split_train_val_data(
        data_path, 
        label_path, 
        output_dir,
        val_split_ratio=0.2,  # 검증 데이터 20%
        stratify_split=True,  # 클래스 비율 유지
        random_seed=42
    )
    print("2단계 완료!")"""
    """
    print(f"\n✅ 기본 전처리 완료!")
    print(f"결과 저장 위치: {result_dir}")
    print(f"- Train 데이터: {result_dir}/exercise_train_data_joint.npy")
    print(f"- Train 라벨: {result_dir}/exercise_train_label.pkl")
    print(f"- Val 데이터: {result_dir}/exercise_val_data_joint.npy")
    print(f"- Val 라벨: {result_dir}/exercise_val_label.pkl")
    print(f"- 클래스 정보: {result_dir}/exercise_class_info.pkl")"""

if __name__ == "__main__":
    main()