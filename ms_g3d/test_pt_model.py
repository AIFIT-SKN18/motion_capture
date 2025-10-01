#!/usr/bin/env python3
"""
원본 .pt 모델로 비디오를 분류하는 테스트 스크립트
PTL 모델과 비교하기 위해 동일한 전처리 로직 사용
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

# MS-G3D 모듈 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MS-G3D'))

# 비디오 처리를 위한 import
try:
    import cv2
    import mediapipe as mp
    VIDEO_PROCESSING_AVAILABLE = True
except ImportError:
    VIDEO_PROCESSING_AVAILABLE = False
    print("경고: cv2 또는 mediapipe가 설치되지 않아 비디오 처리 기능을 사용할 수 없습니다.")

def import_class(name):
    """동적으로 클래스 import"""
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def load_pt_model(weights_path):
    """원본 .pt 모델 로드"""
    print(f"PT 모델 로드 중: {weights_path}")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {weights_path}")
    
    # 체크포인트 분석
    weights = torch.load(weights_path, map_location='cpu')
    
    # 가중치 키 정리 (module. 접두사 제거)
    cleaned_weights = OrderedDict()
    for k, v in weights.items():
        key = k.replace('module.', '') if k.startswith('module.') else k
        cleaned_weights[key] = v
    
    # 설정 추출
    data_bn_size = cleaned_weights['data_bn.weight'].shape[0]
    fc_weight = cleaned_weights['fc.weight']
    num_class = fc_weight.shape[0]
    fc_in_features = fc_weight.shape[1]
    
    # 표준 설정
    in_channels = 3
    num_point = 25
    actual_num_person = 1
    
    print(f"감지된 설정:")
    print(f"  - num_class: {num_class}")
    print(f"  - num_point: {num_point}")
    print(f"  - num_person: {actual_num_person}")
    print(f"  - in_channels: {in_channels}")
    
    # 모델 설정
    model_args = {
        'num_class': num_class,
        'num_point': num_point,
        'num_person': actual_num_person,
        'num_gcn_scales': 13,
        'num_g3d_scales': 6,
        'graph': 'graph.ntu_rgb_d.AdjMatrixGraph',
        'in_channels': in_channels
    }
    
    # 모델 클래스 import
    Model = import_class('model.msg3d.Model')
    
    # 모델 초기화
    model = Model(**model_args)
    
    # 가중치 로드
    model.load_state_dict(cleaned_weights)
    model.eval()
    
    print(f"PT 모델 로드 완료 - 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    return model, model_args

class VideoClassifierPT:
    """PT 모델을 사용한 비디오 분류기"""
    
    def __init__(self, model_path, class_names=None):
        """
        Args:
            model_path: .pt 모델 파일 경로
            class_names: 클래스 이름 리스트
        """
        if not VIDEO_PROCESSING_AVAILABLE:
            raise RuntimeError("비디오 처리를 위해 opencv-python과 mediapipe를 설치해주세요.")
        
        # 모델 로드
        print(f"PT 모델 로드 중: {model_path}")
        self.model, self.model_args = load_pt_model(model_path)
        
        # 클래스 이름 설정
        if class_names is None:
            self.class_names = [
                'benchpress',       # 0
                'deadlift',         # 1
                'lunges',           # 2
                'side_lateral_raise', # 3
                'squat',            # 4
                'others'            # 5
            ]
        else:
            self.class_names = class_names
        
        # MediaPipe Pose 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print(f"PT 모델 로드 완료. 클래스: {self.class_names}")
    
    def mediapipe_to_ntu(self, pose_landmarks):
        """
        MediaPipe 포즈를 NTU RGB+D 25개 관절점으로 변환
        PTL 버전과 동일한 매핑 사용
        """
        joints = np.zeros((25, 3))
        landmarks = pose_landmarks.landmark
        
        # 0: spine_base (hip center)
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
                     (landmarks[11].y + landmarks[12].y) / 2 - 0.05,
                     (landmarks[11].z + landmarks[12].z) / 2]
        
        # 3: head
        joints[3] = [landmarks[0].x, landmarks[0].y, landmarks[0].z]
        
        # 4-7: left arm
        joints[4] = [landmarks[11].x, landmarks[11].y, landmarks[11].z]
        joints[5] = [landmarks[13].x, landmarks[13].y, landmarks[13].z]
        joints[6] = [landmarks[15].x, landmarks[15].y, landmarks[15].z]
        joints[7] = [landmarks[15].x, landmarks[15].y, landmarks[15].z]
        
        # 8-11: right arm
        joints[8] = [landmarks[12].x, landmarks[12].y, landmarks[12].z]
        joints[9] = [landmarks[14].x, landmarks[14].y, landmarks[14].z]
        joints[10] = [landmarks[16].x, landmarks[16].y, landmarks[16].z]
        joints[11] = [landmarks[16].x, landmarks[16].y, landmarks[16].z]
        
        # 12-15: left leg
        joints[12] = [landmarks[23].x, landmarks[23].y, landmarks[23].z]
        joints[13] = [landmarks[25].x, landmarks[25].y, landmarks[25].z]
        joints[14] = [landmarks[27].x, landmarks[27].y, landmarks[27].z]
        joints[15] = [landmarks[31].x, landmarks[31].y, landmarks[31].z]
        
        # 16-19: right leg
        joints[16] = [landmarks[24].x, landmarks[24].y, landmarks[24].z]
        joints[17] = [landmarks[26].x, landmarks[26].y, landmarks[26].z]
        joints[18] = [landmarks[28].x, landmarks[28].y, landmarks[28].z]
        joints[19] = [landmarks[32].x, landmarks[32].y, landmarks[32].z]
        
        # 20: spine_shoulder
        joints[20] = shoulder_center
        
        # 21-24: hand tips and thumbs
        joints[21] = [landmarks[17].x, landmarks[17].y, landmarks[17].z]
        joints[22] = [landmarks[21].x, landmarks[21].y, landmarks[21].z]
        joints[23] = [landmarks[18].x, landmarks[18].y, landmarks[18].z]
        joints[24] = [landmarks[22].x, landmarks[22].y, landmarks[22].z]
        
        return joints
    
    def extract_skeleton_from_video(self, video_path, max_frames=64):
        """비디오에서 스켈레톤 시퀀스 추출"""
        # 절대 경로로 변환
        video_path = os.path.abspath(video_path)
        
        if not os.path.exists(video_path):
            # 상대 경로로 다시 시도
            alternative_paths = [
                video_path,
                os.path.join(os.getcwd(), video_path),
                os.path.join('MS-G3D', 'AI_Fit', os.path.basename(video_path))
            ]
            
            found = False
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    video_path = alt_path
                    found = True
                    break
            
            if not found:
                raise FileNotFoundError(
                    f"비디오 파일을 찾을 수 없습니다: {video_path}\n"
                    f"시도한 경로들:\n" + "\n".join(f"  - {p}" for p in alternative_paths)
                )
        
        cap = cv2.VideoCapture(video_path)
        skeleton_sequence = []
        frame_count = 0
        
        print(f"비디오 처리 중: {video_path}")
        
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
            raise ValueError("비디오에서 포즈를 추출하지 못했습니다.")
        
        print(f"추출된 프레임 수: {len(skeleton_sequence)}")
        return np.array(skeleton_sequence)
    
    def preprocess_skeleton(self, skeleton_seq):
        """
        스켈레톤 시퀀스를 모델 입력 형식으로 변환
        (T, V, C) -> (1, C, T, V, M)
        """
        # (T, V, C) -> (C, T, V)
        skeleton_data = skeleton_seq.transpose(2, 0, 1)
        # (C, T, V) -> (C, T, V, M)
        skeleton_data = np.expand_dims(skeleton_data, axis=-1)
        # (C, T, V, M) -> (1, C, T, V, M)
        skeleton_data = np.expand_dims(skeleton_data, axis=0)
        
        return torch.from_numpy(skeleton_data).float()
    
    def classify_video(self, video_path):
        """
        비디오를 분류하고 결과 반환
        
        Returns:
            dict: {
                'predicted_class': 예측된 클래스 이름,
                'class_index': 클래스 인덱스,
                'confidence': 신뢰도,
                'all_probabilities': 모든 클래스별 확률
            }
        """
        # 1. 비디오에서 스켈레톤 추출
        skeleton_seq = self.extract_skeleton_from_video(video_path)
        
        # 2. 전처리
        input_data = self.preprocess_skeleton(skeleton_seq)
        
        print(f"입력 데이터 형태: {input_data.shape}")
        
        # 3. 예측
        with torch.no_grad():
            output = self.model(input_data)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_idx = torch.argmax(output, dim=1).item()
            confidence = probabilities[predicted_idx].item()
        
        # 4. 결과 구성
        result = {
            'predicted_class': self.class_names[predicted_idx],
            'class_index': predicted_idx,
            'confidence': confidence,
            'all_probabilities': {
                self.class_names[i]: probabilities[i].item()
                for i in range(len(self.class_names))
            }
        }
        
        return result
    
    def print_classification_result(self, result):
        """분류 결과를 보기 좋게 출력"""
        print("\n" + "="*60)
        print("분류 결과")
        print("="*60)
        print(f"예측된 클래스: {result['predicted_class']}")
        print(f"신뢰도: {result['confidence']:.2%}")
        print(f"\n모든 클래스별 확률:")
        
        # 확률 순으로 정렬
        sorted_probs = sorted(
            result['all_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for class_name, prob in sorted_probs:
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  {class_name:20s} {bar} {prob:.2%}")
        print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description='PT 모델로 비디오 분류 테스트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 클래스 이름 사용
  python test_pt_model.py --model MS-G3D/AI_Fit/weights-9-837.pt --video test_video.mp4
  
  # 사용자 정의 클래스 이름
  python test_pt_model.py --model weights.pt --video video.mp4 --class-names class1 class2 class3
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='PT 모델 파일 경로'
    )
    
    parser.add_argument(
        '--video', '-v',
        type=str,
        required=True,
        help='분류할 비디오 파일 경로'
    )
    
    parser.add_argument(
        '--class-names', '-c',
        type=str,
        nargs='+',
        default=None,
        help='클래스 이름 리스트 (공백으로 구분)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=64,
        help='처리할 최대 프레임 수 (기본값: 64)'
    )
    
    args = parser.parse_args()
    
    # 의존성 확인
    if not VIDEO_PROCESSING_AVAILABLE:
        print("오류: opencv-python과 mediapipe가 설치되어 있지 않습니다.")
        print("다음 명령어로 설치해주세요:")
        print("  pip install opencv-python mediapipe")
        sys.exit(1)
    
    try:
        # 분류기 초기화
        print(f"모델 로드 중: {args.model}")
        classifier = VideoClassifierPT(args.model, class_names=args.class_names)
        
        # 비디오 분류
        print(f"\n비디오 분류 시작: {args.video}")
        result = classifier.classify_video(args.video)
        
        # 결과 출력
        classifier.print_classification_result(result)
        
        # 상세 정보 출력
        print("상세 정보:")
        print(f"  예측 클래스 인덱스: {result['class_index']}")
        print(f"  신뢰도: {result['confidence']:.4f}")
        
    except FileNotFoundError as e:
        print(f"오류: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
