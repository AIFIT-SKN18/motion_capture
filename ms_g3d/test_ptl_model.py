#!/usr/bin/env python3
"""
변환된 .ptl 모델로 비디오를 분류하는 테스트 스크립트

사용법:
  python test_ptl_model.py --model weights-9-837.ptl --video your_video.mp4
  python test_ptl_model.py --model weights-9-837.ptl --video your_video.mp4 --class-names benchpress deadlift squat
"""

import argparse
import sys
from convert_pt_to_ptl import VideoClassifier, VIDEO_PROCESSING_AVAILABLE

def main():
    parser = argparse.ArgumentParser(
        description='PTL 모델로 비디오 분류 테스트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 클래스 이름 사용
  python test_ptl_model.py --model MS-G3D/AI_Fit/weights-9-837.ptl --video test_video.mp4
  
  # 사용자 정의 클래스 이름
  python test_ptl_model.py --model weights.ptl --video video.mp4 --class-names class1 class2 class3
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='PTL 모델 파일 경로'
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
        classifier = VideoClassifier(args.model, class_names=args.class_names)
        
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

