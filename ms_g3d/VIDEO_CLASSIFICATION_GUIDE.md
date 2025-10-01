# PTL 모델로 비디오 분류하기

변환된 `.ptl` 모델을 사용하여 새로운 운동 비디오를 분류하는 가이드입니다.

## 사용 방법

### 1. 명령줄에서 직접 실행

```bash
# 기본 사용법
python test_ptl_model.py --model MS-G3D/AI_Fit/weights-9-837.ptl --video test2_side_lateral_raise.mp4

# 사용자 정의 클래스 이름 지정
python test_ptl_model.py --model weights-9-837.ptl --video test2_side_lateral_raise.mp4 --class-names benchpress deadlift lunges side_lateral_raise squat
```

### 2. Python 스크립트에서 사용

```python
from convert_pt_to_ptl import VideoClassifier

# 모델 로드
classifier = VideoClassifier('MS-G3D/AI_Fit/weights-9-837.ptl')

# 비디오 분류
result = classifier.classify_video('your_video.mp4')

# 결과 출력
classifier.print_classification_result(result)

# 결과 딕셔너리 접근
print(f"예측 클래스: {result['predicted_class']}")
print(f"신뢰도: {result['confidence']:.2%}")
print(f"모든 확률: {result['all_probabilities']}")
```

### 3. 여러 비디오 일괄 처리

```python
from convert_pt_to_ptl import VideoClassifier
import os

# 모델 로드
classifier = VideoClassifier('MS-G3D/AI_Fit/weights-9-837.ptl')

# 비디오 폴더
video_folder = 'test_videos'
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

# 모든 비디오 분류
results = []
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    print(f"\n처리 중: {video_file}")
    
    try:
        result = classifier.classify_video(video_path)
        result['filename'] = video_file
        results.append(result)
        
        print(f"  -> {result['predicted_class']} ({result['confidence']:.2%})")
    except Exception as e:
        print(f"  오류: {e}")

# 결과 요약
print("\n=== 분류 결과 요약 ===")
for r in results:
    print(f"{r['filename']:30s} -> {r['predicted_class']:20s} ({r['confidence']:.2%})")
```

## 지원되는 클래스

기본 클래스 (6개):
1. `benchpress` - 벤치프레스
2. `deadlift` - 데드리프트
3. `lunges` - 런지
4. `side_lateral_raise` - 사이드 레터럴 레이즈
5. `squat` - 스쿼트
6. `others` - 기타 운동

## 출력 형식

```
============================================================
분류 결과
============================================================
예측된 클래스: squat
신뢰도: 95.23%

모든 클래스별 확률:
  squat                ██████████████████████████████████████████████████ 95.23%
  deadlift             ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 3.45%
  others               ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.87%
  benchpress           ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.25%
  lunges               ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.15%
  side_lateral_raise   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.05%
============================================================
```

## 문제 해결

### 1. MediaPipe가 포즈를 감지하지 못하는 경우
- 비디오가 너무 어둡거나 사람이 명확하게 보이지 않을 수 있습니다
- 카메라가 전신을 촬영하도록 확인하세요

### 2. 메모리 부족 오류
- `--max-frames` 옵션으로 처리 프레임 수를 줄이세요:
  ```bash
  python test_ptl_model.py --model model.ptl --video video.mp4 --max-frames 150
  ```

### 3. 정확도가 낮은 경우
- 비디오 품질 확인
- 운동 동작이 명확하게 보이는지 확인
- 학습 데이터와 유사한 각도/환경에서 촬영했는지 확인

## 데이터 형식

모델은 다음 형식의 입력을 기대합니다:
- **입력 형태**: `(1, 3, T, 25, 1)`
  - `1`: 배치 크기
  - `3`: 좌표 (x, y, z)
  - `64`: 프레임 수 (가변)
  - `25`: NTU RGB+D 관절점 개수
  - `1`: 사람 수

- **좌표 범위**: 0.0 ~ 1.0 (정규화된 MediaPipe 좌표)

## 성능 최적화

- GPU 사용 시 자동으로 CUDA 가속 적용
- CPU만 사용하는 경우 프레임 수를 줄여 처리 속도 향상
- 비디오 해상도를 낮추면 MediaPipe 처리 속도 향상

