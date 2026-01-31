# motion_capture 개발 정리

## 개요
이 프로젝트는 **MediaPipe Pose → NTU RGB+D 25관절 포맷 변환 → MS‑G3D 분류 모델** 흐름으로 운동 동작을 분류합니다. 100프레임 고정 길이 시퀀스를 기준으로 학습/추론하며, PyTorch 모델을 ONNX → TFLite로 변환해 로컬 추론에 사용합니다.

## 파일 구성
- `MS_G3D_Improved_100frame.ipynb`
  - 100프레임 기준 데이터 전처리/학습/추론을 정리한 메인 노트북
- `pose_extractor.py`
  - MediaPipe Pose 결과를 **NTU RGB+D 25관절** 포맷으로 변환
  - 비디오에서 프레임별 스켈레톤 시퀀스 추출
- `convert_to_tflite_100frames.py`
  - PyTorch MS‑G3D 모델 로드 → ONNX 내보내기 → 단순화 → onnx2tf로 TFLite 변환
- `evaluate_single_100frames.py`
  - 비디오 한 건씩 TFLite 추론, softmax 확률/정답 비교
- `weights-9-837_100frame.pt`
  - 100프레임 기준 학습된 MS‑G3D 가중치
- `tflite_simplified_float16.tflite`, `tflite_simplified_float32.tflite`
  - 변환된 TFLite 모델
- `requirements.txt`
  - 실행에 필요한 패키지 목록

## 개발 흐름 요약
1) **포즈 추출 (MediaPipe → NTU 25관절)**
- `pose_extractor.PoseExtractor.extract_from_video()`가 비디오에서 프레임별 관절을 추출합니다.
- MediaPipe 33관절을 NTU 25관절 포맷으로 매핑합니다.
- 프레임을 최대 300까지 읽으며, 인식 실패 시 직전 프레임을 복사합니다.

2) **100프레임 시퀀스 전처리**
- `evaluate_single_100frames.py`에서 (T, 25, 3) 형태의 시퀀스를 생성한 뒤
  - (3, T, 25, 1)로 전치
  - T가 100보다 길면 균등 샘플링, 짧으면 edge padding
  - 최종 입력은 (1, 100, 25, 1, 3)으로 변환

3) **모델 변환 (PyTorch → ONNX → TFLite)**
- `convert_to_tflite_100frames.py`가 PyTorch 가중치 로드 후 ONNX로 내보냅니다.
- ONNX 모델을 onnxsim으로 단순화한 뒤, onnx2tf로 TFLite를 생성합니다.
- 변환 후 float16/float32 모델 파일이 생성됩니다.

4) **추론/평가**
- `evaluate_single_100frames.py`에서 TFLite Interpreter로 추론합니다.
- 클래스는 6개로 정의되어 있습니다:
  - benchpress, deadlift, lunges, side_lateral_raise, squat, others
- 여러 테스트 비디오에 대해 예측/정확도 로그를 출력합니다.

## 환경/경로 주의사항
현재 스크립트에는 **절대 경로**가 하드코딩되어 있습니다. 아래 경로는 환경에 맞게 수정이 필요합니다.
- `convert_to_tflite_100frames.py`
  - MS‑G3D 모듈 경로(`sys.path.insert`)
  - `PT_MODEL_PATH`, `OUTPUT_ONNX`, `OUTPUT_DIR`
- `evaluate_single_100frames.py`
  - `test_folder`, `tflite_model_path`

또한, `convert_to_tflite_100frames.py`는 MS‑G3D의 `model.msg3d` 및 `graph.ntu_rgb_d` 모듈이 필요합니다.
현재 저장소에는 MS‑G3D 소스가 포함되어 있지 않으므로, 로컬에 해당 코드가 존재해야 합니다.

## 빠른 실행 순서(로컬)
1) 의존성 설치
   - `pip install -r requirements.txt`
2) 모델 변환
   - `python convert_to_tflite_100frames.py`
3) 단일 비디오 추론
   - `python evaluate_single_100frames.py`

## 비고
- README 기존 내용은 일부 인코딩 문제가 있어, 실제 파일 기준으로 다시 정리했습니다.
- 필요하면 경로를 상대 경로로 바꾸고, MS‑G3D를 서브모듈/하위폴더로 포함하는 방식으로 개선 가능합니다.
