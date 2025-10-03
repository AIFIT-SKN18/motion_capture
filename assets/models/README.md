# Models Directory

이 디렉토리에 TensorFlow Lite 모델 파일을 배치하세요.

## 필요한 파일

- `weights_8_744_simplified_float16.tflite`

## 모델 정보

- **입력 Shape**: [1, 3, 64, 25, 1]
  - Batch: 1
  - Channels: 3 (x, y, z)
  - Time: 64 frames
  - Vertices: 25 joints (NTU RGB+D format)
  - Persons: 1

- **출력 Shape**: [1, 6]
  - 6개 클래스: benchpress, deadlift, lunges, side_lateral_raise, squat, others

- **데이터 타입**: float16
