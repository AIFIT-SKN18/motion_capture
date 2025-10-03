# AI Fit - 운동 동작 인식 앱

TensorFlow Lite와 MediaPipe를 사용한 실시간 운동 동작 인식 Flutter 애플리케이션입니다.

## 주요 기능

- ✅ **실시간 동작 인식**: 카메라를 통한 실시간 운동 동작 분류
- ✅ **동영상 분석**: 갤러리에서 동영상을 선택하여 분석
- ✅ **6개 운동 클래스 지원**: Bench Press, Deadlift, Lunges, Side Lateral Raise, Squat, Others
- ✅ **3초 카운트다운**: 운동 준비 시간 제공
- ✅ **신뢰도 표시**: 분류 결과의 신뢰도를 백분율로 표시

## 기술 스택

- **프레임워크**: Flutter 3.9+
- **ML 모델**: TensorFlow Lite (float16)
- **포즈 감지**: Google ML Kit (MediaPipe Pose Detection)
- **플랫폼**: Android (API 21+)

## 설치 및 실행

### 1. 사전 요구사항

- Flutter SDK 3.9 이상
- Android Studio 또는 VS Code
- Android 기기 또는 에뮬레이터

### 2. 패키지 설치

```bash
flutter pub get
```

### 3. 모델 파일 준비

`assets/models/` 디렉토리에 다음 파일을 배치하세요:
- `weights_8_744_simplified_float16.tflite`

**모델 정보**:
- 입력 Shape: [1, 3, 64, 25, 1]
- 출력 Shape: [1, 6]
- 데이터 타입: float16

### 4. 앱 실행

```bash
flutter run
```

## 프로젝트 구조

```
lib/
├── main.dart                 # 앱 진입점
├── screens/
│   ├── home_screen.dart      # 홈 화면
│   ├── live_screen.dart      # 실시간 인식 화면
│   └── video_screen.dart     # 동영상 분석 화면
├── services/
│   ├── pose_classifier.dart  # TFLite 모델 추론
│   └── pose_detector.dart    # MediaPipe 포즈 감지
└── utils/
    └── ntu_converter.dart    # MediaPipe → NTU 변환

assets/
└── models/
    └── weights_8_744_simplified_float16.tflite
```

## 사용 방법

### 실시간 동작 인식

1. 홈 화면에서 "실시간 동작 인식" 버튼 클릭
2. 카메라 권한 허용
3. "시작" 버튼 클릭
4. 3초 카운트다운 후 운동 동작 수행
5. 실시간으로 분류 결과 확인
6. "중지" 버튼으로 종료

### 동영상 분석

1. 홈 화면에서 "동영상 분석" 버튼 클릭
2. "동영상 선택" 버튼으로 갤러리에서 동영상 선택
3. "분석 시작" 버튼 클릭
4. 분석 결과 및 신뢰도 확인

## 지원 운동

1. **Bench Press** (벤치프레스)
2. **Deadlift** (데드리프트)
3. **Lunges** (런지)
4. **Side Lateral Raise** (사이드 레터럴 레이즈)
5. **Squat** (스쿼트)
6. **Others** (기타)

## 기술 세부사항

### 포즈 데이터 처리

1. MediaPipe로 33개 포즈 랜드마크 추출
2. NTU RGB+D 25개 관절점 형식으로 변환
3. 좌표 정규화 (0~1 범위)
4. 64 프레임 시퀀스 준비
5. (T, V, C) → (C, T, V, M) transpose
6. TFLite 모델 추론

### 성능 최적화

- 실시간 모드에서 프레임 스킵 적용
- 비동기 처리로 UI 블로킹 방지
- 메모리 효율적인 리소스 관리
- 백그라운드 전환 시 자동 리소스 해제

## 권한

앱 실행 시 다음 권한이 필요합니다:

- **카메라**: 실시간 동작 인식
- **저장소**: 동영상 선택

## 빌드

### Debug 빌드
```bash
flutter build apk --debug
```

### Release 빌드
```bash
flutter build apk --release
```

## 문제 해결

### 모델 로드 실패
- `assets/models/` 디렉토리에 모델 파일이 있는지 확인
- `pubspec.yaml`에 assets 경로가 올바르게 설정되었는지 확인

### 카메라 권한 오류
- 설정에서 앱의 카메라 권한을 수동으로 허용

### 포즈 감지 실패
- 조명이 충분한 환경에서 촬영
- 전신이 카메라에 잘 보이도록 위치 조정


