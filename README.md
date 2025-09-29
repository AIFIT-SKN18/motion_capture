# AIFit 운동 자세 인식 앱

MS-G3D 모델을 사용한 실시간 운동 자세 분류 Flutter 앱입니다.

## 지원하는 운동 클래스

- **벤치프레스** (benchpress): 가슴 운동
- **데드리프트** (deadlift): 전신 운동  
- **런지** (lunges): 하체 운동
- **사이드 레터럴 레이즈** (side_lateral_raise): 어깨 운동
- **스쿼트** (squat): 하체 운동
- **기타** (others): 인식되지 않은 동작

## 주요 기능

- 실시간 카메라 피드
- MediaPipe를 통한 포즈 랜드마크 감지
- MS-G3D TFLite 모델을 사용한 운동 분류
- 운동별 색상 구분 및 가이드 메시지
- 신뢰도 표시

## 설치 및 실행

### 1. 모델 파일 준비

```bash
# android/assets/models/msg3d_direct.tflite 파일을 
# assets/models/msg3d_direct.tflite로 복사
cp android/assets/models/msg3d_direct.tflite assets/models/msg3d_direct.tflite
```

### 2. 의존성 설치

```bash
flutter pub get
```

### 3. 앱 실행

**Android:**
```bash
flutter run
```

**iOS:**
```bash
flutter run
```

## 기술 스택

- **Flutter**: 크로스 플랫폼 앱 개발
- **MediaPipe**: 포즈 랜드마크 감지
- **TFLite**: MS-G3D 모델 추론
- **Camera**: 실시간 카메라 피드

## 아키텍처

### PoseClassifier 클래스
- TFLite 모델 로드 및 초기화
- MediaPipe 포즈 데이터를 MS-G3D 입력 형태로 전처리
- 실시간 운동 분류 수행

### PoseDetectionScreen 위젯
- 카메라 피드 표시
- 포즈 랜드마크 시각화
- 운동 분류 결과 UI 표시
- 운동별 가이드 메시지

## 데이터 전처리

1. **키포인트 선택**: MediaPipe의 33개 랜드마크 중 MS-G3D에 필요한 17개 선택
2. **정규화**: 좌표를 정규화된 형태로 변환
3. **시퀀스 구성**: 최근 30프레임의 랜드마크 데이터를 누적
4. **모델 입력**: [1, 3, 300, 17, 2] 형태로 변환

## 성능 최적화

- 프레임 히스토리 제한 (최대 30프레임)
- 비동기 처리로 UI 블로킹 방지
- 신뢰도 임계값 설정으로 오분류 방지

## 문제 해결

### 모델 로드 실패
- `assets/models/msg3d_direct.tflite` 파일이 올바른 위치에 있는지 확인
- 모델 파일이 손상되지 않았는지 확인

### 카메라 권한 오류
- Android: `AndroidManifest.xml`에 카메라 권한 확인
- iOS: `Info.plist`에 카메라 사용 설명 확인

### 성능 문제
- 해상도를 `ResolutionPreset.medium`으로 설정
- 프레임 히스토리 길이 조정
- 신뢰도 임계값 조정