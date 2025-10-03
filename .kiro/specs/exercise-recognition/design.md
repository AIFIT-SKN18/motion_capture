# Design Document

## Overview

이 애플리케이션은 Flutter 기반의 Android 운동 동작 인식 시스템입니다. TensorFlow Lite float16 모델을 사용하여 실시간 카메라 및 동영상에서 운동 동작을 분류합니다. MediaPipe를 통해 포즈를 감지하고, NTU RGB+D 형식으로 변환하여 모델에 입력합니다.

### 주요 기술 스택
- **프레임워크**: Flutter (Dart)
- **ML 모델**: TensorFlow Lite (float16)
- **포즈 감지**: Google ML Kit (MediaPipe Pose Detection)
- **플랫폼**: Android (API 21+)

### 핵심 기능
1. 홈 화면에서 실시간/동영상 모드 선택
2. 실시간 카메라 기반 운동 동작 인식 (3초 카운트다운)
3. 갤러리 동영상 분석
4. 6개 운동 클래스 분류

## Architecture

### 아키텍처 패턴
간단한 **레이어드 아키텍처**를 사용합니다:

```
┌─────────────────────────────────────┐
│         Presentation Layer          │
│  (UI Screens: Home, Live, Video)    │
└─────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│         Service Layer               │
│  (PoseClassifier, PoseDetector)     │
└─────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│         Data Layer                  │
│  (TFLite Model, MediaPipe)          │
└─────────────────────────────────────┘
```

### 디렉토리 구조
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

## Components and Interfaces

### 1. Presentation Layer

#### HomeScreen
홈 화면 - 기능 선택 UI

**책임**:
- 실시간 인식 버튼 표시
- 동영상 분석 버튼 표시
- 화면 네비게이션

**주요 메서드**:
```dart
class HomeScreen extends StatelessWidget {
  void navigateToLive(BuildContext context);
  void navigateToVideo(BuildContext context);
}
```

#### LiveScreen
실시간 동작 인식 화면

**책임**:
- 카메라 권한 요청 및 프리뷰 표시
- 3초 카운트다운 표시
- 실시간 포즈 감지 및 분류
- 분류 결과 실시간 표시

**주요 메서드**:
```dart
class LiveScreen extends StatefulWidget {
  Future<void> initializeCamera();
  Future<void> startCountdown();
  Future<void> processFrame(CameraImage image);
  void displayResult(String exercise, double confidence);
}
```

**상태 관리**:
- `isRecording`: 인식 진행 중 여부
- `countdown`: 카운트다운 숫자 (3, 2, 1)
- `currentExercise`: 현재 인식된 운동
- `confidence`: 신뢰도

#### VideoScreen
동영상 분석 화면

**책임**:
- 갤러리에서 동영상 선택
- 동영상 프레임 추출
- 포즈 감지 및 분류
- 분석 결과 표시

**주요 메서드**:
```dart
class VideoScreen extends StatefulWidget {
  Future<void> pickVideo();
  Future<void> analyzeVideo(File videoFile);
  Future<List<Pose>> extractPoses(File videoFile);
  void displayResult(String exercise, double confidence);
}
```

### 2. Service Layer

#### PoseClassifier
TensorFlow Lite 모델 관리 및 추론

**책임**:
- TFLite 모델 로드 및 초기화
- 포즈 데이터 전처리
- 모델 추론 실행
- 결과 후처리 (소프트맥스)

**주요 메서드**:
```dart
class PoseClassifier {
  Future<void> initialize();
  Future<Map<String, dynamic>> classify(List<List<PoseLandmark>> sequence);
  Float32List preprocessPose(List<List<PoseLandmark>> sequence);
  List<double> applySoftmax(List<double> logits);
  void dispose();
}
```

**속성**:
- `interpreter`: TFLite 인터프리터
- `inputShape`: [1, 3, 64, 25, 1]
- `outputShape`: [1, 6]
- `classes`: ['benchpress', 'deadlift', 'lunges', 'side_lateral_raise', 'squat', 'others']

#### PoseDetector
MediaPipe 포즈 감지

**책임**:
- MediaPipe 초기화
- 이미지/프레임에서 포즈 감지
- 33개 랜드마크 추출

**주요 메서드**:
```dart
class PoseDetector {
  Future<void> initialize();
  Future<List<Pose>> detectPose(InputImage image);
  void dispose();
}
```

### 3. Utility Layer

#### NTUConverter
MediaPipe 랜드마크를 NTU RGB+D 형식으로 변환

**책임**:
- 33개 MediaPipe 랜드마크 → 25개 NTU 관절점 매핑
- 좌표 정규화 (0~1 범위)
- (T, V, C) → (C, T, V, M) transpose

**주요 메서드**:
```dart
class NTUConverter {
  static List<List<double>> mediapipeToNTU(
    List<PoseLandmark> landmarks,
    double imageWidth,
    double imageHeight
  );
  
  static Float32List createModelInput(
    List<List<List<double>>> sequence  // T x V x C
  );
}
```

**NTU 관절점 매핑** (25개):
```
0: spine_base (hip center)
1: spine_mid
2: neck
3: head
4-7: left arm (shoulder, elbow, wrist, hand)
8-11: right arm
12-15: left leg (hip, knee, ankle, foot)
16-19: right leg
20-24: additional points
```

## Data Models

### PoseLandmark (from MediaPipe)
```dart
class PoseLandmark {
  double x;      // 픽셀 좌표
  double y;      // 픽셀 좌표
  double z;      // 상대적 깊이
  double visibility;
}
```

### ClassificationResult
```dart
class ClassificationResult {
  String exercise;           // 분류된 운동 이름
  double confidence;         // 신뢰도 (0~1)
  int classIndex;           // 클래스 인덱스 (0~5)
  Map<String, double> allProbabilities;  // 모든 클래스 확률
}
```

### ModelInput
```dart
// Shape: [1, 3, 64, 25, 1]
// - Batch: 1
// - Channels: 3 (x, y, z)
// - Time: 64 frames
// - Vertices: 25 joints
// - Persons: 1
Float32List modelInput;
```

## Error Handling

### 오류 유형 및 처리 전략

#### 1. 모델 로딩 오류
```dart
try {
  await poseClassifier.initialize();
} catch (e) {
  showDialog(
    context: context,
    builder: (context) => AlertDialog(
      title: Text('모델 로드 실패'),
      content: Text('TFLite 모델을 로드할 수 없습니다.'),
    ),
  );
}
```

#### 2. 카메라 권한 오류
```dart
if (cameraStatus.isDenied) {
  showDialog(
    context: context,
    builder: (context) => AlertDialog(
      title: Text('카메라 권한 필요'),
      content: Text('실시간 인식을 위해 카메라 권한이 필요합니다.'),
      actions: [
        TextButton(
          onPressed: () => openAppSettings(),
          child: Text('설정으로 이동'),
        ),
      ],
    ),
  );
}
```

#### 3. 포즈 감지 실패
```dart
if (poses.isEmpty) {
  // 포즈가 감지되지 않음 - 사용자에게 알림
  setState(() {
    currentExercise = '포즈를 감지할 수 없습니다';
  });
  return;
}
```

#### 4. 추론 오류
```dart
try {
  result = await poseClassifier.classify(poseSequence);
} catch (e) {
  print('추론 오류: $e');
  result = ClassificationResult(
    exercise: 'others',
    confidence: 0.0,
  );
}
```

## Testing Strategy

### 1. 단위 테스트 (Unit Tests)

#### NTUConverter 테스트
```dart
test('MediaPipe to NTU conversion', () {
  final landmarks = createMockLandmarks();
  final ntuJoints = NTUConverter.mediapipeToNTU(landmarks, 640, 480);
  
  expect(ntuJoints.length, 25);
  expect(ntuJoints[0].length, 3);  // x, y, z
  expect(ntuJoints[0][0], greaterThanOrEqualTo(0.0));
  expect(ntuJoints[0][0], lessThanOrEqualTo(1.0));
});
```

#### PoseClassifier 테스트
```dart
test('Model input shape', () {
  final input = createMockPoseSequence(64);
  final modelInput = poseClassifier.preprocessPose(input);
  
  expect(modelInput.length, 1 * 3 * 64 * 25 * 1);  // 4800
});

test('Softmax output', () {
  final logits = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
  final probs = poseClassifier.applySoftmax(logits);
  
  expect(probs.length, 6);
  expect(probs.reduce((a, b) => a + b), closeTo(1.0, 0.001));
});
```

### 2. 통합 테스트 (Integration Tests)

#### 전체 파이프라인 테스트
```dart
testWidgets('Video analysis flow', (tester) async {
  await tester.pumpWidget(MyApp());
  
  // 홈 화면에서 동영상 분석 선택
  await tester.tap(find.text('동영상 분석'));
  await tester.pumpAndSettle();
  
  // 동영상 선택 (mock)
  await tester.tap(find.text('동영상 선택'));
  await tester.pumpAndSettle();
  
  // 분석 시작
  await tester.tap(find.text('분석 시작'));
  await tester.pump(Duration(seconds: 5));
  
  // 결과 확인
  expect(find.textContaining('분류 결과'), findsOneWidget);
});
```

### 3. 위젯 테스트 (Widget Tests)

#### HomeScreen 테스트
```dart
testWidgets('Home screen buttons', (tester) async {
  await tester.pumpWidget(MaterialApp(home: HomeScreen()));
  
  expect(find.text('실시간 동작 인식'), findsOneWidget);
  expect(find.text('동영상 분석'), findsOneWidget);
});
```

### 4. 수동 테스트 시나리오

#### 실시간 인식 테스트
1. 앱 실행 → 홈 화면 확인
2. "실시간 동작 인식" 버튼 탭
3. 카메라 권한 허용
4. "시작" 버튼 탭
5. 3초 카운트다운 확인
6. 운동 동작 수행 (예: squat)
7. 실시간 분류 결과 확인
8. "중지" 버튼으로 종료

#### 동영상 분석 테스트
1. "동영상 분석" 버튼 탭
2. 갤러리에서 운동 동영상 선택
3. "분석 시작" 버튼 탭
4. 로딩 인디케이터 확인
5. 분류 결과 및 신뢰도 확인

## Implementation Details

### TensorFlow Lite 모델 통합

#### 모델 초기화
```dart
Future<void> initialize() async {
  try {
    // 모델 로드
    interpreter = await Interpreter.fromAsset(
      'assets/models/weights_8_744_simplified_float16.tflite',
    );
    
    // 입력/출력 shape 확인
    print('Input shape: ${interpreter.getInputTensor(0).shape}');
    print('Output shape: ${interpreter.getOutputTensor(0).shape}');
    
    isInitialized = true;
  } catch (e) {
    throw Exception('모델 로드 실패: $e');
  }
}
```

#### 데이터 전처리
```dart
Float32List preprocessPose(List<List<PoseLandmark>> sequence) {
  // 1. 64 프레임 준비 (부족하면 패딩)
  List<List<PoseLandmark>> frames = [];
  for (int i = 0; i < 64; i++) {
    if (i < sequence.length) {
      frames.add(sequence[i]);
    } else {
      frames.add(sequence.last);  // 마지막 프레임으로 패딩
    }
  }
  
  // 2. MediaPipe → NTU 변환 (T x V x C)
  List<List<List<double>>> ntuSequence = [];
  for (var frame in frames) {
    var ntuJoints = NTUConverter.mediapipeToNTU(frame, 640, 480);
    ntuSequence.add(ntuJoints);
  }
  
  // 3. (T, V, C) → (C, T, V, M) transpose
  List<double> flatInput = [];
  for (int c = 0; c < 3; c++) {        // Channels
    for (int t = 0; t < 64; t++) {     // Time
      for (int v = 0; v < 25; v++) {   // Vertices
        for (int m = 0; m < 1; m++) {  // Persons
          flatInput.add(ntuSequence[t][v][c]);
        }
      }
    }
  }
  
  return Float32List.fromList(flatInput);
}
```

#### 추론 실행
```dart
Future<ClassificationResult> classify(List<List<PoseLandmark>> sequence) async {
  // 전처리
  var input = preprocessPose(sequence);
  
  // 출력 버퍼 준비
  var output = List.filled(6, 0.0).reshape([1, 6]);
  
  // 추론
  interpreter.run(input.reshape([1, 3, 64, 25, 1]), output);
  
  // 소프트맥스 적용
  var probabilities = applySoftmax(output[0]);
  
  // 최고 확률 클래스 찾기
  int maxIndex = 0;
  double maxProb = probabilities[0];
  for (int i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      maxIndex = i;
    }
  }
  
  return ClassificationResult(
    exercise: classes[maxIndex],
    confidence: maxProb,
    classIndex: maxIndex,
    allProbabilities: {
      for (int i = 0; i < classes.length; i++)
        classes[i]: probabilities[i]
    },
  );
}
```

### MediaPipe 통합

#### 포즈 감지
```dart
Future<List<Pose>> detectPose(InputImage image) async {
  final poseDetector = PoseDetector(
    options: PoseDetectorOptions(
      mode: PoseDetectionMode.stream,  // 실시간용
    ),
  );
  
  final poses = await poseDetector.processImage(image);
  return poses;
}
```

### 실시간 처리 최적화

#### 프레임 스킵
```dart
int frameCount = 0;
const int skipFrames = 2;  // 2프레임마다 처리

void processFrame(CameraImage image) {
  frameCount++;
  if (frameCount % skipFrames != 0) return;
  
  // 포즈 감지 및 분류
  detectAndClassify(image);
}
```

## Performance Considerations

### 메모리 관리
- 포즈 히스토리는 최대 64 프레임만 유지
- 사용하지 않는 리소스는 즉시 해제
- 백그라운드 전환 시 카메라 및 모델 리소스 해제

### 추론 최적화
- TFLite float16 모델 사용 (모델 크기 감소)
- 실시간 모드에서 프레임 스킵 (2프레임마다 처리)
- 비동기 처리로 UI 블로킹 방지

### UI 반응성
- 모든 무거운 작업은 별도 isolate에서 실행
- 로딩 인디케이터로 사용자 피드백 제공
- 카운트다운으로 사용자 준비 시간 제공

## Security Considerations

### 권한 관리
- 카메라 권한: 실시간 인식에 필요
- 저장소 권한: 동영상 선택에 필요
- 런타임 권한 요청 및 거부 처리

### 데이터 프라이버시
- 모든 처리는 로컬에서 수행 (서버 전송 없음)
- 동영상 데이터는 메모리에서만 처리
- 영구 저장소에 사용자 데이터 저장하지 않음

## Deployment

### 빌드 설정

#### pubspec.yaml
```yaml
dependencies:
  flutter:
    sdk: flutter
  camera: ^0.10.0
  google_mlkit_pose_detection: ^0.10.0
  image_picker: ^1.0.0
  tflite_flutter: ^0.10.0
  permission_handler: ^11.0.0

flutter:
  assets:
    - assets/models/weights_8_744_simplified_float16.tflite
```

#### AndroidManifest.xml
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-feature android:name="android.hardware.camera" />
```

### Release 빌드
```bash
flutter build apk --release
```

## Future Enhancements

### 단기 개선사항
1. 운동 횟수 카운팅 기능
2. 운동 자세 교정 피드백
3. 운동 기록 저장 및 통계

### 장기 개선사항
1. 더 많은 운동 클래스 지원
2. 사용자 맞춤형 운동 추천
3. 소셜 기능 (친구와 경쟁)
4. 웨어러블 기기 연동
