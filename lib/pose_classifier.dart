import 'dart:typed_data';
import 'dart:math' as math;
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

class PoseClassifier {
  static const platform = MethodChannel('com.aifit.app/pytorch');
  bool _isInitialized = false;

  // 메모리 추적을 위한 변수
  int _inferenceCount = 0;

  // 운동 클래스 정의
  static const List<String> exerciseClasses = [
    'benchpress',    // 0
    'deadlift',      // 1
    'lunges',        // 2
    'side_lateral_raise', // 3
    'squat',         // 4
    'others'        // 5
  ];

  // 5D 모델 입력 형태 (실제 모델: [1, 3, 64, 25, 1])
  static const int inputFrames = 64; // 프레임 수 (T)
  static const int numKeypoints = 25; // 실제 모델의 관절점 수
  static const int numChannels = 3; // x, y, z 좌표
  static const int numPersons = 1; // 사람 수

  // 포즈 히스토리 저장
  List<List<PoseLandmark>> _poseHistory = [];
  static const int maxHistoryLength = 64; // 모델 입력 프레임 수와 동일하게 설정

  // 분류 빈도 제어 (실시간 성능 최적화)
  int _frameCount = 0;
  static const int classificationInterval = 2; // 2프레임마다 분류 수행 (성능과 반응성 균형)

  // 5D 모델은 5D 입력을 지원
  static const bool _isInput5D = true;

  // 실제 모델이 기대하는 입력 크기
  int _expectedInputSize = 4800; // 실제 모델 크기: 1 * 3 * 64 * 25 * 1 = 4800

  // 로그 출력 제어
  bool _hasLoggedNtuConversion = false;
  bool _hasLoggedNtuStart = false;

  Future<void> initialize() async {
    try {
      print('🚀 PoseClassifier 초기화 시작...');

      // 초기화 전 메모리 상태
      _logMemoryUsage('모델 로드 전');

      // 플랫폼 채널을 통해 PyTorch 모델 로드
      final String result = await platform.invokeMethod('loadModel', {
        'modelPath': 'assets/models/model_lite.ptl'
      });

      // 초기화 후 메모리 상태
      _logMemoryUsage('모델 로드 후');

      print('✅ $result');
      print('📊 모델 입력 정보:');
      print('  - 입력 형태: [1, 3, 64, 25, 1] (5D)');
      print('  - 예상 입력 크기: $_expectedInputSize');
      print('  - 출력 클래스 수: ${exerciseClasses.length}');

      _isInitialized = true;
      print('🎉 PoseClassifier 초기화 완료!');

    } catch (e) {
      print('❌ PoseClassifier 초기화 실패: $e');
      throw Exception('모델 로드 실패: $e');
    }
  }

  // 메모리 사용량 로깅 헬퍼 메서드
  void _logMemoryUsage(String label) {
    try {
      print('💾 [$label] 메모리 정보 수집 중...');

      // Android/iOS 네이티브 메모리 정보
      if (Platform.isAndroid || Platform.isIOS) {
        platform.invokeMethod('getMemoryInfo').then((result) {
          if (result != null && result is Map) {
            print('💾 [$label] 네이티브 메모리:');
            print('  - 총 메모리: ${result['totalMemory']} MB');
            print('  - 사용 가능 메모리: ${result['availableMemory']} MB');
            print('  - 사용 중 메모리: ${result['usedMemory']} MB');
            print('  - 임계값: ${result['threshold']} MB');
            print('  - Native Heap 크기: ${result['nativeHeapSize']} MB');
            print('  - Native Heap 할당: ${result['nativeHeapAllocated']} MB');
            print('  - Native Heap 여유: ${result['nativeHeapFree']} MB');
            print('  - 메모리 부족 상태: ${result['lowMemory']}');

            // 메모리 사용률 계산
            final totalMem = result['totalMemory'] as int;
            final usedMem = result['usedMemory'] as int;
            final usagePercent = (usedMem / totalMem * 100).toStringAsFixed(1);
            print('  - 메모리 사용률: $usagePercent%');
          }
        }).catchError((e) {
          print('⚠️ 네이티브 메모리 정보 가져오기 실패: $e');
        });
      }
    } catch (e) {
      print('⚠️ 메모리 사용량 로깅 실패: $e');
    }
  }

  void dispose() {
    if (_isInitialized) {
      // PyTorch Lite는 자동으로 메모리 관리
      _isInitialized = false;
    }
  }

  // 모델 상태 확인
  bool get isInitialized => _isInitialized;

  // MediaPipe 포즈를 NTU RGB+D 25개 관절점으로 변환 - 정규화 버전
  List<List<double>> _mediapipeToNtuNormalized(List<PoseLandmark> landmarks, double imageWidth, double imageHeight) {
    List<List<double>> joints = List.generate(25, (_) => List.filled(3, 0.0));

    if (landmarks.length < 33) {
      print('❌ [NTU 변환] 충분한 랜드마크가 없습니다: ${landmarks.length}/33');
      return joints;
    }

    if (!_hasLoggedNtuStart) {
      print('🔄 [NTU 변환] 25개 관절점 변환 시작 (정규화 적용)');
      print('  - 이미지 크기: ${imageWidth}x${imageHeight}');
      _hasLoggedNtuStart = true;
    }

    // 좌표 정규화 함수 (픽셀 → 0~1 범위)
    double normalizeX(double x) => (x / imageWidth).clamp(0.0, 1.0);
    double normalizeY(double y) => (y / imageHeight).clamp(0.0, 1.0);
    // z는 이미 상대적 깊이이므로 그대로 사용하되 스케일 조정 및 클리핑
    double normalizeZ(double z) => (z / imageWidth).clamp(-1.0, 1.0);

    // NTU RGB+D 25개 관절점 매핑 (정규화된 좌표 사용)

    // 0: spine_base (hip center) - 양쪽 힙의 중점
    joints[0] = [
      normalizeX((landmarks[23].x + landmarks[24].x) / 2),
      normalizeY((landmarks[23].y + landmarks[24].y) / 2),
      normalizeZ((landmarks[23].z + landmarks[24].z) / 2)
    ];

    // 1: spine_mid - hip center와 shoulder center의 중점
    List<double> hipCenter = joints[0];
    List<double> shoulderCenter = [
      normalizeX((landmarks[11].x + landmarks[12].x) / 2),
      normalizeY((landmarks[11].y + landmarks[12].y) / 2),
      normalizeZ((landmarks[11].z + landmarks[12].z) / 2)
    ];
    joints[1] = [
      (hipCenter[0] + shoulderCenter[0]) / 2,
      (hipCenter[1] + shoulderCenter[1]) / 2,
      (hipCenter[2] + shoulderCenter[2]) / 2
    ];

    // 2: neck - shoulder center에서 약간 위로
    joints[2] = [
      shoulderCenter[0],
      shoulderCenter[1] - 0.05,
      shoulderCenter[2]
    ];

    // 3: head - nose
    joints[3] = [normalizeX(landmarks[0].x), normalizeY(landmarks[0].y), normalizeZ(landmarks[0].z)];

    // 4-7: left arm
    joints[4] = [normalizeX(landmarks[11].x), normalizeY(landmarks[11].y), normalizeZ(landmarks[11].z)];
    joints[5] = [normalizeX(landmarks[13].x), normalizeY(landmarks[13].y), normalizeZ(landmarks[13].z)];
    joints[6] = [normalizeX(landmarks[15].x), normalizeY(landmarks[15].y), normalizeZ(landmarks[15].z)];
    joints[7] = [normalizeX(landmarks[15].x), normalizeY(landmarks[15].y), normalizeZ(landmarks[15].z)];

    // 8-11: right arm
    joints[8] = [normalizeX(landmarks[12].x), normalizeY(landmarks[12].y), normalizeZ(landmarks[12].z)];
    joints[9] = [normalizeX(landmarks[14].x), normalizeY(landmarks[14].y), normalizeZ(landmarks[14].z)];
    joints[10] = [normalizeX(landmarks[16].x), normalizeY(landmarks[16].y), normalizeZ(landmarks[16].z)];
    joints[11] = [normalizeX(landmarks[16].x), normalizeY(landmarks[16].y), normalizeZ(landmarks[16].z)];

    // 12-15: left leg
    joints[12] = [normalizeX(landmarks[23].x), normalizeY(landmarks[23].y), normalizeZ(landmarks[23].z)];
    joints[13] = [normalizeX(landmarks[25].x), normalizeY(landmarks[25].y), normalizeZ(landmarks[25].z)];
    joints[14] = [normalizeX(landmarks[27].x), normalizeY(landmarks[27].y), normalizeZ(landmarks[27].z)];
    joints[15] = [normalizeX(landmarks[31].x), normalizeY(landmarks[31].y), normalizeZ(landmarks[31].z)];

    // 16-19: right leg
    joints[16] = [normalizeX(landmarks[24].x), normalizeY(landmarks[24].y), normalizeZ(landmarks[24].z)];
    joints[17] = [normalizeX(landmarks[26].x), normalizeY(landmarks[26].y), normalizeZ(landmarks[26].z)];
    joints[18] = [normalizeX(landmarks[28].x), normalizeY(landmarks[28].y), normalizeZ(landmarks[28].z)];
    joints[19] = [normalizeX(landmarks[32].x), normalizeY(landmarks[32].y), normalizeZ(landmarks[32].z)];

    // 20-24: additional points
    joints[20] = shoulderCenter;
    joints[21] = [normalizeX(landmarks[17].x), normalizeY(landmarks[17].y), normalizeZ(landmarks[17].z)];
    joints[22] = [normalizeX(landmarks[21].x), normalizeY(landmarks[21].y), normalizeZ(landmarks[21].z)];
    joints[23] = [normalizeX(landmarks[18].x), normalizeY(landmarks[18].y), normalizeZ(landmarks[18].z)];
    joints[24] = [normalizeX(landmarks[22].x), normalizeY(landmarks[22].y), normalizeZ(landmarks[22].z)];


    // 완료 로그는 1번만 출력
    if (!_hasLoggedNtuConversion) {
      print('✅ [NTU 변환] 25개 관절점 변환 완료');
      print('📍 [NTU 변환] NTU 관절점 샘플 (첫 5개):');
      for (int i = 0; i < math.min(5, joints.length); i++) {
        print('   [$i] x:${joints[i][0].toStringAsFixed(3)}, y:${joints[i][1].toStringAsFixed(3)}, z:${joints[i][2].toStringAsFixed(3)}');
      }
      _hasLoggedNtuConversion = true;
    }
    return joints;
  }

  // MediaPipe 포즈를 NTU RGB+D 25개 관절점으로 변환 (원본 - 정규화 없음)
  List<List<double>> _mediapipeToNtu(List<PoseLandmark> landmarks) {
    // 실시간 포즈 감지용 (정규화하지 않음)
    List<List<double>> joints = List.generate(25, (_) => List.filled(3, 0.0));

    if (landmarks.length < 33) {
      return joints;
    }

    // 0: spine_base
    joints[0] = [(landmarks[23].x + landmarks[24].x) / 2, (landmarks[23].y + landmarks[24].y) / 2, (landmarks[23].z + landmarks[24].z) / 2];

    List<double> hipCenter = joints[0];
    List<double> shoulderCenter = [(landmarks[11].x + landmarks[12].x) / 2, (landmarks[11].y + landmarks[12].y) / 2, (landmarks[11].z + landmarks[12].z) / 2];

    joints[1] = [(hipCenter[0] + shoulderCenter[0]) / 2, (hipCenter[1] + shoulderCenter[1]) / 2, (hipCenter[2] + shoulderCenter[2]) / 2];
    joints[2] = [shoulderCenter[0], shoulderCenter[1] - 0.05, shoulderCenter[2]];
    joints[3] = [landmarks[0].x, landmarks[0].y, landmarks[0].z];
    joints[4] = [landmarks[11].x, landmarks[11].y, landmarks[11].z];
    joints[5] = [landmarks[13].x, landmarks[13].y, landmarks[13].z];
    joints[6] = [landmarks[15].x, landmarks[15].y, landmarks[15].z];
    joints[7] = [landmarks[15].x, landmarks[15].y, landmarks[15].z];
    joints[8] = [landmarks[12].x, landmarks[12].y, landmarks[12].z];
    joints[9] = [landmarks[14].x, landmarks[14].y, landmarks[14].z];
    joints[10] = [landmarks[16].x, landmarks[16].y, landmarks[16].z];
    joints[11] = [landmarks[16].x, landmarks[16].y, landmarks[16].z];
    joints[12] = [landmarks[23].x, landmarks[23].y, landmarks[23].z];
    joints[13] = [landmarks[25].x, landmarks[25].y, landmarks[25].z];
    joints[14] = [landmarks[27].x, landmarks[27].y, landmarks[27].z];
    joints[15] = [landmarks[31].x, landmarks[31].y, landmarks[31].z];
    joints[16] = [landmarks[24].x, landmarks[24].y, landmarks[24].z];
    joints[17] = [landmarks[26].x, landmarks[26].y, landmarks[26].z];
    joints[18] = [landmarks[28].x, landmarks[28].y, landmarks[28].z];
    joints[19] = [landmarks[32].x, landmarks[32].y, landmarks[32].z];
    joints[20] = shoulderCenter;
    joints[21] = [landmarks[17].x, landmarks[17].y, landmarks[17].z];
    joints[22] = [landmarks[21].x, landmarks[21].y, landmarks[21].z];
    joints[23] = [landmarks[18].x, landmarks[18].y, landmarks[18].z];
    joints[24] = [landmarks[22].x, landmarks[22].y, landmarks[22].z];

    return joints;
  }

  // _preprocessPoseData는 더 이상 사용하지 않음 - _create5DInput에서 직접 처리


  // 포즈 히스토리에 추가
  void _addPoseToHistory(List<PoseLandmark> landmarks) {
    _poseHistory.add(landmarks);
    if (_poseHistory.length > maxHistoryLength) {
      _poseHistory.removeAt(0);
    }
  }

  // 실시간 포즈 분류 (MS-G3D 모델 사용) - 정규화 적용
  Future<String> classifyPose(List<PoseLandmark> landmarks, {double imageWidth = 640, double imageHeight = 480}) async {
    if (!_isInitialized) {
      return 'others';
    }

    try {
      _inferenceCount++;

      // 10번째 추론마다 메모리 상태 로깅
      if (_inferenceCount % 10 == 0) {
        _logMemoryUsage('추론 #$_inferenceCount');
      }
      // 포즈 히스토리에 추가
      _addPoseToHistory(landmarks);

      // 성능 최적화: 분류 빈도 제어
      _frameCount++;
      if (_frameCount % classificationInterval != 0) {
        return 'others';
      }

      // 최소 프레임 수 확인 (실시간 성능 최적화)
      int minFrames = 3; // 더 빠른 반응을 위해 최소 프레임 수 감소
      if (_poseHistory.length < minFrames) {
        return 'others';
      }

      // 최근 프레임 사용
      int framesToUse = _poseHistory.length > inputFrames ? inputFrames : _poseHistory.length;
      List<List<PoseLandmark>> recentPoses = _poseHistory.sublist(_poseHistory.length - framesToUse);

      // 5D 입력 모델 처리 (정규화 적용)
      Float32List inputBuffer = _create5DInputNormalized(recentPoses, imageWidth, imageHeight);

      // 입력 데이터 검증
      if (inputBuffer.length != _expectedInputSize) {
        print('❌ 입력 크기 불일치: 예상=${_expectedInputSize}, 실제=${inputBuffer.length}');
        return 'others';
      }

      // NaN 또는 무한값 검사
      bool hasInvalidValue = false;
      for (int i = 0; i < inputBuffer.length; i++) {
        if (inputBuffer[i].isNaN || inputBuffer[i].isInfinite) {
          hasInvalidValue = true;
          break;
        }
      }

      if (hasInvalidValue) {
        print('❌ 입력 데이터에 유효하지 않은 값이 포함됨');
        return 'others';
      }

      // 성능 모니터링 (간소화)
      if (_frameCount % 30 == 0) { // 30프레임마다 한 번씩만 출력
        print('📊 실시간 분류 상태: 프레임=${_poseHistory.length}, 사용=${framesToUse}, 입력크기=${inputBuffer.length}');
        print('📊 입력 데이터 범위: ${inputBuffer.reduce((a, b) => a < b ? a : b)} ~ ${inputBuffer.reduce((a, b) => a > b ? a : b)}');
      }

      // PyTorch Lite 모델 추론 실행
      List<double> output = [];
      try {
        // 디버깅 로그는 30프레임마다만 출력 (성능 최적화)
        if (_frameCount % 30 == 0) {
          print('🔍 입력 데이터 디버깅:');
          print('  - 입력 버퍼 크기: ${inputBuffer.length}');
          print('  - 예상 입력 크기: $_expectedInputSize');
          print('  - 입력 형태: [1, 3, 64, 25, 1]');
          print('  - 입력 데이터 범위: ${inputBuffer.reduce((a, b) => a < b ? a : b)} ~ ${inputBuffer.reduce((a, b) => a > b ? a : b)}');
        }

        // PyTorch Lite는 입력을 Float32List로 받음
        output = await _runPytorchModel(inputBuffer);
        
        if (_frameCount % 30 == 0) {
          print('✅ PyTorch 모델 추론 성공');
          print('🔍 출력 데이터 디버깅:');
          print('  - 출력 버퍼 크기: ${output.length}');
          print('  - 원시 출력값: $output');
          print('  - 출력값 범위: ${output.reduce((a, b) => a < b ? a : b)} ~ ${output.reduce((a, b) => a > b ? a : b)}');
        }

      } catch (e) {
        print('❌ PyTorch 모델 추론 실행 오류: $e');
        print('❌ 입력 버퍼 크기: ${inputBuffer.length}');
        print('❌ 예상 입력 크기: $_expectedInputSize');
        return 'others';
      }

      // 소프트맥스 함수 적용 (원시 로짓을 확률로 변환)
      List<double> probabilities = _applySoftmax(output);

      // 편향 보정: lunges가 너무 높은 확률을 가지는 경우 조정
      if (probabilities[2] > 0.8) { // lunges 인덱스는 2
        print('🔧 편향 보정: lunges 확률이 너무 높음 (${(probabilities[2] * 100).toStringAsFixed(1)}%)');
        // 다른 클래스들의 확률을 상대적으로 높임
        for (int i = 0; i < probabilities.length; i++) {
          if (i != 2) {
            probabilities[i] *= 1.5; // 다른 클래스 확률 증가
          }
        }
        // 재정규화
        double sum = probabilities.reduce((a, b) => a + b);
        for (int i = 0; i < probabilities.length; i++) {
          probabilities[i] /= sum;
        }
        print('🔧 보정 후 lunges 확률: ${(probabilities[2] * 100).toStringAsFixed(1)}%');
      }

      // 가장 높은 확률의 클래스 찾기
      int predictedClass = 0;
      double maxProbability = probabilities[0];

      for (int i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxProbability) {
          maxProbability = probabilities[i];
          predictedClass = i;
        }
      }

      String predictedExercise = exerciseClasses[predictedClass];

      // 매번 실시간 추론 결과 출력 (디버깅용)
      print('🚀 실시간 추론 #$_inferenceCount: $predictedExercise (${(maxProbability * 100).toStringAsFixed(1)}%)');
      
      // 상세한 분류 결과 출력 (30프레임마다만)
      if (_frameCount % 30 == 0) {
        print('🎯 ===== 모델 분류 결과 상세 =====');
        print('🎯 원시 로짓 값:');
        for (int i = 0; i < exerciseClasses.length; i++) {
          String marker = i == predictedClass ? ' ⭐' : '';
          print('  - ${exerciseClasses[i]}: ${output[i].toStringAsFixed(4)}$marker');
        }
        print('🎯 소프트맥스 적용 후 확률:');
        for (int i = 0; i < exerciseClasses.length; i++) {
          String marker = i == predictedClass ? ' ⭐' : '';
          print('  - ${exerciseClasses[i]}: ${(probabilities[i] * 100).toStringAsFixed(2)}%$marker');
        }
        print('🎯 최종 분류 결과: $predictedExercise (${(maxProbability * 100).toStringAsFixed(2)}%)');
        print('🎯 side_lateral_raise 확률: ${(probabilities[3] * 100).toStringAsFixed(2)}%');
        print('🎯 ================================');
      }

      // 신뢰도가 낮으면 'others' 반환 (임계값 조정)
      double confidenceThreshold = 0.25; // 임계값을 높여서 더 확실한 경우만 분류
      if (maxProbability < confidenceThreshold) {
        print('📊 신뢰도 낮음: ${(maxProbability * 100).toStringAsFixed(1)}% < ${(confidenceThreshold * 100).toStringAsFixed(1)}% -> others 반환');
        return 'others';
      }

      return predictedExercise;

    } catch (e) {
      print('❌ 포즈 분류 오류: $e');
      return 'others';
    }
  }

  // 신뢰도와 함께 분류 결과 반환 (정규화된 좌표 사용 - 실시간용)
  Future<Object> classifyPoseWithConfidenceNormalized(List<PoseLandmark> landmarks, double imageWidth, double imageHeight) async {
    if (!_isInitialized) {
      return {'exercise': 'others', 'confidence': 0.0};
    }

    try {
      _inferenceCount++;

      // 10번째 추론마다 메모리 상태 로깅
      if (_inferenceCount % 10 == 0) {
        _logMemoryUsage('추론(정규화) #$_inferenceCount');
      }

      // 디버깅 로그는 30프레임마다만 출력 (성능 최적화)
      if (_frameCount % 30 == 0) {
        print('📥 [정규화 입력] MediaPipe 랜드마크 수: ${landmarks.length}개');
        print('📥 [정규화 입력] 이미지 크기: ${imageWidth}x${imageHeight}');
      }

      // 포즈 히스토리에 추가
      _addPoseToHistory(landmarks);

      if (_frameCount % 30 == 0) {
        print('📚 [정규화 히스토리] 포즈 히스토리 크기: ${_poseHistory.length}/${maxHistoryLength}');
        print('📚 [정규화 히스토리] 분류에 필요한 최소 프레임: 3개');
      }

      // 성능 최적화: 분류 빈도 제어
      _frameCount++;
      if (_frameCount % classificationInterval != 0) {
        return {'exercise': 'others', 'confidence': 0.0};
      }

      // 최소 프레임 수 확인 (실시간 성능 최적화)
      int minFrames = 3; // 더 빠른 반응을 위해 최소 프레임 수 감소
      if (_poseHistory.length < minFrames) {
        if (_frameCount % 30 == 0) {
          print('⚠️ [정규화 히스토리] 프레임 부족: ${_poseHistory.length}/$minFrames (최소 필요)');
        }
        return {'exercise': 'others', 'confidence': 0.0};
      }

      // 최근 프레임 사용
      int framesToUse = _poseHistory.length > inputFrames ? inputFrames : _poseHistory.length;
      List<List<PoseLandmark>> recentPoses = _poseHistory.sublist(_poseHistory.length - framesToUse);

      if (_frameCount % 30 == 0) {
        print('📊 [정규화 히스토리] 사용할 프레임 수: $framesToUse개');
        print('📊 [정규화 히스토리] 프레임 범위: [${_poseHistory.length - framesToUse}:${_poseHistory.length}]');
      }

      // 5D 입력 모델 처리 (정규화된 좌표 사용)
      Float32List inputBuffer = _create5DInputNormalized(recentPoses, imageWidth, imageHeight);

      // 입력 데이터 검증
      if (inputBuffer.length != _expectedInputSize) {
        print('❌ [정규화] 입력 크기 불일치: 예상=${_expectedInputSize}, 실제=${inputBuffer.length}');
        return {'exercise': 'others', 'confidence': 0.0};
      }

      // NaN 또는 무한값 검사
      bool hasInvalidValue = false;
      for (int i = 0; i < inputBuffer.length; i++) {
        if (inputBuffer[i].isNaN || inputBuffer[i].isInfinite) {
          hasInvalidValue = true;
          break;
        }
      }

      if (hasInvalidValue) {
        print('❌ [정규화] 입력 데이터에 유효하지 않은 값이 포함됨');
        return {'exercise': 'others', 'confidence': 0.0};
      }

      // 성능 모니터링 (간소화)
      if (_frameCount % 30 == 0) { // 30프레임마다 한 번씩만 출력
        print('📊 [정규화] 실시간 분류 상태: 프레임=${_poseHistory.length}, 사용=${framesToUse}, 입력크기=${inputBuffer.length}');
        print('📊 [정규화] 입력 데이터 범위: ${inputBuffer.reduce((a, b) => a < b ? a : b)} ~ ${inputBuffer.reduce((a, b) => a > b ? a : b)}');
      }

      // PyTorch Lite 모델 추론 실행
      List<double> output = [];
      try {
        // 입력 데이터 디버깅 (정규화 함수용) - 30프레임마다만
        if (_frameCount % 30 == 0) {
          print('🔍 [정규화 CONFIDENCE] 입력 데이터 디버깅:');
          print('  - 입력 버퍼 크기: ${inputBuffer.length}');
          print('  - 예상 입력 크기: $_expectedInputSize');
          print('  - 입력 형태: [1, 3, 64, 25, 1]');
          print('  - 입력 데이터 범위: ${inputBuffer.reduce((a, b) => a < b ? a : b)} ~ ${inputBuffer.reduce((a, b) => a > b ? a : b)}');
        }

        // PyTorch Lite는 입력을 Float32List로 받음
        output = await _runPytorchModel(inputBuffer);
        
        if (_frameCount % 30 == 0) {
          print('✅ [정규화 CONFIDENCE] PyTorch 모델 추론 성공');
          print('🔍 [정규화 CONFIDENCE] 출력 데이터 디버깅:');
          print('  - 출력 버퍼 크기: ${output.length}');
          print('  - 원시 출력값: $output');
          print('  - 출력값 범위: ${output.reduce((a, b) => a < b ? a : b)} ~ ${output.reduce((a, b) => a > b ? a : b)}');
        }

      } catch (e) {
        print('❌ [정규화 CONFIDENCE] PyTorch 모델 추론 실행 오류: $e');
        print('❌ [정규화 CONFIDENCE] 입력 버퍼 크기: ${inputBuffer.length}');
        print('❌ [정규화 CONFIDENCE] 예상 입력 크기: $_expectedInputSize');
        return {'exercise': 'others', 'confidence': 0.0};
      }

      // 소프트맥스 함수 적용 (원시 로짓을 확률로 변환)
      List<double> probabilities = _applySoftmax(output);

      // 가장 높은 확률의 클래스 찾기
      int predictedClass = 0;
      double maxProbability = probabilities[0];

      for (int i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxProbability) {
          maxProbability = probabilities[i];
          predictedClass = i;
        }
      }

      String predictedExercise = exerciseClasses[predictedClass];

      // 상세한 분류 결과 출력 (정규화 함수용) - 30프레임마다만
      if (_frameCount % 30 == 0) {
        print('🎯 ===== 정규화 실시간 분류 결과 =====');
        print('🎯 원시 로짓 값:');
        for (int i = 0; i < exerciseClasses.length; i++) {
          String marker = i == predictedClass ? ' ⭐' : '';
          print('  - ${exerciseClasses[i]}: ${output[i].toStringAsFixed(4)}$marker');
        }
        print('🎯 소프트맥스 적용 후 확률:');
        for (int i = 0; i < exerciseClasses.length; i++) {
          String marker = i == predictedClass ? ' ⭐' : '';
          print('  - ${exerciseClasses[i]}: ${(probabilities[i] * 100).toStringAsFixed(2)}%$marker');
        }
        print('🎯 최종 분류 결과: $predictedExercise (${(maxProbability * 100).toStringAsFixed(2)}%)');
        print('🎯 side_lateral_raise 확률: ${(probabilities[3] * 100).toStringAsFixed(2)}%');
        print('🎯 ================================');
      }

      // 신뢰도가 낮으면 'others' 반환 (더 낮은 임계값 적용)
      if (maxProbability < 0.15) {
        if (_frameCount % 30 == 0) {
          print('📊 [정규화] 신뢰도 낮음: ${(maxProbability * 100).toStringAsFixed(1)}% -> others 반환');
        }
        return {'exercise': 'others', 'confidence': maxProbability};
      }

      return {'exercise': predictedExercise, 'confidence': maxProbability};

    } catch (e) {
      print('❌ [정규화] 포즈 분류 오류: $e');
      return {'exercise': 'others', 'confidence': 0.0};
    }
  }

  // 신뢰도와 함께 분류 결과 반환 (MS-G3D 모델 사용 - 기존 함수)
  Future<Object> classifyPoseWithConfidence(List<PoseLandmark> landmarks) async {
    if (!_isInitialized) {
      return {'exercise': 'others', 'confidence': 0.0};
    }

    try {
      _inferenceCount++;

      // 10번째 추론마다 메모리 상태 로깅
      if (_inferenceCount % 10 == 0) {
        _logMemoryUsage('추론(신뢰도) #$_inferenceCount');
      }

      // 디버깅 로그는 30프레임마다만 출력 (성능 최적화)
      if (_frameCount % 30 == 0) {
        print('📥 [입력] MediaPipe 랜드마크 수: ${landmarks.length}개');
      }

      // 포즈 히스토리에 추가
      _addPoseToHistory(landmarks);

      if (_frameCount % 30 == 0) {
        print('📚 [히스토리] 포즈 히스토리 크기: ${_poseHistory.length}/${maxHistoryLength}');
        print('📚 [히스토리] 분류에 필요한 최소 프레임: 3개');
      }

      // 성능 최적화: 분류 빈도 제어
      _frameCount++;
      if (_frameCount % classificationInterval != 0) {
        return {'exercise': 'others', 'confidence': 0.0};
      }

      // 최소 프레임 수 확인 (실시간 성능 최적화)
      int minFrames = 3; // 더 빠른 반응을 위해 최소 프레임 수 감소
      if (_poseHistory.length < minFrames) {
        if (_frameCount % 30 == 0) {
          print('⚠️ [히스토리] 프레임 부족: ${_poseHistory.length}/$minFrames (최소 필요)');
        }
        return {'exercise': 'others', 'confidence': 0.0};
      }

      // 최근 프레임 사용
      int framesToUse = _poseHistory.length > inputFrames ? inputFrames : _poseHistory.length;
      List<List<PoseLandmark>> recentPoses = _poseHistory.sublist(_poseHistory.length - framesToUse);

      if (_frameCount % 30 == 0) {
        print('📊 [히스토리] 사용할 프레임 수: $framesToUse개');
        print('📊 [히스토리] 프레임 범위: [${_poseHistory.length - framesToUse}:${_poseHistory.length}]');
      }

      // 5D 입력 모델 처리 ([1, 3, 64, 25, 1])
      Float32List inputBuffer = _create5DInput(recentPoses);

      // 입력 데이터 검증
      if (inputBuffer.length != _expectedInputSize) {
        print('❌ 입력 크기 불일치: 예상=${_expectedInputSize}, 실제=${inputBuffer.length}');
        return {'exercise': 'others', 'confidence': 0.0};
      }

      // NaN 또는 무한값 검사
      bool hasInvalidValue = false;
      for (int i = 0; i < inputBuffer.length; i++) {
        if (inputBuffer[i].isNaN || inputBuffer[i].isInfinite) {
          hasInvalidValue = true;
          break;
        }
      }

      if (hasInvalidValue) {
        print('❌ 입력 데이터에 유효하지 않은 값이 포함됨');
        return {'exercise': 'others', 'confidence': 0.0};
      }

      // 성능 모니터링 (간소화)
      if (_frameCount % 30 == 0) { // 30프레임마다 한 번씩만 출력
        print('📊 실시간 분류 상태: 프레임=${_poseHistory.length}, 사용=${framesToUse}, 입력크기=${inputBuffer.length}');
        print('📊 입력 데이터 범위: ${inputBuffer.reduce((a, b) => a < b ? a : b)} ~ ${inputBuffer.reduce((a, b) => a > b ? a : b)}');
      }

      // PyTorch Lite 모델 추론 실행
      List<double> output = [];
      try {
        // 입력 데이터 디버깅 (신뢰도 함수용) - 30프레임마다만
        if (_frameCount % 30 == 0) {
          print('🔍 [CONFIDENCE] 입력 데이터 디버깅:');
          print('  - 입력 버퍼 크기: ${inputBuffer.length}');
          print('  - 예상 입력 크기: $_expectedInputSize');
          print('  - 입력 형태: [1, 3, 64, 25, 1]');
          print('  - 입력 데이터 범위: ${inputBuffer.reduce((a, b) => a < b ? a : b)} ~ ${inputBuffer.reduce((a, b) => a > b ? a : b)}');
        }

        // PyTorch Lite는 입력을 Float32List로 받음
        output = await _runPytorchModel(inputBuffer);
        
        if (_frameCount % 30 == 0) {
          print('✅ [CONFIDENCE] PyTorch 모델 추론 성공');
          print('🔍 [CONFIDENCE] 출력 데이터 디버깅:');
          print('  - 출력 버퍼 크기: ${output.length}');
          print('  - 원시 출력값: $output');
          print('  - 출력값 범위: ${output.reduce((a, b) => a < b ? a : b)} ~ ${output.reduce((a, b) => a > b ? a : b)}');
        }

      } catch (e) {
        print('❌ [CONFIDENCE] PyTorch 모델 추론 실행 오류: $e');
        print('❌ [CONFIDENCE] 입력 버퍼 크기: ${inputBuffer.length}');
        print('❌ [CONFIDENCE] 예상 입력 크기: $_expectedInputSize');
        return {'exercise': 'others', 'confidence': 0.0};
      }

      // 소프트맥스 함수 적용 (원시 로짓을 확률로 변환)
      List<double> probabilities = _applySoftmax(output);

      // 가장 높은 확률의 클래스 찾기
      int predictedClass = 0;
      double maxProbability = probabilities[0];

      for (int i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxProbability) {
          maxProbability = probabilities[i];
          predictedClass = i;
        }
      }

      String predictedExercise = exerciseClasses[predictedClass];

      // 상세한 분류 결과 출력 (신뢰도 함수용) - 30프레임마다만
      if (_frameCount % 30 == 0) {
        print('🎯 ===== 신뢰도 함수 분류 결과 =====');
        print('🎯 원시 로짓 값:');
        for (int i = 0; i < exerciseClasses.length; i++) {
          String marker = i == predictedClass ? ' ⭐' : '';
          print('  - ${exerciseClasses[i]}: ${output[i].toStringAsFixed(4)}$marker');
        }
        print('🎯 소프트맥스 적용 후 확률:');
        for (int i = 0; i < exerciseClasses.length; i++) {
          String marker = i == predictedClass ? ' ⭐' : '';
          print('  - ${exerciseClasses[i]}: ${(probabilities[i] * 100).toStringAsFixed(2)}%$marker');
        }
        print('🎯 최종 분류 결과: $predictedExercise (${(maxProbability * 100).toStringAsFixed(2)}%)');
        print('🎯 side_lateral_raise 확률: ${(probabilities[3] * 100).toStringAsFixed(2)}%');
        print('🎯 ================================');
      }

      // 신뢰도가 낮으면 'others' 반환 (더 낮은 임계값 적용)
      if (maxProbability < 0.15) {
        if (_frameCount % 30 == 0) {
          print('📊 신뢰도 낮음: ${(maxProbability * 100).toStringAsFixed(1)}% -> others 반환');
        }
        return {'exercise': 'others', 'confidence': maxProbability};
      }

      return {'exercise': predictedExercise, 'confidence': maxProbability};

    } catch (e) {
      print('❌ 포즈 분류 오류: $e');
      return {'exercise': 'others', 'confidence': 0.0};
    }
  }

  // 5D 모델용 입력 생성 [1, 3, T, 25, 1] (정규화된 좌표 사용 - 실시간용)
  Float32List _create5DInputNormalized(List<List<PoseLandmark>> recentPoses, double imageWidth, double imageHeight) {
    // 로그는 30프레임마다만 출력 (성능 최적화)
    if (_frameCount % 30 == 0) {
      print('🏗️ [정규화 5D 입력] 입력 텐서 생성 시작 (정규화 적용)');
      print('  - 입력 프레임 수: ${recentPoses.length}개');
      print('  - 목표 프레임 수: $inputFrames개');
      print('  - 이미지 크기: ${imageWidth}x${imageHeight}');
    }

    int paddingCount = 0;

    // 1단계: (T, V, C) 형태로 데이터 구성 (정규화된 좌표 사용)
    List<List<List<double>>> skeletonSeq = [];
    for (int t = 0; t < inputFrames; t++) {
      List<List<double>> frameJoints;

      if (t < recentPoses.length) {
        frameJoints = _mediapipeToNtuNormalized(recentPoses[t], imageWidth, imageHeight);
      } else {
        // 부족한 프레임은 마지막 프레임으로 패딩
        frameJoints = _mediapipeToNtuNormalized(recentPoses.last, imageWidth, imageHeight);
        paddingCount++;
      }

      skeletonSeq.add(frameJoints); // (T, V, C)
    }

    if (_frameCount % 30 == 0) {
      print('  - 패딩된 프레임 수: $paddingCount개');
    }

    // 2단계: (T, V, C) -> (C, T, V) transpose (Python과 동일)
    // Python: skeleton_data = skeleton_seq.transpose(2, 0, 1)
    List<double> flatInput = [];

    // 실제 모델 입력 형태: [batch, channels, time, vertices, persons]
    // batch=1, channels=3, time=T, vertices=25, persons=1

    for (int c = 0; c < numChannels; c++) {       // C (x, y, z)
      for (int t = 0; t < inputFrames; t++) {      // T (time)
        for (int v = 0; v < numKeypoints; v++) {   // V (vertices/joints)
          for (int m = 0; m < numPersons; m++) {   // M (persons)
            // skeletonSeq[t][v][c] -> transposed to [c][t][v]
            double value = skeletonSeq[t][v][c];
            flatInput.add(value);
          }
        }
      }
    }

    // 예상 입력 크기: 1 × 3 × T × 25 × 1
    int expectedSize = numChannels * inputFrames * numKeypoints * numPersons;

    if (flatInput.length != expectedSize) {
      print('⚠️ [정규화 5D 입력] 크기 불일치: 예상=$expectedSize, 실제=${flatInput.length}');
    }

    // 첫 프레임의 첫 5개 관절점 확인 (디버깅) - 30프레임마다만
    if (_frameCount % 30 == 0) {
      print('  [정규화 디버깅] 첫 프레임 샘플:');
      for (int v = 0; v < math.min(5, numKeypoints); v++) {
        print('    관절점$v: x=${skeletonSeq[0][v][0].toStringAsFixed(4)}, '
            'y=${skeletonSeq[0][v][1].toStringAsFixed(4)}, '
            'z=${skeletonSeq[0][v][2].toStringAsFixed(4)}');
      }

      // 입력 데이터 후처리 - 추가 정규화 및 클리핑
      for (int i = 0; i < flatInput.length; i++) {
        // NaN이나 무한값 처리
        if (flatInput[i].isNaN || flatInput[i].isInfinite) {
          flatInput[i] = 0.0;
        }
        // 극값 클리핑 (모델 안정성을 위해)
        flatInput[i] = flatInput[i].clamp(-2.0, 2.0);
      }

      // 통계
      double sum = flatInput.reduce((a, b) => a + b);
      double mean = sum / flatInput.length;
      double min = flatInput.reduce((a, b) => a < b ? a : b);
      double max = flatInput.reduce((a, b) => a > b ? a : b);
      int zeroCount = flatInput.where((v) => v == 0.0).length;
      print('  [정규화 통계] 평균: ${mean.toStringAsFixed(6)}, 범위: ${min.toStringAsFixed(3)}~${max.toStringAsFixed(3)}, 0의 개수: $zeroCount/${flatInput.length}');

      print('✅ [정규화 5D 입력] 입력 텐서 생성 완료: ${flatInput.length} 요소 [1, 3, $inputFrames, 25, 1]');
    }
    
    return Float32List.fromList(flatInput);
  }

  // 5D 모델용 입력 생성 [1, 3, T, 25, 1] (기존 함수 - 정규화 없음)
  Float32List _create5DInput(List<List<PoseLandmark>> recentPoses) {
    // 로그는 30프레임마다만 출력 (성능 최적화)
    if (_frameCount % 30 == 0) {
      print('🏗️ [5D 입력] 입력 텐서 생성 시작 (Python preprocess_skeleton 방식)');
      print('  - 입력 프레임 수: ${recentPoses.length}개');
      print('  - 목표 프레임 수: $inputFrames개');
    }

    int paddingCount = 0;

    // 1단계: (T, V, C) 형태로 데이터 구성 (Python과 동일)
    List<List<List<double>>> skeletonSeq = [];
    for (int t = 0; t < inputFrames; t++) {
      List<List<double>> frameJoints;

      if (t < recentPoses.length) {
        frameJoints = _mediapipeToNtu(recentPoses[t]);
      } else {
        // 부족한 프레임은 마지막 프레임으로 패딩
        frameJoints = _mediapipeToNtu(recentPoses.last);
        paddingCount++;
      }

      skeletonSeq.add(frameJoints); // (T, V, C)
    }

    print('  - 패딩된 프레임 수: $paddingCount개');

    // 2단계: (T, V, C) -> (C, T, V) transpose (Python과 동일)
    // Python: skeleton_data = skeleton_seq.transpose(2, 0, 1)
    List<double> flatInput = [];

    // 실제 모델 입력 형태: [batch, channels, time, vertices, persons]
    // batch=1, channels=3, time=T, vertices=25, persons=1

    for (int c = 0; c < numChannels; c++) {       // C (x, y, z)
      for (int t = 0; t < inputFrames; t++) {      // T (time)
        for (int v = 0; v < numKeypoints; v++) {   // V (vertices/joints)
          for (int m = 0; m < numPersons; m++) {   // M (persons)
            // skeletonSeq[t][v][c] -> transposed to [c][t][v]
            double value = skeletonSeq[t][v][c];
            flatInput.add(value);
          }
        }
      }
    }

    // 예상 입력 크기: 1 × 3 × T × 25 × 1
    int expectedSize = numChannels * inputFrames * numKeypoints * numPersons;

    if (flatInput.length != expectedSize) {
      print('⚠️ [5D 입력] 크기 불일치: 예상=$expectedSize, 실제=${flatInput.length}');
    }

    // 첫 프레임의 첫 5개 관절점 확인 (디버깅)
    print('  [디버깅] 첫 프레임 샘플:');
    for (int v = 0; v < math.min(5, numKeypoints); v++) {
      print('    관절점$v: x=${skeletonSeq[0][v][0].toStringAsFixed(4)}, '
          'y=${skeletonSeq[0][v][1].toStringAsFixed(4)}, '
          'z=${skeletonSeq[0][v][2].toStringAsFixed(4)}');
    }

    // 통계
    double sum = flatInput.reduce((a, b) => a + b);
    double mean = sum / flatInput.length;
    int zeroCount = flatInput.where((v) => v == 0.0).length;
    print('  [통계] 평균: ${mean.toStringAsFixed(6)}, 0의 개수: $zeroCount/${flatInput.length}');

    print('✅ [5D 입력] 입력 텐서 생성 완료: ${flatInput.length} 요소 [1, 3, $inputFrames, 25, 1]');
    return Float32List.fromList(flatInput);
  }

  // PyTorch 모델 실행 헬퍼 메서드 (플랫폼 채널 사용)
  Future<List<double>> _runPytorchModel(Float32List inputBuffer) async {
    try {
      // 추론 시간 측정 시작
      final startTime = DateTime.now();

      // 플랫폼 채널을 통해 네이티브 코드에서 PyTorch 추론 실행
      final List<dynamic> result = await platform.invokeMethod('runInference', {
        'inputData': inputBuffer.toList(),
        'shape': [1, numChannels, inputFrames, numKeypoints, numPersons], // [1, 3, 64, 25, 1]
      });

      // 추론 시간 측정 종료
      final endTime = DateTime.now();
      final inferenceTime = endTime.difference(startTime).inMilliseconds;

      // 결과를 List<double>로 변환
      List<double> output = result.map((e) => (e as num).toDouble()).toList();

      print('✅ 네이티브 PyTorch 추론 성공: 출력 크기 ${output.length}');
      print('⏱️ Dart 측정 추론 시간: ${inferenceTime}ms');

      return output;
    } catch (e) {
      print('❌ PyTorch 모델 실행 중 오류: $e');
      rethrow;
    }
  }

  // 1차원 배열을 5D 배열로 변환 [1, 3, 64, 25, 1] (실제 모델 형태)
  List<List<List<List<List<double>>>>> _reshapeTo5D(Float32List flatInput) {
    List<List<List<List<List<double>>>>> result = [];
    int index = 0;

    for (int b = 0; b < 1; b++) {
      List<List<List<List<double>>>> batch = [];

      for (int c = 0; c < numChannels; c++) {
        List<List<List<double>>> channel = [];

        for (int t = 0; t < inputFrames; t++) {
          List<List<double>> frame = [];

          for (int k = 0; k < numKeypoints; k++) {
            List<double> keypoint = [];

            for (int p = 0; p < numPersons; p++) {
              if (index < flatInput.length) {
                keypoint.add(flatInput[index].toDouble());
                index++;
              } else {
                keypoint.add(0.0);
              }
            }
            frame.add(keypoint);
          }
          channel.add(frame);
        }
        batch.add(channel);
      }
      result.add(batch);
    }

    print('✅ 입력 데이터를 5D 형태로 변환: [1, 3, 64, 25, 1]');
    return result;
  }

  // NTU 관절점을 외부에서 접근할 수 있도록 제공 (정규화 없음 - deprecated)
  List<List<double>> getNtuJoints(List<PoseLandmark> landmarks) {
    return _mediapipeToNtu(landmarks);
  }

  // NTU 관절점을 정규화하여 반환 (비디오용)
  List<List<double>> getNtuJointsNormalized(List<PoseLandmark> landmarks, double imageWidth, double imageHeight) {
    return _mediapipeToNtuNormalized(landmarks, imageWidth, imageHeight);
  }

  // 비디오 분류용 함수 - 정규화된 버전
  Future<Map<String, dynamic>> classifyVideoSequenceNormalized(
      List<List<PoseLandmark>> landmarkSequence,
      double imageWidth,
      double imageHeight
      ) async {
    if (!_isInitialized) {
      return {'exercise': 'others', 'confidence': 0.0};
    }

    if (landmarkSequence.isEmpty) {
      print('❌ [비디오 분류] 입력 시퀀스가 비어있음');
      return {'exercise': 'others', 'confidence': 0.0};
    }

    print('');
    print('🎬 [비디오 분류] ===== 전체 시퀀스 분류 시작 =====');
    print('📊 [비디오 분류] 입력 프레임 수: ${landmarkSequence.length}개');

    try {
      // Python과 동일: 전체 시퀀스를 5D 입력으로 변환 (정규화 적용)
      Float32List inputBuffer = _create5DInputFromSequenceNormalized(landmarkSequence, imageWidth, imageHeight);

      // 입력 데이터 검증
      if (inputBuffer.length != _expectedInputSize) {
        print('❌ [비디오 분류] 입력 크기 불일치: 예상=${_expectedInputSize}, 실제=${inputBuffer.length}');
        return {'exercise': 'others', 'confidence': 0.0};
      }

      // 모델 추론
      List<double> output = await _runPytorchModel(inputBuffer);
      print('✅ [비디오 분류] PyTorch 모델 추론 성공');

      // 소프트맥스 적용
      List<double> probabilities = _applySoftmax(output);

      // 최고 확률 클래스 찾기
      int predictedClass = 0;
      double maxProbability = probabilities[0];

      for (int i = 0; i < probabilities.length; i++) {
        if (probabilities[i] > maxProbability) {
          maxProbability = probabilities[i];
          predictedClass = i;
        }
      }

      String predictedExercise = exerciseClasses[predictedClass];

      // 결과 출력
      print('🎯 [비디오 분류] ===== 분류 결과 =====');
      print('🎯 원시 로짓 값:');
      for (int i = 0; i < exerciseClasses.length; i++) {
        print('  - ${exerciseClasses[i]}: ${output[i].toStringAsFixed(4)}');
      }
      print('🎯 소프트맥스 적용 후 확률:');
      for (int i = 0; i < exerciseClasses.length; i++) {
        print('  - ${exerciseClasses[i]}: ${(probabilities[i] * 100).toStringAsFixed(2)}%');
      }
      print('🎯 최종 분류 결과: $predictedExercise (${(maxProbability * 100).toStringAsFixed(2)}%)');
      print('🎯 ===============================');
      print('');

      return {
        'exercise': predictedExercise,
        'confidence': maxProbability,
        'class_index': predictedClass,
        'all_probabilities': {
          for (int i = 0; i < exerciseClasses.length; i++)
            exerciseClasses[i]: probabilities[i]
        }
      };

    } catch (e) {
      print('❌ [비디오 분류] 오류: $e');
      return {'exercise': 'others', 'confidence': 0.0};
    }
  }

  // 비디오 시퀀스 전용 5D 입력 생성 - 정규화 버전
  Float32List _create5DInputFromSequenceNormalized(
      List<List<PoseLandmark>> sequence,
      double imageWidth,
      double imageHeight
      ) {
    print('🏗️ [비디오 입력] 시퀀스 전용 5D 입력 생성');
    print('  - 시퀀스 길이: ${sequence.length}개 프레임');
    print('  - 목표 프레임 수: $inputFrames개');

    int paddingCount = 0;

    // 1단계: (T, V, C) 형태로 데이터 구성
    List<List<List<double>>> skeletonSeq = [];
    for (int t = 0; t < inputFrames; t++) {
      List<List<double>> frameJoints;

      if (t < sequence.length) {
        frameJoints = _mediapipeToNtuNormalized(sequence[t], imageWidth, imageHeight);
      } else {
        // 부족한 프레임은 마지막 프레임으로 패딩
        if (sequence.isNotEmpty) {
          frameJoints = _mediapipeToNtuNormalized(sequence.last, imageWidth, imageHeight);
          paddingCount++;
        } else {
          // 빈 시퀀스인 경우 0으로 채움
          frameJoints = List.generate(25, (_) => [0.0, 0.0, 0.0]);
        }
      }

      skeletonSeq.add(frameJoints);
    }

    print('  - 실제 프레임 수: ${sequence.length}개');
    print('  - 패딩된 프레임 수: $paddingCount개');

    // 첫 프레임과 마지막 프레임 검증
    if (skeletonSeq.isNotEmpty) {
      print('  [검증] 첫 프레임 첫 3개 관절점:');
      for (int v = 0; v < math.min(3, numKeypoints); v++) {
        print('    J$v: x=${skeletonSeq[0][v][0].toStringAsFixed(4)}, '
            'y=${skeletonSeq[0][v][1].toStringAsFixed(4)}, '
            'z=${skeletonSeq[0][v][2].toStringAsFixed(4)}');
      }

      // 좌표 범위 확인
      double minX = double.infinity, maxX = double.negativeInfinity;
      double minY = double.infinity, maxY = double.negativeInfinity;
      double minZ = double.infinity, maxZ = double.negativeInfinity;

      for (var frame in skeletonSeq) {
        for (var joint in frame) {
          if (joint[0] < minX) minX = joint[0];
          if (joint[0] > maxX) maxX = joint[0];
          if (joint[1] < minY) minY = joint[1];
          if (joint[1] > maxY) maxY = joint[1];
          if (joint[2] < minZ) minZ = joint[2];
          if (joint[2] > maxZ) maxZ = joint[2];
        }
      }

      print('  [좌표 범위]');
      print('    X: $minX ~ $maxX');
      print('    Y: $minY ~ $maxY');
      print('    Z: $minZ ~ $maxZ');
    }

    // 2단계: (T, V, C) -> (C, T, V, M) transpose
    List<double> flatInput = [];

    for (int c = 0; c < numChannels; c++) {
      for (int t = 0; t < inputFrames; t++) {
        for (int v = 0; v < numKeypoints; v++) {
          for (int m = 0; m < numPersons; m++) {
            double value = skeletonSeq[t][v][c];
            flatInput.add(value);
          }
        }
      }
    }

    // 입력 텐서 통계
    double sum = flatInput.reduce((a, b) => a + b);
    double mean = sum / flatInput.length;
    double min = flatInput.reduce((a, b) => a < b ? a : b);
    double max = flatInput.reduce((a, b) => a > b ? a : b);
    int zeroCount = flatInput.where((v) => v == 0.0).length;

    print('  [입력 텐서 통계]');
    print('    크기: ${flatInput.length} 요소');
    print('    평균: ${mean.toStringAsFixed(6)}');
    print('    범위: ${min.toStringAsFixed(6)} ~ ${max.toStringAsFixed(6)}');
    print('    0의 개수: $zeroCount/${flatInput.length} (${(zeroCount / flatInput.length * 100).toStringAsFixed(1)}%)');

    // 첫 10개 값 샘플
    print('  [첫 10개 값] ${flatInput.take(10).map((v) => v.toStringAsFixed(4)).join(", ")}');

    print('✅ [비디오 입력] 입력 텐서 생성 완료: [1, 3, $inputFrames, 25, 1]');
    return Float32List.fromList(flatInput);
  }

  // 소프트맥스 함수 적용 (원시 로짓을 확률로 변환)
  List<double> _applySoftmax(List<double> logits) {
    // 수치 안정성을 위해 최대값을 빼기
    double maxLogit = logits.reduce((a, b) => a > b ? a : b);
    List<double> shiftedLogits = logits.map((x) => x - maxLogit).toList();

    // 지수 함수 적용
    List<double> exponentials = shiftedLogits.map((x) => math.exp(x)).toList();

    // 합계 계산
    double sum = exponentials.reduce((a, b) => a + b);

    // 정규화 (확률로 변환)
    List<double> probabilities = exponentials.map((x) => x / sum).toList();

    return probabilities;
  }

  // 시퀀스 기반 분류 (호환성을 위해 유지)
  Future<String> classifyPoseSequence(List<List<PoseLandmark>> landmarkSequence) async {
    if (landmarkSequence.isEmpty) return 'others';
    return await classifyPose(landmarkSequence.last);
  }
}