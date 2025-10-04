import 'dart:typed_data';
import 'dart:math' as math;
import 'package:flutter/services.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import '../utils/ntu_converter.dart';

/// TensorFlow Lite 모델을 사용한 운동 동작 분류 서비스 (플랫폼 채널)
class PoseClassifier {
  static const platform = MethodChannel('com.example.aifit/tflite');
  bool _isInitialized = false;

  // 모델 입력/출력 shape (ms_g3d TFLite: [1, 100, 25, 1, 3])
  static const List<int> inputShape = [1, 100, 25, 1, 3];
  static const List<int> outputShape = [1, 6];

  // 운동 클래스
  static const List<String> exerciseClasses = [
    'benchpress',
    'deadlift',
    'lunges',
    'side_lateral_raise',
    'squat',
    'others',
  ];

  // 포즈 히스토리 (실시간 인식용, NTU 포맷)
  final List<List<List<double>>> _ntuHistory = [];
  static const int maxHistoryLength = 100;

  bool get isInitialized => _isInitialized;

  /// TensorFlow Lite 모델 초기화
  Future<void> initialize() async {
    try {
      print('🚀 TFLite 모델 로딩 시작...');

      // 네이티브 코드에서 모델 로드
      final String result = await platform.invokeMethod('loadModel', {
        'modelPath': 'flutter_assets/assets/models/tflite_simplified_float16_100frame.tflite',
      });

      print('✅ $result');
      print('📊 입력 shape: $inputShape');
      print('📊 출력 shape: $outputShape');
      print('📊 클래스 수: ${exerciseClasses.length}');

      _isInitialized = true;
    } catch (e) {
      print('❌ 모델 로드 실패: $e');
      throw Exception('TFLite 모델 로드 실패: $e');
    }
  }

  /// 실시간 프레임을 처리하고 NTU 포맷으로 변환하여 히스토리에 추가
  void processPoseForRealtime(
    List<PoseLandmark> landmarks,
    double imageWidth,
    double imageHeight,
  ) {
    final ntuJoints = NTUConverter.mediapipeToNTU(
      landmarks,
      imageWidth,
      imageHeight,
    );
    _ntuHistory.add(ntuJoints);
    if (_ntuHistory.length > maxHistoryLength) {
      _ntuHistory.removeAt(0);
    }
  }

  /// 실시간 프레임을 처리하고 NTU 포맷으로 변환하여 히스토리에 추가 (좌우 반전 지원)
  void processPoseForRealtimeWithFlip(
    List<PoseLandmark> landmarks,
    double imageWidth,
    double imageHeight,
    bool flipHorizontally,
  ) {
    final ntuJoints = NTUConverter.mediapipeToNTU(
      landmarks,
      imageWidth,
      imageHeight,
      flipHorizontally: flipHorizontally,
    );
    _ntuHistory.add(ntuJoints);
    if (_ntuHistory.length > maxHistoryLength) {
      _ntuHistory.removeAt(0);
    }
  }

  /// 포즈 히스토리 초기화
  void clearHistory() {
    _ntuHistory.clear();
  }

  /// 실시간 포즈 분류 (히스토리 사용)
  Future<ClassificationResult> classifyRealtime() async {
    if (!_isInitialized) {
      throw Exception('모델이 초기화되지 않았습니다');
    }

    // 최소 프레임 수 확인 (더 빠른 반응을 위해 10프레임으로 감소)
    if (_ntuHistory.length < 10) {
      return ClassificationResult(
        exercise: 'others',
        confidence: 0.0,
        classIndex: 5,
        allProbabilities: {},
      );
    }

    // 히스토리를 100프레임으로 조정 (동영상 분류와 동일)
    List<List<List<double>>> adjustedSequence;
    final int currentLength = _ntuHistory.length;
    
    if (currentLength >= maxHistoryLength) {
      // 100프레임 이상이면 최근 100프레임 사용
      adjustedSequence = _ntuHistory.sublist(_ntuHistory.length - maxHistoryLength);
    } else {
      // 100프레임 미만이면 현재까지의 시퀀스를 반복하여 패딩 (동영상 분석과 동일한 방식)
      adjustedSequence = List.from(_ntuHistory);
      final padCount = maxHistoryLength - currentLength;
      final source = List.from(_ntuHistory);
      if (source.isNotEmpty) {
        for (int i = 0; i < padCount; i++) {
          adjustedSequence.add(source[i % source.length]);
        }
      }
    }

    // 분류 실행
    return await classify(adjustedSequence);
  }

  /// 포즈 시퀀스 분류 (NTU 포맷 입력)
  Future<ClassificationResult> classify(
    List<List<List<double>>> ntuSequence,
  ) async {
    if (!_isInitialized) {
      throw Exception('모델이 초기화되지 않았습니다');
    }

    try {
      // 1. (T,V,C) → (C,T,V,M)
      final Float32List chw = NTUConverter.createModelInput(ntuSequence);
      
      // 2. 앱 전달은 (T,V,M,C) 순서로 고정 전달 (네이티브에서 실제 텐서 순서에 맞게 재배열)
      const int C = 3, T = 100, V = 25, M = 1;
      final Float32List input = Float32List(T * V * M * C);
      int idx = 0;
      for (int t = 0; t < T; t++) {
        for (int v = 0; v < V; v++) {
          for (int m = 0; m < M; m++) {
            for (int c = 0; c < C; c++) {
              final src = c * (T * V * M) + t * (V * M) + v * M + m;
              input[idx++] = chw[src];
            }
          }
        }
      }

      // 3. 입력 데이터 검증
      if (input.any((v) => v.isNaN || v.isInfinite)) {
        print('⚠️ 입력 데이터에 유효하지 않은 값 포함');
        return ClassificationResult(
          exercise: 'others',
          confidence: 0.0,
          classIndex: 5,
          allProbabilities: {},
        );
      }

      // 4. 네이티브 코드에서 모델 추론 실행 (실제 입력 텐서 shape을 조회해 맞게 재배열)
      final List<dynamic> outputDynamic = await platform.invokeMethod('runInference', {
        'inputData': input.toList(),
        'shape': inputShape,
      });

      // 5. 출력을 List<double>로 변환
      List<double> output = outputDynamic.map((e) => (e as num).toDouble()).toList();

      // 6. 소프트맥스 적용 (수치 안정성)
      List<double> probabilities = _applySoftmax(output);

      // 7. 최고 확률 클래스 찾기
      int maxIndex = 0;
      double maxProb = probabilities[0];
      for (int i = 1; i < probabilities.length; i++) {
        if (probabilities[i] > maxProb) {
          maxProb = probabilities[i];
          maxIndex = i;
        }
      }

      // 디버그: 상위 확률 로깅
      try {
        final debug = [for (int i = 0; i < exerciseClasses.length; i++)
          '${exerciseClasses[i]}=${(probabilities[i] * 100).toStringAsFixed(1)}%'];
        print('🔎 분류 확률: ${debug.join(', ')}');
      } catch (_) {}

      return ClassificationResult(
        exercise: exerciseClasses[maxIndex],
        confidence: maxProb,
        classIndex: maxIndex,
        allProbabilities: {
          for (int i = 0; i < exerciseClasses.length; i++)
            exerciseClasses[i]: probabilities[i]
        },
      );
    } catch (e) {
      print('❌ 분류 오류: $e');
      return ClassificationResult(
        exercise: 'others',
        confidence: 0.0,
        classIndex: 5,
        allProbabilities: {},
      );
    }
  }

  /// 소프트맥스 함수
  List<double> _applySoftmax(List<double> logits) {
    // 수치 안정성을 위해 최대값 빼기
    double maxLogit = logits.reduce((a, b) => a > b ? a : b);
    List<double> shiftedLogits = logits.map((x) => x - maxLogit).toList();

    // 지수 함수 적용
    List<double> exponentials = shiftedLogits.map((x) => math.exp(x)).toList();

    // 합계 계산
    double sum = exponentials.reduce((a, b) => a + b);

    // 정규화
    return exponentials.map((x) => x / sum).toList();
  }

  /// 리소스 해제
  Future<void> dispose() async {
    try {
      await platform.invokeMethod('closeModel');
      _isInitialized = false;
      _ntuHistory.clear();
    } catch (e) {
      print('⚠️ 모델 종료 오류: $e');
    }
  }
}


/// 분류 결과 모델
class ClassificationResult {
  final String exercise;
  final double confidence;
  final int classIndex;
  final Map<String, double> allProbabilities;

  ClassificationResult({
    required this.exercise,
    required this.confidence,
    required this.classIndex,
    required this.allProbabilities,
  });
}