import 'dart:typed_data';
import 'dart:math' as math;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

class PoseClassifier {
  late Interpreter _interpreter;
  bool _isInitialized = false;
  
  // 운동 클래스 정의 (MS-G3D 모델에 맞춤)
  static const List<String> exerciseClasses = [
    'benchpress',    // 0
    'deadlift',      // 1
    'lunges',        // 2
    'side_lateral_raise', // 3
    'squat',         // 4
    'others'         // 5
  ];
  
  // 4D 모델 입력 형태 (4D 입력: [1, 64, 25, 3])
  static const int inputFrames = 64; // 프레임 수
  static const int numKeypoints = 25; // NTU RGB+D 데이터셋의 관절점 수
  static const int numChannels = 3; // x, y, confidence
  
  // 포즈 히스토리 저장
  List<List<PoseLandmark>> _poseHistory = [];
  static const int maxHistoryLength = 64; // 모델 입력 프레임 수와 동일하게 설정
  
  // 분류 빈도 제어 (실시간 성능 최적화)
  int _frameCount = 0;
  static const int classificationInterval = 3; // 3프레임마다 분류 수행 (1024×1024 입력 최적화)
  
  // 4D 모델은 4D 입력을 지원
  static const bool _isInput4D = true;
  
  // 실제 모델이 기대하는 입력 크기
  int _expectedInputSize = 4800; // 기본값
  
  // 로그 출력 제어
  bool _hasLoggedNtuConversion = false;
  bool _hasLoggedNtuStart = false;
  
  Future<void> initialize() async {
    try {
      print('🚀 PoseClassifier 초기화 시작...');
      
        // TFLite 모델 로드
        _interpreter = await Interpreter.fromAsset('assets/models/ctvm_4d_model_fixed.tflite');
      
      // 모델 정보 출력
      print('✅ TFLite 모델 로드 성공!');
      var inputDetails = _interpreter.getInputTensors();
      var outputDetails = _interpreter.getOutputTensors();
      
      print('📊 모델 입력 정보:');
      for (var input in inputDetails) {
        print('  - 입력 텐서: ${input.name}, 형태: ${input.shape}');
        print('  - 데이터 타입: ${input.type}');
        print('  - 총 요소 수: ${input.shape.reduce((a, b) => a * b)}');
        print('  - 4D 모델 입력 형태: ${input.shape}');
        
        // 실제 모델이 기대하는 입력 크기 저장
        _expectedInputSize = input.shape.reduce((a, b) => a * b);
        print('  - 예상 입력 크기: $_expectedInputSize');
      }
      
      print('📊 모델 출력 정보:');
      for (var output in outputDetails) {
        print('  - 출력 텐서: ${output.name}, 형태: ${output.shape}');
        print('  - 데이터 타입: ${output.type}');
        print('  - 총 요소 수: ${output.shape.reduce((a, b) => a * b)}');
      }
      
      _isInitialized = true;
      print('🎉 PoseClassifier 초기화 완료!');
      
    } catch (e) {
      print('❌ PoseClassifier 초기화 실패: $e');
      throw Exception('모델 로드 실패: $e');
    }
  }
  
  void dispose() {
    if (_isInitialized) {
      _interpreter.close();
      _isInitialized = false;
    }
  }
  
  // 모델 상태 확인
  bool get isInitialized => _isInitialized;
  
  // MediaPipe 포즈를 NTU RGB+D 25개 관절점으로 변환 (Python 코드 기반)
  List<List<double>> _mediapipeToNtu(List<PoseLandmark> landmarks) {
    List<List<double>> joints = List.generate(25, (_) => List.filled(3, 0.0));
    
    // 안전한 인덱스 접근을 위한 검증
    if (landmarks.length < 33) {
      print('❌ 충분한 랜드마크가 없습니다: ${landmarks.length}/33');
      return joints;
    }
    
    // NTU 관절점 변환 로그는 1번만 출력
    if (!_hasLoggedNtuStart) {
      print('🔄 NTU 25개 관절점 변환 시작 (입력: ${landmarks.length}개 랜드마크)');
      _hasLoggedNtuStart = true;
    }
    
    // NTU RGB+D 25개 관절점 매핑 (Python 코드와 일치)
    
    // 0: spine_base (hip center) - 양쪽 힙의 중점
    joints[0] = [
      (landmarks[23].x + landmarks[24].x) / 2,
      (landmarks[23].y + landmarks[24].y) / 2,
      (landmarks[23].z + landmarks[24].z) / 2
    ];
    
    // 1: spine_mid - hip center와 shoulder center의 중점
    List<double> hipCenter = joints[0];
    List<double> shoulderCenter = [
      (landmarks[11].x + landmarks[12].x) / 2,
      (landmarks[11].y + landmarks[12].y) / 2,
      (landmarks[11].z + landmarks[12].z) / 2
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
    joints[3] = [landmarks[0].x, landmarks[0].y, landmarks[0].z];
    
    // 4-7: left arm
    joints[4] = [landmarks[11].x, landmarks[11].y, landmarks[11].z]; // left_shoulder
    joints[5] = [landmarks[13].x, landmarks[13].y, landmarks[13].z]; // left_elbow
    joints[6] = [landmarks[15].x, landmarks[15].y, landmarks[15].z]; // left_wrist
    joints[7] = [landmarks[15].x, landmarks[15].y, landmarks[15].z]; // left_hand (wrist와 동일)
    
    // 8-11: right arm
    joints[8] = [landmarks[12].x, landmarks[12].y, landmarks[12].z]; // right_shoulder
    joints[9] = [landmarks[14].x, landmarks[14].y, landmarks[14].z]; // right_elbow
    joints[10] = [landmarks[16].x, landmarks[16].y, landmarks[16].z]; // right_wrist
    joints[11] = [landmarks[16].x, landmarks[16].y, landmarks[16].z]; // right_hand (wrist와 동일)
    
    // 12-15: left leg
    joints[12] = [landmarks[23].x, landmarks[23].y, landmarks[23].z]; // left_hip
    joints[13] = [landmarks[25].x, landmarks[25].y, landmarks[25].z]; // left_knee
    joints[14] = [landmarks[27].x, landmarks[27].y, landmarks[27].z]; // left_ankle
    joints[15] = [landmarks[31].x, landmarks[31].y, landmarks[31].z]; // left_foot
    
    // 16-19: right leg
    joints[16] = [landmarks[24].x, landmarks[24].y, landmarks[24].z]; // right_hip
    joints[17] = [landmarks[26].x, landmarks[26].y, landmarks[26].z]; // right_knee
    joints[18] = [landmarks[28].x, landmarks[28].y, landmarks[28].z]; // right_ankle
    joints[19] = [landmarks[32].x, landmarks[32].y, landmarks[32].z]; // right_foot
    
    // 20-24: additional points
    joints[20] = shoulderCenter; // spine_shoulder
    joints[21] = [landmarks[17].x, landmarks[17].y, landmarks[17].z]; // left_hand_tip
    joints[22] = [landmarks[21].x, landmarks[21].y, landmarks[21].z]; // left_thumb
    joints[23] = [landmarks[18].x, landmarks[18].y, landmarks[18].z]; // right_hand_tip
    joints[24] = [landmarks[22].x, landmarks[22].y, landmarks[22].z]; // right_thumb
    
    // 완료 로그는 1번만 출력
    if (!_hasLoggedNtuConversion) {
      print('✅ NTU 25개 관절점 변환 완료');
      _hasLoggedNtuConversion = true;
    }
    return joints;
  }
  
  // MS-G3D 모델 입력 형태로 데이터 전처리 (실제 모델에 맞게 수정)
  List<List<List<double>>> _preprocessPoseData(List<PoseLandmark> landmarks) {
    List<List<double>> joints = _mediapipeToNtu(landmarks);
    List<List<List<double>>> processedData = [];
    
    // 좌표 정규화를 위한 중심점 계산 (hip center 기준)
    double centerX = joints[0][0];
    double centerY = joints[0][1];
    
    // 변환된 모델용 간소화된 디버깅
    if (!_hasLoggedNtuStart) {
      print('🔍 4D 모델용 좌표 정규화 시작');
      print('  - 중심점: ($centerX, $centerY)');
      print('  - 원본 MediaPipe 랜드마크 수: ${landmarks.length}');
      print('  - 변환된 NTU 관절점 수: ${joints.length}');
    }
    
    // 4D 모델에 맞는 좌표 정규화
    double scaleFactor = 5.0; // 4D 모델에 최적화된 스케일 팩터
    
    // 실제 모델 입력 형태에 맞게 데이터 구성
    // 각 채널별로 데이터 구성: 1차원 데이터 사용
    for (int channel = 0; channel < numChannels; channel++) {
      List<List<double>> channelData = [];
      
      for (int i = 0; i < numKeypoints; i++) {
        List<double> keypointData = [];
        
        if (i < joints.length) {
          // 좌표 정규화 (중심점 기준으로 상대 좌표 계산)
          double normalizedX = (joints[i][0] - centerX) * scaleFactor;
          double normalizedY = (joints[i][1] - centerY) * scaleFactor;
          
          // 값 범위 제한 (4D 모델에 맞게 조정)
          normalizedX = normalizedX.clamp(-5.0, 5.0);
          normalizedY = normalizedY.clamp(-5.0, 5.0);
          
          switch (channel) {
            case 0: // x 좌표 채널
              keypointData.add(normalizedX);
              break;
            case 1: // y 좌표 채널  
              keypointData.add(normalizedY);
              break;
            case 2: // confidence 채널
              keypointData.add(joints[i][2].clamp(0.0, 1.0));
              break;
          }
        } else {
          keypointData.add(0.0);
        }
        
        channelData.add(keypointData);
      }
      
      processedData.add(channelData);
    }
    
    return processedData;
  }
  
  
  // 포즈 히스토리에 추가
  void _addPoseToHistory(List<PoseLandmark> landmarks) {
    _poseHistory.add(landmarks);
    if (_poseHistory.length > maxHistoryLength) {
      _poseHistory.removeAt(0);
    }
  }
  
  // 실시간 포즈 분류 (MS-G3D 모델 사용)
  String classifyPose(List<PoseLandmark> landmarks) {
    if (!_isInitialized) {
      return 'others';
    }
    
    try {
      // 포즈 히스토리에 추가
      _addPoseToHistory(landmarks);
      
      // 성능 최적화: 분류 빈도 제어
      _frameCount++;
      if (_frameCount % classificationInterval != 0) {
        return 'others';
      }
      
      // 최소 프레임 수 확인 (실시간 성능 최적화)
      int minFrames = 8; // 1024×1024 입력은 적당한 프레임 수로 충분
      if (_poseHistory.length < minFrames) {
        return 'others';
      }
      
      // 최근 프레임 사용
      int framesToUse = _poseHistory.length > inputFrames ? inputFrames : _poseHistory.length;
      List<List<PoseLandmark>> recentPoses = _poseHistory.sublist(_poseHistory.length - framesToUse);
      
      // 4D 입력 모델 처리 ([1, 64, 25, 3])
      Float32List inputBuffer = _create4DInput(recentPoses);
      
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
      
      // 출력 버퍼 준비
      List<List<double>> output = List.generate(1, (_) => List.filled(exerciseClasses.length, 0.0));
      
      // 모델 추론 실행
      try {
        // 1차원 배열을 4D 배열로 변환하여 시도
        List<List<List<List<double>>>> reshapedInput = _reshapeTo4D(inputBuffer);
        _interpreter.run(reshapedInput, output);
        print('✅ 모델 추론 성공 (4D 배열)');
      } catch (e1) {
        print('⚠️ 4D 배열 실행 실패, 1D 배열로 재시도: $e1');
        try {
          _interpreter.run(inputBuffer, output);
          print('✅ 모델 추론 성공 (1D 배열)');
        } catch (e2) {
          print('❌ 모델 추론 실행 오류 (1D): $e2');
          print('❌ 입력 버퍼 크기: ${inputBuffer.length}');
          print('❌ 출력 버퍼 크기: ${output.length}x${output[0].length}');
          print('❌ 예상 입력 크기: $_expectedInputSize');
          return 'others';
        }
      }
      
      // 소프트맥스 함수 적용 (원시 로짓을 확률로 변환)
      List<double> probabilities = _applySoftmax(output[0]);
      
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
      
      // 상세한 분류 결과 출력 (원시값과 확률 모두)
      print('🎯 원시 로짓 값:');
      for (int i = 0; i < exerciseClasses.length; i++) {
        print('  - ${exerciseClasses[i]}: ${output[0][i].toStringAsFixed(3)}');
      }
      print('🎯 소프트맥스 적용 후:');
      for (int i = 0; i < exerciseClasses.length; i++) {
        print('  - ${exerciseClasses[i]}: ${(probabilities[i] * 100).toStringAsFixed(1)}%');
      }
      print('🎯 최종 분류 결과: $predictedExercise (${(maxProbability * 100).toStringAsFixed(1)}%)');
      
      // 신뢰도가 낮으면 'others' 반환 (적절한 임계값 적용)
      if (maxProbability < 0.3) {
        print('📊 신뢰도 낮음: ${(maxProbability * 100).toStringAsFixed(1)}% -> others 반환');
        return 'others';
      }
      
      return predictedExercise;
      
    } catch (e) {
      print('❌ 포즈 분류 오류: $e');
      return 'others';
    }
  }
  
  // 신뢰도와 함께 분류 결과 반환 (MS-G3D 모델 사용)
  Object classifyPoseWithConfidence(List<PoseLandmark> landmarks) {
    if (!_isInitialized) {
      return {'exercise': 'others', 'confidence': 0.0};
    }
    
    try {
      // 포즈 히스토리에 추가
      _addPoseToHistory(landmarks);
      
      // 성능 최적화: 분류 빈도 제어
      _frameCount++;
      if (_frameCount % classificationInterval != 0) {
        return {'exercise': 'others', 'confidence': 0.0};
      }
      
      // 최소 프레임 수 확인 (실시간 성능 최적화)
      int minFrames = 5; // 4D 모델은 최소 5프레임으로 충분
      if (_poseHistory.length < minFrames) {
        return {'exercise': 'others', 'confidence': 0.0};
      }
      
      // 최근 프레임 사용
      int framesToUse = _poseHistory.length > inputFrames ? inputFrames : _poseHistory.length;
      List<List<PoseLandmark>> recentPoses = _poseHistory.sublist(_poseHistory.length - framesToUse);
      
      // 4D 입력 모델 처리 ([1, 64, 25, 3])
      Float32List inputBuffer = _create4DInput(recentPoses);
      
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
      
      // 출력 버퍼 준비
      List<List<double>> output = List.generate(1, (_) => List.filled(exerciseClasses.length, 0.0));
      
      // 모델 추론 실행
      try {
        // 1차원 배열을 4D 배열로 변환하여 시도
        List<List<List<List<double>>>> reshapedInput = _reshapeTo4D(inputBuffer);
        _interpreter.run(reshapedInput, output);
        print('✅ 모델 추론 성공 (4D 배열)');
      } catch (e1) {
        print('⚠️ 4D 배열 실행 실패, 1D 배열로 재시도: $e1');
        try {
          _interpreter.run(inputBuffer, output);
          print('✅ 모델 추론 성공 (1D 배열)');
        } catch (e2) {
          print('❌ 모델 추론 실행 오류 (1D): $e2');
          print('❌ 입력 버퍼 크기: ${inputBuffer.length}');
          print('❌ 출력 버퍼 크기: ${output.length}x${output[0].length}');
          print('❌ 예상 입력 크기: $_expectedInputSize');
          return {'exercise': 'others', 'confidence': 0.0};
        }
      }
      
      // 소프트맥스 함수 적용 (원시 로짓을 확률로 변환)
      List<double> probabilities = _applySoftmax(output[0]);
      
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
      
      // 신뢰도가 낮으면 'others' 반환 (적절한 임계값 적용)
      if (maxProbability < 0.3) {
        print('📊 신뢰도 낮음: ${(maxProbability * 100).toStringAsFixed(1)}% -> others 반환');
        return {'exercise': 'others', 'confidence': maxProbability};
      }
      
      return {'exercise': predictedExercise, 'confidence': maxProbability};
      
    } catch (e) {
      print('❌ 포즈 분류 오류: $e');
      return {'exercise': 'others', 'confidence': 0.0};
    }
  }
  
  // 4D 모델용 입력 생성 [1, 64, 25, 3]
  Float32List _create4DInput(List<List<PoseLandmark>> recentPoses) {
    List<double> flatInput = [];
    
    // 4D 입력 생성: [batch, time, keypoints, channels]
    // batch=1, time=64, keypoints=25, channels=3
    for (int t = 0; t < inputFrames; t++) {
      List<List<List<double>>> frameData;
      
      if (t < recentPoses.length) {
        frameData = _preprocessPoseData(recentPoses[t]);
      } else {
        // 부족한 프레임은 마지막 프레임으로 패딩
        frameData = _preprocessPoseData(recentPoses.last);
      }
      
      // 각 관절점에 대해 3채널 데이터 추가
      for (int k = 0; k < numKeypoints; k++) {
        for (int c = 0; c < numChannels; c++) {
          double value = 0.0;
          if (c < frameData.length && 
              k < frameData[c].length && 
              frameData[c][k].isNotEmpty) {
            value = frameData[c][k][0];
          }
          flatInput.add(value);
        }
      }
    }
    
    // 예상 입력 크기: 1 × 64 × 25 × 3 = 4800
    int expectedSize = inputFrames * numKeypoints * numChannels;
    
    if (flatInput.length != expectedSize) {
      print('⚠️ 4D 입력 크기 조정: 예상=$expectedSize, 실제=${flatInput.length}');
      // 크기 맞춤
      if (flatInput.length < expectedSize) {
        // 부족한 부분은 0으로 패딩
        flatInput.addAll(List.filled(expectedSize - flatInput.length, 0.0));
      } else {
        // 초과하는 부분은 잘라냄
        flatInput = flatInput.sublist(0, expectedSize);
      }
    }
    
    print('✅ 4D 입력 생성: ${flatInput.length} 요소 [1, 64, 25, 3]');
    return Float32List.fromList(flatInput);
  }

  // 1차원 배열을 4D 배열로 변환 [1, 64, 25, 3]
  List<List<List<List<double>>>> _reshapeTo4D(Float32List flatInput) {
    List<List<List<List<double>>>> result = [];
    int index = 0;
    
    for (int b = 0; b < 1; b++) {
      List<List<List<double>>> batch = [];
      
      for (int t = 0; t < inputFrames; t++) {
        List<List<double>> frame = [];
        
        for (int k = 0; k < numKeypoints; k++) {
          List<double> keypoint = [];
          
          for (int c = 0; c < numChannels; c++) {
            if (index < flatInput.length) {
              keypoint.add(flatInput[index].toDouble());
            } else {
              keypoint.add(0.0);
            }
            index++;
          }
          frame.add(keypoint);
        }
        batch.add(frame);
      }
      result.add(batch);
    }
    
    print('✅ 입력 데이터를 4D 형태로 변환: [1, 64, 25, 3]');
    return result;
  }

  // NTU 관절점을 외부에서 접근할 수 있도록 제공
  List<List<double>> getNtuJoints(List<PoseLandmark> landmarks) {
    return _mediapipeToNtu(landmarks);
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
  String classifyPoseSequence(List<List<PoseLandmark>> landmarkSequence) {
    if (landmarkSequence.isEmpty) return 'others';
    return classifyPose(landmarkSequence.last);
  }
}