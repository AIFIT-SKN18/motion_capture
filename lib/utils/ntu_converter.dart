import 'dart:typed_data';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

/// MediaPipe 포즈 랜드마크를 NTU RGB+D 25개 관절점 형식으로 변환하는 유틸리티
class NTUConverter {
  // MediaPipe 33 인덱스를 ML Kit PoseLandmarkType에 매핑
  static const List<PoseLandmarkType> _mpIndexToType = [
    PoseLandmarkType.nose,                // 0
    PoseLandmarkType.leftEyeInner,        // 1
    PoseLandmarkType.leftEye,             // 2
    PoseLandmarkType.leftEyeOuter,        // 3
    PoseLandmarkType.rightEyeInner,       // 4
    PoseLandmarkType.rightEye,            // 5
    PoseLandmarkType.rightEyeOuter,       // 6
    PoseLandmarkType.leftEar,             // 7
    PoseLandmarkType.rightEar,            // 8
    PoseLandmarkType.leftMouth,           // 9
    PoseLandmarkType.rightMouth,          // 10
    PoseLandmarkType.leftShoulder,        // 11
    PoseLandmarkType.rightShoulder,       // 12
    PoseLandmarkType.leftElbow,           // 13
    PoseLandmarkType.rightElbow,          // 14
    PoseLandmarkType.leftWrist,           // 15
    PoseLandmarkType.rightWrist,          // 16
    PoseLandmarkType.leftPinky,           // 17
    PoseLandmarkType.rightPinky,          // 18
    PoseLandmarkType.leftIndex,           // 19
    PoseLandmarkType.rightIndex,          // 20
    PoseLandmarkType.leftThumb,           // 21
    PoseLandmarkType.rightThumb,          // 22
    PoseLandmarkType.leftHip,             // 23
    PoseLandmarkType.rightHip,            // 24
    PoseLandmarkType.leftKnee,            // 25
    PoseLandmarkType.rightKnee,           // 26
    PoseLandmarkType.leftAnkle,           // 27
    PoseLandmarkType.rightAnkle,          // 28
    PoseLandmarkType.leftHeel,            // 29
    PoseLandmarkType.rightHeel,           // 30
    PoseLandmarkType.leftFootIndex,       // 31
    PoseLandmarkType.rightFootIndex,      // 32
  ];

  static PoseLandmark? _landmarkByMpIndex(
    List<PoseLandmark> landmarks,
    int mpIndex,
  ) {
    if (mpIndex < 0 || mpIndex >= _mpIndexToType.length) return null;
    final type = _mpIndexToType[mpIndex];
    for (final lm in landmarks) {
      if (lm.type == type) return lm;
    }
    return null;
  }
  /// MediaPipe 33개 랜드마크를 NTU RGB+D 25개 관절점으로 변환
  /// 
  /// [landmarks]: MediaPipe 포즈 랜드마크 (33개)
  /// [imageWidth]: 이미지 너비 (정규화용)
  /// [imageHeight]: 이미지 높이 (정규화용)
  /// 
  /// Returns: 25 x 3 (joints x coordinates) 배열
  static List<List<double>> mediapipeToNTU(
    List<PoseLandmark> landmarks,
    double imageWidth,
    double imageHeight, {
    bool flipHorizontally = false,
  }) {
    // 25개 관절점 초기화
    List<List<double>> joints = List.generate(25, (_) => [0.0, 0.0, 0.0]);

    // landmarks는 Map 순서가 보장되지 않으므로 타입기반 조회 사용

    // 좌표 정규화 함수 (ms_g3d와 동일 의도: x,y는 0~1, z는 원본 사용)
    double normalizeX(double x) {
      double px = flipHorizontally ? (imageWidth - x) : x;
      return (px / imageWidth).clamp(0.0, 1.0);
    }
    double normalizeY(double y) => (y / imageHeight).clamp(0.0, 1.0);
    // MediaPipe의 z는 이미지 폭을 기준으로 정규화된 값(대개 음수)인데
    // ML Kit의 z 스케일과 다를 수 있어 혼선을 줄이기 위해 0으로 고정
    // (훈련 파이프라인과 차원을 맞추되, z 스케일 불일치 영향 제거)
    // -> z값을 0으로 고정하면 스쿼트, 런지, 데드리프트와 같이 깊이 정보가 중요한 운동을 구분하기 어려움.
    // -> Python 코드(원본)는 z값을 사용하므로, Dart에서도 z값을 사용하도록 수정.
    // -> ML Kit의 z값은 x와 유사한 스케일을 가지므로 imageWidth로 정규화.
    double normalizeZ(double z) => z / imageWidth;

    // NTU RGB+D 25개 관절점 매핑
    
    // 0: spine_base (hip center) - 양쪽 힙의 중점
    final lm23 = _landmarkByMpIndex(landmarks, 23);
    final lm24 = _landmarkByMpIndex(landmarks, 24);
    if (lm23 == null || lm24 == null) return joints;
    joints[0] = [
      normalizeX((lm23.x + lm24.x) / 2),
      normalizeY((lm23.y + lm24.y) / 2),
      normalizeZ((lm23.z + lm24.z) / 2),
    ];

    // 1: spine_mid - hip center와 shoulder center의 중점
    List<double> hipCenter = joints[0];
    final lm11 = _landmarkByMpIndex(landmarks, 11);
    final lm12 = _landmarkByMpIndex(landmarks, 12);
    if (lm11 == null || lm12 == null) return joints;
    List<double> shoulderCenter = [
      normalizeX((lm11.x + lm12.x) / 2),
      normalizeY((lm11.y + lm12.y) / 2),
      normalizeZ((lm11.z + lm12.z) / 2),
    ];
    joints[1] = [
      (hipCenter[0] + shoulderCenter[0]) / 2,
      (hipCenter[1] + shoulderCenter[1]) / 2,
      (hipCenter[2] + shoulderCenter[2]) / 2,
    ];

    // 2: neck - shoulder center에서 약간 위로
    joints[2] = [
      shoulderCenter[0],
      shoulderCenter[1] - 0.05,
      shoulderCenter[2],
    ];

    // 3: head - nose
    final lm0 = _landmarkByMpIndex(landmarks, 0);
    if (lm0 != null) {
      joints[3] = [normalizeX(lm0.x), normalizeY(lm0.y), normalizeZ(lm0.z)];
    }

    // 4-7: left arm (shoulder, elbow, wrist, hand)
    // 좌우 반전 시 left/right 관절점을 바꿔서 매핑
    if (flipHorizontally) {
      // 좌우 반전 시: left arm = right arm의 좌표
      joints[4] = [normalizeX(lm12.x), normalizeY(lm12.y), normalizeZ(lm12.z)];
      final lm14 = _landmarkByMpIndex(landmarks, 14);
      if (lm14 != null) joints[5] = [normalizeX(lm14.x), normalizeY(lm14.y), normalizeZ(lm14.z)];
      final lm16 = _landmarkByMpIndex(landmarks, 16);
      if (lm16 != null) joints[6] = [normalizeX(lm16.x), normalizeY(lm16.y), normalizeZ(lm16.z)];
      if (lm16 != null) joints[7] = [normalizeX(lm16.x), normalizeY(lm16.y), normalizeZ(lm16.z)];

      // 8-11: right arm (shoulder, elbow, wrist, hand)
      // 좌우 반전 시: right arm = left arm의 좌표
      joints[8] = [normalizeX(lm11.x), normalizeY(lm11.y), normalizeZ(lm11.z)];
      final lm13 = _landmarkByMpIndex(landmarks, 13);
      if (lm13 != null) joints[9] = [normalizeX(lm13.x), normalizeY(lm13.y), normalizeZ(lm13.z)];
      final lm15 = _landmarkByMpIndex(landmarks, 15);
      if (lm15 != null) joints[10] = [normalizeX(lm15.x), normalizeY(lm15.y), normalizeZ(lm15.z)];
      if (lm15 != null) joints[11] = [normalizeX(lm15.x), normalizeY(lm15.y), normalizeZ(lm15.z)];
    } else {
      // 정상 매핑
      joints[4] = [normalizeX(lm11.x), normalizeY(lm11.y), normalizeZ(lm11.z)];
      final lm13 = _landmarkByMpIndex(landmarks, 13);
      if (lm13 != null) joints[5] = [normalizeX(lm13.x), normalizeY(lm13.y), normalizeZ(lm13.z)];
      final lm15 = _landmarkByMpIndex(landmarks, 15);
      if (lm15 != null) joints[6] = [normalizeX(lm15.x), normalizeY(lm15.y), normalizeZ(lm15.z)];
      if (lm15 != null) joints[7] = [normalizeX(lm15.x), normalizeY(lm15.y), normalizeZ(lm15.z)];

      // 8-11: right arm (shoulder, elbow, wrist, hand)
      joints[8] = [normalizeX(lm12.x), normalizeY(lm12.y), normalizeZ(lm12.z)];
      final lm14 = _landmarkByMpIndex(landmarks, 14);
      if (lm14 != null) joints[9] = [normalizeX(lm14.x), normalizeY(lm14.y), normalizeZ(lm14.z)];
      final lm16 = _landmarkByMpIndex(landmarks, 16);
      if (lm16 != null) joints[10] = [normalizeX(lm16.x), normalizeY(lm16.y), normalizeZ(lm16.z)];
      if (lm16 != null) joints[11] = [normalizeX(lm16.x), normalizeY(lm16.y), normalizeZ(lm16.z)];
    }

    // 12-15: left leg (hip, knee, ankle, foot)
    // 16-19: right leg (hip, knee, ankle, foot)
    if (flipHorizontally) {
      // 좌우 반전 시: left leg = right leg의 좌표
      joints[12] = [normalizeX(lm24.x), normalizeY(lm24.y), normalizeZ(lm24.z)];
      final lm26 = _landmarkByMpIndex(landmarks, 26);
      if (lm26 != null) joints[13] = [normalizeX(lm26.x), normalizeY(lm26.y), normalizeZ(lm26.z)];
      final lm28 = _landmarkByMpIndex(landmarks, 28);
      if (lm28 != null) joints[14] = [normalizeX(lm28.x), normalizeY(lm28.y), normalizeZ(lm28.z)];
      final lm32 = _landmarkByMpIndex(landmarks, 32);
      if (lm32 != null) joints[15] = [normalizeX(lm32.x), normalizeY(lm32.y), normalizeZ(lm32.z)];

      // 좌우 반전 시: right leg = left leg의 좌표
      joints[16] = [normalizeX(lm23.x), normalizeY(lm23.y), normalizeZ(lm23.z)];
      final lm25 = _landmarkByMpIndex(landmarks, 25);
      if (lm25 != null) joints[17] = [normalizeX(lm25.x), normalizeY(lm25.y), normalizeZ(lm25.z)];
      final lm27 = _landmarkByMpIndex(landmarks, 27);
      if (lm27 != null) joints[18] = [normalizeX(lm27.x), normalizeY(lm27.y), normalizeZ(lm27.z)];
      final lm31 = _landmarkByMpIndex(landmarks, 31);
      if (lm31 != null) joints[19] = [normalizeX(lm31.x), normalizeY(lm31.y), normalizeZ(lm31.z)];
    } else {
      // 정상 매핑
      joints[12] = [normalizeX(lm23.x), normalizeY(lm23.y), normalizeZ(lm23.z)];
      final lm25 = _landmarkByMpIndex(landmarks, 25);
      if (lm25 != null) joints[13] = [normalizeX(lm25.x), normalizeY(lm25.y), normalizeZ(lm25.z)];
      final lm27 = _landmarkByMpIndex(landmarks, 27);
      if (lm27 != null) joints[14] = [normalizeX(lm27.x), normalizeY(lm27.y), normalizeZ(lm27.z)];
      final lm31 = _landmarkByMpIndex(landmarks, 31);
      if (lm31 != null) joints[15] = [normalizeX(lm31.x), normalizeY(lm31.y), normalizeZ(lm31.z)];

      joints[16] = [normalizeX(lm24.x), normalizeY(lm24.y), normalizeZ(lm24.z)];
      final lm26 = _landmarkByMpIndex(landmarks, 26);
      if (lm26 != null) joints[17] = [normalizeX(lm26.x), normalizeY(lm26.y), normalizeZ(lm26.z)];
      final lm28 = _landmarkByMpIndex(landmarks, 28);
      if (lm28 != null) joints[18] = [normalizeX(lm28.x), normalizeY(lm28.y), normalizeZ(lm28.z)];
      final lm32 = _landmarkByMpIndex(landmarks, 32);
      if (lm32 != null) joints[19] = [normalizeX(lm32.x), normalizeY(lm32.y), normalizeZ(lm32.z)];
    }

    // 20-24: additional points
    joints[20] = shoulderCenter;
    
    if (flipHorizontally) {
      // 좌우 반전 시: left hand points = right hand points의 좌표
      final lm18 = _landmarkByMpIndex(landmarks, 18);
      if (lm18 != null) joints[21] = [normalizeX(lm18.x), normalizeY(lm18.y), normalizeZ(lm18.z)];
      final lm22 = _landmarkByMpIndex(landmarks, 22);
      if (lm22 != null) joints[22] = [normalizeX(lm22.x), normalizeY(lm22.y), normalizeZ(lm22.z)];
      
      // 좌우 반전 시: right hand points = left hand points의 좌표
      final lm17 = _landmarkByMpIndex(landmarks, 17);
      if (lm17 != null) joints[23] = [normalizeX(lm17.x), normalizeY(lm17.y), normalizeZ(lm17.z)];
      final lm21 = _landmarkByMpIndex(landmarks, 21);
      if (lm21 != null) joints[24] = [normalizeX(lm21.x), normalizeY(lm21.y), normalizeZ(lm21.z)];
    } else {
      // 정상 매핑
      final lm17 = _landmarkByMpIndex(landmarks, 17);
      if (lm17 != null) joints[21] = [normalizeX(lm17.x), normalizeY(lm17.y), normalizeZ(lm17.z)];
      final lm21 = _landmarkByMpIndex(landmarks, 21);
      if (lm21 != null) joints[22] = [normalizeX(lm21.x), normalizeY(lm21.y), normalizeZ(lm21.z)];
      final lm18 = _landmarkByMpIndex(landmarks, 18);
      if (lm18 != null) joints[23] = [normalizeX(lm18.x), normalizeY(lm18.y), normalizeZ(lm18.z)];
      final lm22 = _landmarkByMpIndex(landmarks, 22);
      if (lm22 != null) joints[24] = [normalizeX(lm22.x), normalizeY(lm22.y), normalizeZ(lm22.z)];
    }

    return joints;
  }

  /// 포즈 시퀀스를 모델 입력 형식으로 변환
  /// 
  /// [sequence]: 프레임별 NTU 관절점 시퀀스 (T x V x C)
  /// [targetFrames]: 목표 프레임 수 (기본값: 64)
  /// 
  /// Returns: [1, 3, T, 25, 1] shape의 Float32List
  static Float32List createModelInput(
    List<List<List<double>>> sequence, {
    int targetFrames = 100,
  }) {
    const int numChannels = 3; // x, y, z
    const int numKeypoints = 25;
    const int numPersons = 1;

    // 1. 프레임 수 조정 (ms_g3d와 동일: T>64는 균등간격 샘플, T<64는 edge 패딩)
    final int Tsrc = sequence.length;
    List<List<List<double>>> adjustedSequence;
    if (Tsrc > targetFrames) {
      adjustedSequence = [];
      for (int i = 0; i < targetFrames; i++) {
        final double pos = i * (Tsrc - 1) / (targetFrames - 1);
        final int idx = pos.floor();
        adjustedSequence.add(sequence[idx]);
      }
    } else if (Tsrc < targetFrames) {
      adjustedSequence = List.from(sequence);
      final padCount = targetFrames - Tsrc;
      final List<List<double>> padFrame = sequence.isNotEmpty
          ? sequence.last
          : List.generate(25, (_) => [0.0, 0.0, 0.0]);
      for (int i = 0; i < padCount; i++) {
        adjustedSequence.add(padFrame);
      }
    } else {
      adjustedSequence = sequence;
    }

    // 2. (T, V, C) → (C, T, V, M) transpose (정규화 없음)
    List<double> flatInput = [];
    for (int c = 0; c < numChannels; c++) {
      for (int t = 0; t < targetFrames; t++) {
        for (int v = 0; v < numKeypoints; v++) {
          for (int m = 0; m < numPersons; m++) {
            double value = adjustedSequence[t][v][c];
            if (value.isNaN || value.isInfinite) {
              value = 0.0;
            }
            flatInput.add(value);
          }
        }
      }
    }

    return Float32List.fromList(flatInput);
  }
}