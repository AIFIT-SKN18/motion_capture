import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

/// MediaPipe 포즈 감지 서비스
class PoseDetectorService {
  PoseDetector? _poseDetector;
  bool _isInitialized = false;

  bool get isInitialized => _isInitialized;

  /// MediaPipe 포즈 감지기 초기화
  Future<void> initialize() async {
    try {
      print('🚀 MediaPipe 포즈 감지기 초기화 시작...');

      _poseDetector = PoseDetector(
        options: PoseDetectorOptions(
          mode: PoseDetectionMode.stream, // 실시간 스트림 모드
          model: PoseDetectionModel.accurate, // 정확도 우선 모델
        ),
      );

      _isInitialized = true;
      print('✅ MediaPipe 포즈 감지기 초기화 완료');
    } catch (e) {
      print('❌ MediaPipe 초기화 실패: $e');
      throw Exception('포즈 감지기 초기화 실패: $e');
    }
  }

  /// 이미지에서 포즈 감지
  /// 
  /// [inputImage]: ML Kit InputImage
  /// Returns: 감지된 포즈 리스트 (보통 1개)
  Future<List<Pose>> detectPose(InputImage inputImage) async {
    if (!_isInitialized || _poseDetector == null) {
      throw Exception('포즈 감지기가 초기화되지 않았습니다');
    }

    try {
      final poses = await _poseDetector!.processImage(inputImage);
      return poses;
    } catch (e) {
      print('❌ 포즈 감지 오류: $e');
      return [];
    }
  }

  /// 리소스 해제
  void dispose() {
    _poseDetector?.close();
    _poseDetector = null;
    _isInitialized = false;
    print('🗑️ MediaPipe 포즈 감지기 리소스 해제');
  }
}