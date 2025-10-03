import 'dart:async';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import '../utils/ntu_converter.dart';
import '../services/pose_detector.dart';
import '../services/pose_classifier.dart';

/// 실시간 동작 인식 화면
class LiveScreen extends StatefulWidget {
  const LiveScreen({super.key});

  @override
  State<LiveScreen> createState() => _LiveScreenState();
}

class _LiveScreenState extends State<LiveScreen> with WidgetsBindingObserver {
  // 카메라
  CameraController? _cameraController;
  List<CameraDescription>? _cameras;
  bool _isCameraInitialized = false;

  // 서비스
  final PoseDetectorService _poseDetector = PoseDetectorService();
  final PoseClassifier _poseClassifier = PoseClassifier();

  // 상태
  String? _selectedExercise; // 선택된 운동
  bool _isRecording = false;
  int _countdown = 0;
  String _currentExercise = '';
  double _confidence = 0.0;
  bool _isProcessing = false;
  List<PoseLandmark> _lastLandmarks = [];

  // 5개 운동 클래스 (others 제외)
  static const List<Map<String, String>> exercises = [
    {'id': 'benchpress', 'name': '벤치프레스', 'icon': '🏋️'},
    {'id': 'deadlift', 'name': '데드리프트', 'icon': '💪'},
    {'id': 'lunges', 'name': '런지', 'icon': '🦵'},
    {'id': 'side_lateral_raise', 'name': '사이드 레터럴 레이즈', 'icon': '🙆'},
    {'id': 'squat', 'name': '스쿼트', 'icon': '🏃'},
  ];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeServices();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _cameraController?.dispose();
    _poseDetector.dispose();
    _poseClassifier.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      _cameraController?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initializeCamera();
    }
  }

  Future<void> _initializeServices() async {
    try {
      // 카메라 권한 요청
      final cameraStatus = await Permission.camera.request();
      if (cameraStatus.isDenied) {
        _showPermissionDialog();
        return;
      }

      // 서비스 초기화
      await _poseDetector.initialize();
      await _poseClassifier.initialize();

      // 카메라 초기화는 운동 선택 후에 수행
    } catch (e) {
      _showErrorDialog('초기화 실패', e.toString());
    }
  }

  Future<void> _initializeCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras == null || _cameras!.isEmpty) {
        throw Exception('사용 가능한 카메라가 없습니다');
      }

      // 전면 카메라 선택
      final frontCamera = _cameras!.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.front,
        orElse: () => _cameras!.first,
      );

      _cameraController = CameraController(
        frontCamera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.nv21,
      );

      await _cameraController!.initialize();

      if (mounted) {
        setState(() {
          _isCameraInitialized = true;
        });
      }
    } catch (e) {
      _showErrorDialog('카메라 초기화 실패', e.toString());
    }
  }

  void _selectExercise(String exerciseId) async {
    setState(() {
      _selectedExercise = exerciseId;
    });

    // 운동 선택 후 카메라 초기화
    if (!_isCameraInitialized) {
      await _initializeCamera();
    }
  }

  Future<void> _startCountdown() async {
    if (_selectedExercise == null) {
      _showErrorDialog('운동 선택 필요', '먼저 운동을 선택해주세요');
      return;
    }

    setState(() {
      _countdown = 3;
    });

    for (int i = 3; i > 0; i--) {
      setState(() {
        _countdown = i;
      });
      await Future.delayed(const Duration(seconds: 1));
    }

    setState(() {
      _countdown = 0;
      _isRecording = true;
    });

    _poseClassifier.clearHistory();
    _startPoseDetection();
  }

  void _startPoseDetection() {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    _cameraController!.startImageStream((CameraImage image) {
      if (_isProcessing || !_isRecording) return;
      _processFrame(image);
    });
  }

  Future<void> _processFrame(CameraImage image) async {
    if (_isProcessing) return;

    setState(() {
      _isProcessing = true;
    });

    try {
      // CameraImage를 InputImage로 변환
      final inputImage = _convertCameraImage(image);
      if (inputImage == null) {
        setState(() {
          _isProcessing = false;
        });
        return;
      }

      // 포즈 감지
      final poses = await _poseDetector.detectPose(inputImage);
      if (poses.isEmpty || poses.first.landmarks.isEmpty) {
        setState(() {
          _currentExercise = '포즈를 감지할 수 없습니다';
          _confidence = 0.0;
          _isProcessing = false;
        });
        return;
      }

      // 분류
      final result = await _poseClassifier.classifyRealtime(
        poses.first.landmarks.values.toList(),
        image.width.toDouble(),
        image.height.toDouble(),
      );

      // 선택한 운동과 일치하는지 확인 + 최근 랜드마크 저장
      if (mounted) {
        setState(() {
          if (result.exercise == _selectedExercise) {
            _currentExercise = '✅ ${_getKoreanExerciseName(result.exercise)}';
            _confidence = result.confidence;
          } else {
            _currentExercise = '❌ ${_getKoreanExerciseName(result.exercise)}';
            _confidence = result.confidence;
          }
          _lastLandmarks = poses.first.landmarks.values.toList();
        });
      }
    } catch (e) {
      print('프레임 처리 오류: $e');
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  InputImage? _convertCameraImage(CameraImage image) {
    try {
      final camera = _cameraController!.description;

      InputImageRotation? rotation;
      if (camera.lensDirection == CameraLensDirection.front) {
        rotation = InputImageRotation.rotation270deg;
      } else {
        rotation = InputImageRotation.rotation90deg;
      }

      final format = InputImageFormatValue.fromRawValue(image.format.raw);
      if (format == null) return null;

      final plane = image.planes.first;
      return InputImage.fromBytes(
        bytes: plane.bytes,
        metadata: InputImageMetadata(
          size: Size(image.width.toDouble(), image.height.toDouble()),
          rotation: rotation,
          format: format,
          bytesPerRow: plane.bytesPerRow,
        ),
      );
    } catch (e) {
      print('이미지 변환 오류: $e');
      return null;
    }
  }

  void _stopRecording() {
    _cameraController?.stopImageStream();
    setState(() {
      _isRecording = false;
      _currentExercise = '';
      _confidence = 0.0;
    });
  }

  String _getKoreanExerciseName(String exercise) {
    const Map<String, String> names = {
      'benchpress': '벤치프레스',
      'deadlift': '데드리프트',
      'lunges': '런지',
      'side_lateral_raise': '사이드 레터럴 레이즈',
      'squat': '스쿼트',
      'others': '기타',
    };
    return names[exercise] ?? exercise;
  }

  void _showPermissionDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('카메라 권한 필요'),
        content: const Text('실시간 인식을 위해 카메라 권한이 필요합니다.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('취소'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              openAppSettings();
            },
            child: const Text('설정으로 이동'),
          ),
        ],
      ),
    );
  }

  void _showErrorDialog(String title, String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('확인'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('실시간 동작 인식'),
        centerTitle: true,
      ),
      body: _selectedExercise == null
          ? _buildExerciseSelection()
          : _buildCameraView(),
    );
  }

  // 운동 선택 화면
  Widget _buildExerciseSelection() {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              '인식할 운동을 선택하세요',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 12),
            Text(
              '선택한 운동 동작을 실시간으로 인식합니다',
              style: TextStyle(
                fontSize: 16,
                color: Colors.grey[600],
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 32),
            Expanded(
              child: ListView.builder(
                itemCount: exercises.length,
                itemBuilder: (context, index) {
                  final exercise = exercises[index];
                  return Card(
                    margin: const EdgeInsets.only(bottom: 16),
                    elevation: 2,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: InkWell(
                      onTap: () => _selectExercise(exercise['id']!),
                      borderRadius: BorderRadius.circular(16),
                      child: Padding(
                        padding: const EdgeInsets.all(20.0),
                        child: Row(
                          children: [
                            Container(
                              width: 60,
                              height: 60,
                              decoration: BoxDecoration(
                                color: Colors.blue.withOpacity(0.1),
                                borderRadius: BorderRadius.circular(12),
                              ),
                              child: Center(
                                child: Text(
                                  exercise['icon']!,
                                  style: const TextStyle(fontSize: 32),
                                ),
                              ),
                            ),
                            const SizedBox(width: 16),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    exercise['name']!,
                                    style: const TextStyle(
                                      fontSize: 18,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    exercise['id']!,
                                    style: TextStyle(
                                      fontSize: 14,
                                      color: Colors.grey[600],
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            Icon(
                              Icons.arrow_forward_ios,
                              color: Colors.grey[400],
                            ),
                          ],
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  // 카메라 뷰
  Widget _buildCameraView() {
    return Stack(
      children: [
        // 카메라 프리뷰 (전체화면)
        if (_isCameraInitialized)
          Positioned.fill(
            child: FittedBox(
              fit: BoxFit.cover,
              child: SizedBox(
                width: _cameraController!.value.previewSize!.height,
                height: _cameraController!.value.previewSize!.width,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    CameraPreview(_cameraController!),
                    CustomPaint(
                      painter: _PoseOverlayPainter(
                        landmarks: _lastLandmarks,
                        imageWidth: _cameraController!.value.previewSize!.height,
                        imageHeight: _cameraController!.value.previewSize!.width,
                        drawColor: Colors.greenAccent,
                        strokeWidth: 3.0,
                        isFrontCamera: _cameraController!.description.lensDirection == CameraLensDirection.front,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          )
        else
          const Center(child: CircularProgressIndicator()),

        // 선택된 운동 표시
        Positioned(
          top: 20,
          left: 20,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.black87,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  exercises.firstWhere((e) => e['id'] == _selectedExercise)['icon']!,
                  style: const TextStyle(fontSize: 20),
                ),
                const SizedBox(width: 8),
                Text(
                  exercises.firstWhere((e) => e['id'] == _selectedExercise)['name']!,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
        ),

        // 운동 변경 버튼
        Positioned(
          top: 20,
          right: 20,
          child: IconButton(
            onPressed: () {
              _stopRecording();
              setState(() {
                _selectedExercise = null;
                _isCameraInitialized = false;
              });
              _cameraController?.dispose();
              _cameraController = null;
            },
            icon: const Icon(Icons.change_circle, color: Colors.white, size: 32),
            style: IconButton.styleFrom(
              backgroundColor: Colors.black54,
            ),
          ),
        ),

        // 카운트다운 오버레이
        if (_countdown > 0)
          Container(
            color: Colors.black54,
            child: Center(
              child: Text(
                '$_countdown',
                style: const TextStyle(
                  fontSize: 120,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
            ),
          ),

        // 결과 표시
        if (_isRecording && _currentExercise.isNotEmpty)
          Positioned(
            top: 80,
            left: 0,
            right: 0,
            child: Container(
              margin: const EdgeInsets.symmetric(horizontal: 20),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: _currentExercise.startsWith('✅')
                    ? Colors.green.withOpacity(0.9)
                    : Colors.red.withOpacity(0.9),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                children: [
                  Text(
                    _currentExercise,
                    style: const TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '신뢰도: ${(_confidence * 100).toStringAsFixed(1)}%',
                    style: const TextStyle(
                      fontSize: 16,
                      color: Colors.white70,
                    ),
                  ),
                ],
              ),
            ),
          ),

        // 컨트롤 버튼
        Positioned(
          bottom: 40,
          left: 0,
          right: 0,
          child: Center(
            child: _isRecording
                ? FloatingActionButton.extended(
                    onPressed: _stopRecording,
                    backgroundColor: Colors.red,
                    icon: const Icon(Icons.stop),
                    label: const Text('중지'),
                  )
                : FloatingActionButton.extended(
                    onPressed: _countdown > 0 ? null : _startCountdown,
                    backgroundColor: Colors.blue,
                    icon: const Icon(Icons.play_arrow),
                    label: const Text('시작'),
                  ),
          ),
        ),
      ],
    );
  }
}

class _PoseOverlayPainter extends CustomPainter {
  final List<PoseLandmark> landmarks;
  final double imageWidth;
  final double imageHeight;
  final Color drawColor;
  final double strokeWidth;
  final bool isFrontCamera;

  _PoseOverlayPainter({
    required this.landmarks,
    required this.imageWidth,
    required this.imageHeight,
    required this.drawColor,
    required this.strokeWidth,
    required this.isFrontCamera,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (landmarks.isEmpty || imageWidth == 0 || imageHeight == 0) return;

    final paintLine = Paint()
      ..color = drawColor.withOpacity(0.9)
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final paintPoint = Paint()
      ..color = drawColor
      ..style = PaintingStyle.fill;

    // NTU 25 관절로 변환 (정규화 좌표 0~1)
    final joints = NTUConverter.mediapipeToNTU(
      landmarks,
      imageWidth,
      imageHeight,
    );

    // 좌표 변환 함수: NTU 정규화 좌표 -> 화면 좌표
    Offset toCanvas(double x, double y) {
      double cx = x * size.width;
      double cy = y * size.height;
      if (isFrontCamera) {
        // 전면 카메라는 좌우 반전
        cx = size.width - cx;
      }
      return Offset(cx, cy);
    }

    // 뼈대(에지) 정의 (간단한 연결)
    final edges = <List<int>>[
      [0, 1], [1, 2], [2, 3],
      [1, 4], [4, 5], [5, 6], [6, 7],
      [1, 8], [8, 9], [9, 10], [10, 11],
      [0, 12], [12, 13], [13, 14], [14, 15],
      [0, 16], [16, 17], [17, 18], [18, 19],
    ];

    // 몸통 사각형(hip center ~ shoulder center를 기준으로 어깨/엉덩이 좌우를 근사) 그리기
    // NTU에는 좌/우 어깨/엉덩이 포인트가 직접 없으므로 MediaPipe 인덱스를 사용해 직접 계산
    try {
      final mp = landmarks; // MediaPipe 원본 랜드마크 사용
      if (mp.length >= 33) {
        // 어깨 좌/우: 11(L) 12(R), 엉덩이 좌/우: 23(L) 24(R)
        final leftShoulder = toCanvas(mp[11].x / imageWidth, mp[11].y / imageHeight);
        final rightShoulder = toCanvas(mp[12].x / imageWidth, mp[12].y / imageHeight);
        final leftHip = toCanvas(mp[23].x / imageWidth, mp[23].y / imageHeight);
        final rightHip = toCanvas(mp[24].x / imageWidth, mp[24].y / imageHeight);

        final torsoPath = Path()
          ..moveTo(leftShoulder.dx, leftShoulder.dy)
          ..lineTo(rightShoulder.dx, rightShoulder.dy)
          ..lineTo(rightHip.dx, rightHip.dy)
          ..lineTo(leftHip.dx, leftHip.dy)
          ..close();
        canvas.drawPath(torsoPath, paintLine);
      }
    } catch (_) {}

    // 에지 그리기
    for (final e in edges) {
      final p1 = toCanvas(joints[e[0]][0], joints[e[0]][1]);
      final p2 = toCanvas(joints[e[1]][0], joints[e[1]][1]);
      canvas.drawLine(p1, p2, paintLine);
    }

    // 포인트 그리기
    for (final j in joints) {
      final p = toCanvas(j[0], j[1]);
      canvas.drawCircle(p, strokeWidth + 1.5, paintPoint);
      canvas.drawCircle(p, strokeWidth + 1.5, paintLine);
    }
  }

  @override
  bool shouldRepaint(covariant _PoseOverlayPainter oldDelegate) {
    return oldDelegate.landmarks != landmarks ||
        oldDelegate.imageWidth != imageWidth ||
        oldDelegate.imageHeight != imageHeight ||
        oldDelegate.isFrontCamera != isFrontCamera;
  }
}
