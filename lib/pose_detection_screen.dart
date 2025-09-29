import 'dart:io';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:permission_handler/permission_handler.dart';
import 'pose_classifier.dart';

class PoseDetectionScreen extends StatefulWidget {
  @override
  _PoseDetectionScreenState createState() => _PoseDetectionScreenState();
}

class _PoseDetectionScreenState extends State<PoseDetectionScreen> {
  CameraController? _cameraController;
  late PoseDetector _poseDetector;
  late PoseClassifier _poseClassifier;
  bool _isBusy = false;
  List<Pose> _poses = [];
  String _currentExercise = 'others';
  double _confidence = 0.0;
  List<List<PoseLandmark>> _landmarkHistory = [];
  static const int maxHistoryLength = 30; // 최근 30프레임 저장

  final _orientations = {
    DeviceOrientation.portraitUp: 0,
    DeviceOrientation.landscapeLeft: 90,
    DeviceOrientation.portraitDown: 180,
    DeviceOrientation.landscapeRight: 270,
  };

  @override
  void initState() {
    super.initState();
    _requestPermissions();
    _poseDetector =
        PoseDetector(options: PoseDetectorOptions(mode: PoseDetectionMode.stream));
    _poseClassifier = PoseClassifier();
    _initializeClassifier();
  }
  
  Future<void> _requestPermissions() async {
    try {
      print('🔐 권한 요청 시작...');
      
      // 카메라 권한 요청
      var cameraStatus = await Permission.camera.status;
      if (!cameraStatus.isGranted) {
        cameraStatus = await Permission.camera.request();
      }
      
      if (cameraStatus.isGranted) {
        print('✅ 카메라 권한 허용됨');
        _initializeCamera();
      } else {
        print('❌ 카메라 권한 거부됨');
        if (mounted) {
          _showPermissionDialog();
        }
      }
    } catch (e) {
      print('❌ 권한 요청 오류: $e');
      _initializeCamera(); // 권한 요청 실패 시에도 카메라 초기화 시도
    }
  }
  
  void _showPermissionDialog() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('권한 필요'),
          content: Text('카메라 권한이 필요합니다. 설정에서 권한을 허용해주세요.'),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
                openAppSettings();
              },
              child: Text('설정으로 이동'),
            ),
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
              },
              child: Text('취소'),
            ),
          ],
        );
      },
    );
  }
  
  Future<void> _initializeClassifier() async {
    try {
      print('🎬 앱 시작 - 포즈 분류기 초기화 중...');
      await _poseClassifier.initialize();
      print('✅ 포즈 분류기 초기화 완료');
      
      // 모델 상태 확인
      if (_poseClassifier.isInitialized) {
        print('✅ 모델이 성공적으로 로드되었습니다!');
      } else {
        print('❌ 모델 로드 실패');
      }
      
    } catch (e) {
      print('❌ 포즈 분류기 초기화 실패: $e');
      print('🚨 앱이 정상적으로 작동하지 않을 수 있습니다.');
    }
  }

  Future<void> _initializeCamera() async {
    try {
      print('📷 카메라 초기화 시작...');
      final cameras = await availableCameras();
      print('📷 사용 가능한 카메라 수: ${cameras.length}');
      
      if (cameras.isEmpty) {
        print('❌ 사용 가능한 카메라가 없습니다');
        return;
      }
      
      CameraDescription selectedCamera = cameras.first;
      for (final camera in cameras) {
        if (camera.lensDirection == CameraLensDirection.front) {
          selectedCamera = camera;
          break;
        }
      }
      
      print('📷 선택된 카메라: ${selectedCamera.name} (${selectedCamera.lensDirection})');
      
      _cameraController = CameraController(
        selectedCamera, 
        ResolutionPreset.medium, 
        imageFormatGroup: ImageFormatGroup.nv21,
        enableAudio: false,
      );
      
      await _cameraController!.initialize();
      
      if (!mounted) return;
      
      print('📷 카메라 컨트롤러 초기화 완료');
      print('📷 이미지 스트림 시작...');
      
      _cameraController!.startImageStream(_processCameraImage);
      setState(() {});
      
      print('✅ 카메라 초기화 완료');
    } catch (e) {
      print('❌ 카메라 초기화 실패: $e');
      if (mounted) {
        setState(() {});
      }
    }
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_isBusy || !mounted) return;
    _isBusy = true;

    try {
      final inputImage = _convertToInputImage(image, _cameraController!);
      if (inputImage == null) {
        _isBusy = false;
        return;
      }
      
      final poses = await _poseDetector.processImage(inputImage);

      if (poses.isNotEmpty) {
        final pose = poses.first;
        final landmarks = pose.landmarks.values.toList();
        
        // 랜드마크 히스토리에 추가
        _landmarkHistory.add(landmarks);
        if (_landmarkHistory.length > maxHistoryLength) {
          _landmarkHistory.removeAt(0);
        }
        
        // 운동 분류 수행 (모델이 로드된 경우에만)
        String exercise = 'others';
        double confidence = 0.0;
        
        if (_poseClassifier.isInitialized) {
          try {
            exercise = _poseClassifier.classifyPose(landmarks);
            confidence = exercise != 'others' ? 0.8 : 0.0;
            
            // 운동이 인식되었을 때 추가 로그
            if (exercise != 'others') {
              print('🎉 화면에 표시될 운동: $exercise');
            }
          } catch (e) {
            print('❌ 포즈 분류 오류: $e');
            exercise = 'others';
            confidence = 0.0;
          }
        } else {
          print('⏳ 모델 로딩 중...');
        }
        
        if (mounted) {
          setState(() {
            _poses = poses;
            _currentExercise = exercise;
            _confidence = confidence;
          });
        }
      } else {
        if (mounted) {
          setState(() {
            _poses = poses;
            _currentExercise = 'others';
            _confidence = 0.0;
          });
        }
        print('👤 포즈가 감지되지 않음');
      }
    } catch (e) {
      print('❌ 이미지 처리 오류: $e');
    }

    _isBusy = false;
  }

  InputImage? _convertToInputImage(
      CameraImage image, CameraController controller) {
    final camera = controller.description;
    final sensorOrientation = camera.sensorOrientation;

    InputImageRotation? rotation;
    if (Platform.isAndroid) {
      var rotationCompensation =
          _orientations[controller.value.deviceOrientation] ?? 0;
      if (camera.lensDirection == CameraLensDirection.front) {
        // front-facing
        rotationCompensation = (sensorOrientation + rotationCompensation) % 360;
      } else {
        // back-facing
        rotationCompensation =
            (sensorOrientation - rotationCompensation + 360) % 360;
      }
      rotation = InputImageRotationValue.fromRawValue(rotationCompensation);
    } else {
      rotation = InputImageRotationValue.fromRawValue(sensorOrientation);
    }

    if (rotation == null) return null;

    final format = InputImageFormatValue.fromRawValue(image.format.raw);

    if (format == null ||
        (Platform.isAndroid && format != InputImageFormat.nv21) ||
        (Platform.isIOS && format != InputImageFormat.bgra8888)) {
      return null;
    }

    if (image.planes.length != 1) {
      return null;
    }
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
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _poseDetector.close();
    _poseClassifier.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(_cameraController!),
          CustomPaint(
            painter: PosePainter(_poses, _cameraController!.value.previewSize!,
                _cameraController!.description.lensDirection, poseClassifier: _poseClassifier),
          ),
          // 운동 분류 결과 표시
          Positioned(
            top: 50,
            left: 20,
            right: 20,
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.8),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: _getExerciseColor(_currentExercise),
                  width: 2,
                ),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        _getExerciseIcon(_currentExercise),
                        color: _getExerciseColor(_currentExercise),
                        size: 24,
                      ),
                      SizedBox(width: 8),
                      Text(
                        '운동 분류 결과',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  SizedBox(height: 12),
                  Container(
                    padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    decoration: BoxDecoration(
                      color: _getExerciseColor(_currentExercise).withOpacity(0.2),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      _getExerciseDisplayName(_currentExercise),
                      style: TextStyle(
                        color: _getExerciseColor(_currentExercise),
                        fontSize: 28,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    '신뢰도: ${(_confidence * 100).toStringAsFixed(1)}%',
                    style: TextStyle(
                      color: Colors.white70,
                      fontSize: 14,
                    ),
                  ),
                  Text(
                    '모델 상태: ${_poseClassifier.isInitialized ? "로드됨" : "로드 중..."}',
                    style: TextStyle(
                      color: _poseClassifier.isInitialized ? Colors.green : Colors.orange,
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
  
  String _getExerciseDisplayName(String exercise) {
    switch (exercise) {
      case 'benchpress':
        return 'benchpress';
      case 'deadlift':
        return 'deadlift';
      case 'lunges':
        return 'lunges';
      case 'side_lateral_raise':
        return 'side_lateral_raise';
      case 'squat':
        return 'squat';
      default:
        return 'others';
    }
  }
  
  IconData _getExerciseIcon(String exercise) {
    switch (exercise) {
      case 'benchpress':
        return Icons.fitness_center;
      case 'deadlift':
        return Icons.sports_gymnastics;
      case 'lunges':
        return Icons.directions_run;
      case 'side_lateral_raise':
        return Icons.accessibility_new;
      case 'squat':
        return Icons.sports_handball;
      default:
        return Icons.help_outline;
    }
  }
  
  Color _getExerciseColor(String exercise) {
    switch (exercise) {
      case 'benchpress':
        return Colors.blue;
      case 'deadlift':
        return Colors.red;
      case 'lunges':
        return Colors.green;
      case 'side_lateral_raise':
        return Colors.orange;
      case 'squat':
        return Colors.purple;
      default:
        return Colors.grey;
    }
  }
}

class PosePainter extends CustomPainter {
  final List<Pose> poses;
  final Size previewSize;
  final CameraLensDirection lensDirection;
  final PoseClassifier? poseClassifier;

  PosePainter(this.poses, this.previewSize, this.lensDirection, {this.poseClassifier});

  @override
  void paint(Canvas canvas, Size size) {
    final paintNtuCircle = Paint()
      ..color = Colors.blue
      ..strokeWidth = 3
      ..style = PaintingStyle.fill;

    final paintLine = Paint()
      ..color = Colors.green
      ..strokeWidth = 2;

    for (final pose in poses) {
      final landmarks = pose.landmarks;
      
      // NTU 25개 관절점 표시 (큰 원과 번호)
      if (poseClassifier != null && landmarks.isNotEmpty) {
        final landmarkList = landmarks.values.toList();
        final ntuJoints = poseClassifier!.getNtuJoints(landmarkList);
        
        // 관절점 좌표 저장
        List<Offset> jointOffsets = [];
        
        for (int i = 0; i < ntuJoints.length; i++) {
          final joint = ntuJoints[i];
          // NTU 관절점을 MediaPipe 좌표계로 변환하여 화면에 표시
          final double x = joint[0] * size.width / previewSize.height;
          final double y = joint[1] * size.height / previewSize.width;

        var finalOffset = Offset(x, y);
        if (lensDirection == CameraLensDirection.front) {
          finalOffset = Offset(size.width - x, y);
          }
          
          jointOffsets.add(finalOffset);
          
          canvas.drawCircle(finalOffset, 8, paintNtuCircle);
        }
        
        // 허리 중앙선 그리기 (관절점 0: spine_base, 1: spine_mid, 2: neck, 3: head)
        if (jointOffsets.length >= 4) {
          final spinePaint = Paint()
            ..color = Colors.green
            ..strokeWidth = 3;
          
          // spine_base(0) -> spine_mid(1) -> neck(2) -> head(3)
          canvas.drawLine(jointOffsets[0], jointOffsets[1], spinePaint);
          canvas.drawLine(jointOffsets[1], jointOffsets[2], spinePaint);
          canvas.drawLine(jointOffsets[2], jointOffsets[3], spinePaint);
        }
        
        // 팔과 다리 선 그리기
        if (jointOffsets.length >= 25) {
          final limbPaint = Paint()
            ..color = Colors.green
            ..strokeWidth = 2;
          
          // 왼쪽 팔: shoulder(4) -> elbow(5) -> wrist(6) -> hand(7)
          canvas.drawLine(jointOffsets[4], jointOffsets[5], limbPaint);
          canvas.drawLine(jointOffsets[5], jointOffsets[6], limbPaint);
          canvas.drawLine(jointOffsets[6], jointOffsets[7], limbPaint);
          
          // 왼쪽 손가락 연결: hand(7) -> hand_tip(21) -> thumb(22)
          canvas.drawLine(jointOffsets[7], jointOffsets[21], limbPaint);
          canvas.drawLine(jointOffsets[21], jointOffsets[22], limbPaint);
          
          // 오른쪽 팔: shoulder(8) -> elbow(9) -> wrist(10) -> hand(11)
          canvas.drawLine(jointOffsets[8], jointOffsets[9], limbPaint);
          canvas.drawLine(jointOffsets[9], jointOffsets[10], limbPaint);
          canvas.drawLine(jointOffsets[10], jointOffsets[11], limbPaint);
          
          // 오른쪽 손가락 연결: hand(11) -> hand_tip(23) -> thumb(24)
          canvas.drawLine(jointOffsets[11], jointOffsets[23], limbPaint);
          canvas.drawLine(jointOffsets[23], jointOffsets[24], limbPaint);
          
          // 왼쪽 다리: hip(12) -> knee(13) -> ankle(14) -> foot(15)
          canvas.drawLine(jointOffsets[12], jointOffsets[13], limbPaint);
          canvas.drawLine(jointOffsets[13], jointOffsets[14], limbPaint);
          canvas.drawLine(jointOffsets[14], jointOffsets[15], limbPaint);
          
          // 왼쪽 발가락 연결 (발 끝부분 강조)
          // foot(15)에서 발가락 방향으로 작은 선 추가
          if (jointOffsets.length > 15) {
            final footPaint = Paint()
              ..color = Colors.green
              ..strokeWidth = 1.5;
            // 발 끝에서 앞쪽으로 작은 선 그리기
            final footEnd = jointOffsets[15];
            final toeEnd = Offset(footEnd.dx + 5, footEnd.dy - 3);
            canvas.drawLine(footEnd, toeEnd, footPaint);
          }
          
          // 오른쪽 다리: hip(16) -> knee(17) -> ankle(18) -> foot(19)
          canvas.drawLine(jointOffsets[16], jointOffsets[17], limbPaint);
          canvas.drawLine(jointOffsets[17], jointOffsets[18], limbPaint);
          canvas.drawLine(jointOffsets[18], jointOffsets[19], limbPaint);
          
          // 오른쪽 발가락 연결 (발 끝부분 강조)
          // foot(19)에서 발가락 방향으로 작은 선 추가
          if (jointOffsets.length > 19) {
            final footPaint = Paint()
              ..color = Colors.green
              ..strokeWidth = 1.5;
            // 발 끝에서 앞쪽으로 작은 선 그리기
            final footEnd = jointOffsets[19];
            final toeEnd = Offset(footEnd.dx + 5, footEnd.dy - 3);
            canvas.drawLine(footEnd, toeEnd, footPaint);
          }
          
          // 어깨 연결: left_shoulder(4) -> right_shoulder(8)
          canvas.drawLine(jointOffsets[4], jointOffsets[8], limbPaint);
          
          // 엉덩이 연결: left_hip(12) -> right_hip(16)
          canvas.drawLine(jointOffsets[12], jointOffsets[16], limbPaint);
        }
      }

      void drawLine(PoseLandmarkType a, PoseLandmarkType b) {
        final p1 = landmarks[a];
        final p2 = landmarks[b];
        if (p1 != null && p2 != null) {
          final double x1 = p1.x * size.width / previewSize.height;
          final double y1 = p1.y * size.height / previewSize.width;
          final double x2 = p2.x * size.width / previewSize.height;
          final double y2 = p2.y * size.height / previewSize.width;

          var p1Offset = Offset(x1, y1);
          var p2Offset = Offset(x2, y2);

          if (lensDirection == CameraLensDirection.front) {
            p1Offset = Offset(size.width - x1, y1);
            p2Offset = Offset(size.width - x2, y2);
          }

          canvas.drawLine(p1Offset, p2Offset, paintLine);
        }
      }

      drawLine(PoseLandmarkType.leftShoulder, PoseLandmarkType.rightShoulder);
      drawLine(PoseLandmarkType.rightShoulder, PoseLandmarkType.rightHip);
      drawLine(PoseLandmarkType.rightHip, PoseLandmarkType.leftHip);
      drawLine(PoseLandmarkType.leftHip, PoseLandmarkType.leftShoulder);

      drawLine(PoseLandmarkType.leftShoulder, PoseLandmarkType.leftElbow);
      drawLine(PoseLandmarkType.leftElbow, PoseLandmarkType.leftWrist);
      drawLine(PoseLandmarkType.rightShoulder, PoseLandmarkType.rightElbow);
      drawLine(PoseLandmarkType.rightElbow, PoseLandmarkType.rightWrist);
      drawLine(PoseLandmarkType.leftHip, PoseLandmarkType.leftKnee);
      drawLine(PoseLandmarkType.leftKnee, PoseLandmarkType.leftAnkle);
      drawLine(PoseLandmarkType.rightHip, PoseLandmarkType.rightKnee);
      drawLine(PoseLandmarkType.rightKnee, PoseLandmarkType.rightAnkle);
    }
  }

  @override
  bool shouldRepaint(covariant PosePainter oldDelegate) => true;
}