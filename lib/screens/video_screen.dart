import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:video_player/video_player.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:video_thumbnail/video_thumbnail.dart';
import '../services/pose_detector.dart';
import '../services/pose_classifier.dart';

/// 동영상 분석 화면
class VideoScreen extends StatefulWidget {
  const VideoScreen({super.key});

  @override
  State<VideoScreen> createState() => _VideoScreenState();
}

class _VideoScreenState extends State<VideoScreen> {
  final ImagePicker _picker = ImagePicker();
  final PoseDetectorService _poseDetector = PoseDetectorService();
  final PoseClassifier _poseClassifier = PoseClassifier();

  File? _videoFile;
  VideoPlayerController? _videoController;
  bool _isAnalyzing = false;
  String? _result;
  double? _confidence;
  Map<String, double>? _allProbabilities;
  String _analysisProgress = '';

  @override
  void initState() {
    super.initState();
    _initializeServices();
  }

  @override
  void dispose() {
    _videoController?.dispose();
    _poseDetector.dispose();
    _poseClassifier.dispose();
    super.dispose();
  }

  Future<void> _initializeServices() async {
    try {
      await _poseDetector.initialize();
      await _poseClassifier.initialize();
    } catch (e) {
      _showErrorDialog('초기화 실패', e.toString());
    }
  }

  Future<void> _pickVideo() async {
    try {
      final XFile? video = await _picker.pickVideo(
        source: ImageSource.gallery,
      );

      if (video != null) {
        // 기존 컨트롤러 해제
        _videoController?.dispose();

        // 새 동영상 로드
        final videoFile = File(video.path);
        final controller = VideoPlayerController.file(videoFile);

        await controller.initialize();

        setState(() {
          _videoFile = videoFile;
          _videoController = controller;
          _result = null;
          _confidence = null;
          _allProbabilities = null;
          _analysisProgress = '';
        });
      }
    } catch (e) {
      _showErrorDialog('동영상 선택 실패', e.toString());
    }
  }

  Future<void> _analyzeVideo() async {
    if (_videoFile == null || _videoController == null) return;

    setState(() {
      _isAnalyzing = true;
      _result = null;
      _confidence = null;
      _analysisProgress = '프레임 추출 중...';
    });

    try {
      final duration = _videoController!.value.duration;
      final totalMs = duration.inMilliseconds;

      // 동영상에서 균등하게 64 프레임 추출 (길이에 상관없이 강제 64)
      const framesToExtract = 64;
      final intervalMs = totalMs / framesToExtract;

      List<List<PoseLandmark>> poseSequence = [];
      double? frameImageWidth;
      double? frameImageHeight;

      for (int i = 0; i < framesToExtract; i++) {
        final frameTimeMs = (i * intervalMs).toInt().clamp(0, totalMs - 1);

        setState(() {
          _analysisProgress = '프레임 분석 중... ${i + 1}/$framesToExtract';
        });

        // 썸네일 파일 생성 후 포즈 감지
        String? thumbPath;
        try {
          thumbPath = await VideoThumbnail.thumbnailFile(
            video: _videoFile!.path,
            imageFormat: ImageFormat.PNG,
            timeMs: frameTimeMs,
            quality: 75,
          );

          if (thumbPath != null) {
            // 썸네일 실제 크기 확보 (첫 프레임 기준)
            if (frameImageWidth == null || frameImageHeight == null) {
              try {
                final bytes = await File(thumbPath).readAsBytes();
                final codec = await ui.instantiateImageCodec(bytes);
                final fi = await codec.getNextFrame();
                frameImageWidth = fi.image.width.toDouble();
                frameImageHeight = fi.image.height.toDouble();
              } catch (_) {}
            }
            final inputImage = InputImage.fromFilePath(thumbPath);
            final poses = await _poseDetector.detectPose(inputImage);

            if (poses.isNotEmpty && poses.first.landmarks.isNotEmpty) {
              poseSequence.add(poses.first.landmarks.values.toList());
            }
          }
        } catch (e) {
          print('프레임 $i 처리 오류: $e');
        } finally {
          if (thumbPath != null) {
            try { File(thumbPath).deleteSync(); } catch (_) {}
          }
        }
      }

      if (poseSequence.isEmpty) {
        setState(() {
          _result = '포즈를 감지할 수 없습니다';
          _confidence = 0.0;
          _isAnalyzing = false;
          _analysisProgress = '';
        });
        return;
      }

      setState(() {
        _analysisProgress = '운동 동작 분류 중...';
      });

      // 포즈 시퀀스 길이를 정확히 64로 맞추기 (동일 리스트 참조로 인한 concurrent 수정)
      if (poseSequence.length < 64) {
        final needed = 64 - poseSequence.length;
        final source = List<List<PoseLandmark>>.from(poseSequence);
        for (int i = 0; i < needed; i++) {
          poseSequence.add(source[(i % source.length)]);
        }
      } else if (poseSequence.length > 64) {
        poseSequence = poseSequence.sublist(0, 64);
      }

      // 분류 실행 - 썸네일 실제 이미지 크기를 사용하여 NTU 정규화 일치
      final classificationResult = await _poseClassifier.classify(
        poseSequence,
        frameImageWidth ?? _videoController!.value.size.width,
        frameImageHeight ?? _videoController!.value.size.height,
      );

      setState(() {
        _result = _getKoreanExerciseName(classificationResult.exercise);
        _confidence = classificationResult.confidence;
        _allProbabilities = classificationResult.allProbabilities;
        _isAnalyzing = false;
        _analysisProgress = '';
      });

      // 동영상을 처음으로 되돌리기
      await _videoController!.seekTo(Duration.zero);
    } catch (e) {
      setState(() {
        _isAnalyzing = false;
        _analysisProgress = '';
      });
      _showErrorDialog('분석 실패', e.toString());
    }
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
        title: const Text('동영상 분석'),
        centerTitle: true,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // 동영상 선택 버튼
              ElevatedButton.icon(
                onPressed: _isAnalyzing ? null : _pickVideo,
                icon: const Icon(Icons.video_library),
                label: const Text('동영상 선택'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.all(16),
                ),
              ),
              const SizedBox(height: 24),

              // 선택된 동영상 정보 및 프리뷰
              if (_videoFile != null && _videoController != null) ...[
                Card(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // 동영상 프리뷰
                      if (_videoController!.value.isInitialized)
                        AspectRatio(
                          aspectRatio: _videoController!.value.aspectRatio,
                          child: VideoPlayer(_videoController!),
                        ),
                      Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              '선택된 동영상',
                              style: TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              _videoFile!.path.split('/').last,
                              style: TextStyle(
                                fontSize: 14,
                                color: Colors.grey[600],
                              ),
                            ),
                            if (_videoController!.value.isInitialized) ...[
                              const SizedBox(height: 8),
                              Text(
                                '길이: ${_videoController!.value.duration.inSeconds}초',
                                style: TextStyle(
                                  fontSize: 12,
                                  color: Colors.grey[500],
                                ),
                              ),
                            ],
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 16),

                // 분석 시작 버튼
                ElevatedButton.icon(
                  onPressed: _isAnalyzing ? null : _analyzeVideo,
                  icon: _isAnalyzing
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      : const Icon(Icons.analytics),
                  label: Text(_isAnalyzing ? '분석 중...' : '분석 시작'),
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.all(16),
                    backgroundColor: Colors.green,
                    foregroundColor: Colors.white,
                  ),
                ),

                // 분석 진행 상태
                if (_analysisProgress.isNotEmpty) ...[
                  const SizedBox(height: 16),
                  Card(
                    color: Colors.blue.shade50,
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Row(
                        children: [
                          const SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Text(
                              _analysisProgress,
                              style: const TextStyle(fontSize: 14),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ],

              // 분석 결과
              if (_result != null) ...[
                const SizedBox(height: 32),
                Card(
                  elevation: 4,
                  child: Padding(
                    padding: const EdgeInsets.all(20.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          '분석 결과',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const Divider(height: 24),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            const Text(
                              '운동 종류:',
                              style: TextStyle(fontSize: 16),
                            ),
                            Text(
                              _result!,
                              style: const TextStyle(
                                fontSize: 20,
                                fontWeight: FontWeight.bold,
                                color: Colors.blue,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            const Text(
                              '신뢰도:',
                              style: TextStyle(fontSize: 16),
                            ),
                            Text(
                              '${(_confidence! * 100).toStringAsFixed(1)}%',
                              style: const TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ),
                        if (_allProbabilities != null &&
                            _allProbabilities!.isNotEmpty) ...[
                          const SizedBox(height: 20),
                          const Text(
                            '전체 확률',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(height: 12),
                          ..._allProbabilities!.entries.map((entry) {
                            return Padding(
                              padding: const EdgeInsets.only(bottom: 8.0),
                              child: Row(
                                children: [
                                  Expanded(
                                    flex: 2,
                                    child: Text(
                                      _getKoreanExerciseName(entry.key),
                                      style: const TextStyle(fontSize: 14),
                                    ),
                                  ),
                                  Expanded(
                                    flex: 3,
                                    child: LinearProgressIndicator(
                                      value: entry.value,
                                      backgroundColor: Colors.grey[200],
                                      minHeight: 8,
                                    ),
                                  ),
                                  const SizedBox(width: 8),
                                  SizedBox(
                                    width: 50,
                                    child: Text(
                                      '${(entry.value * 100).toStringAsFixed(1)}%',
                                      style: const TextStyle(fontSize: 12),
                                      textAlign: TextAlign.right,
                                    ),
                                  ),
                                ],
                              ),
                            );
                          }).toList(),
                        ],
                      ],
                    ),
                  ),
                ),
              ],

              // 안내 메시지
              if (_videoFile == null) ...[
                const SizedBox(height: 80),
                Center(
                  child: Column(
                    children: [
                      Icon(
                        Icons.video_library_outlined,
                        size: 80,
                        color: Colors.grey[400],
                      ),
                      const SizedBox(height: 16),
                      Text(
                        '갤러리에서 운동 동영상을 선택하세요',
                        style: TextStyle(
                          fontSize: 16,
                          color: Colors.grey[600],
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 80),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
