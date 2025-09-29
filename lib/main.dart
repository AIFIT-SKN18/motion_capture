import 'package:flutter/material.dart';
import 'pose_detection_screen.dart';

void main() {
  print('🚀 AIFit 운동 자세 인식 앱 시작!');
  print('📱 Flutter 앱이 시작되었습니다.');
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AIFit',
      theme: ThemeData.dark(),
      home: PoseDetectionScreen(),
    );
  }
}
