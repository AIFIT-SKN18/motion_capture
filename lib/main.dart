import 'package:flutter/material.dart';
import 'exercise_selection_screen.dart';
import 'video_claselection_screen.dart';
import 'video_classification_screen.dart';

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
      home: MainMenuScreen(),
    );
  }
}

class MainMenuScreen extends StatelessWidget {
  const MainMenuScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: Text('AIFit - 운동 자세 인식'),
        backgroundColor: Colors.black87,
        foregroundColor: Colors.white,
        centerTitle: true,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // 앱 로고/제목
            Container(
              padding: EdgeInsets.all(20),
              child: Column(
                children: [
                  Icon(
                    Icons.fitness_center,
                    size: 80,
                    color: Colors.blue,
                  ),
                  SizedBox(height: 16),
                  Text(
                    'AIFit',
                    style: TextStyle(
                      fontSize: 48,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  Text(
                    'AI 기반 운동 자세 인식',
                    style: TextStyle(
                      fontSize: 18,
                      color: Colors.white70,
                    ),
                  ),
                ],
              ),
            ),
            
            SizedBox(height: 60),
            
            // 메뉴 버튼들
            _buildMenuButton(
              context,
              '운동 훈련',
              '5가지 운동을 선택하고 실시간 자세 감지로 훈련하세요',
              Icons.sports_gymnastics,
              Colors.blue,
              () => Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => ExerciseSelectionScreen()),
              ),
            ),
            
            SizedBox(height: 20),
            
            _buildMenuButton(
              context,
              '동영상 분류',
              '갤러리의 동영상을 선택하여 운동을 분류합니다',
              Icons.video_library,
              Colors.orange,
              () => Navigator.push(
                context,
                MaterialPageRoute(builder: (context) => VideoClassificationScreen()),
              ),
            ),
            
            SizedBox(height: 40),
            
            // 앱 정보
            Container(
              padding: EdgeInsets.all(16),
              child: Text(
                '지원 운동: 벤치프레스, 데드리프트, 런지, 사이드 레이즈, 스쿼트',
                style: TextStyle(
                  color: Colors.white54,
                  fontSize: 14,
                ),
                textAlign: TextAlign.center,
              ),
            ),
          ],
        ),
      ),
    );
  }
  
  Widget _buildMenuButton(
    BuildContext context,
    String title,
    String subtitle,
    IconData icon,
    Color color,
    VoidCallback onTap,
  ) {
    return Container(
      width: 300,
      child: Card(
        color: Colors.grey[900],
        elevation: 8,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(16),
          child: Padding(
            padding: EdgeInsets.all(20),
            child: Row(
              children: [
                Container(
                  padding: EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(
                    icon,
                    color: color,
                    size: 32,
                  ),
                ),
                SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        title,
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      SizedBox(height: 4),
                      Text(
                        subtitle,
                        style: TextStyle(
                          color: Colors.white70,
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ),
                ),
                Icon(
                  Icons.arrow_forward_ios,
                  color: Colors.white54,
                  size: 16,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
