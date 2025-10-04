import 'dart:io';
import 'package:flutter/foundation.dart';

/// RAM 사용량 모니터링 유틸리티
class MemoryMonitor {
  static void printMemoryUsage(String stage) {
    if (kDebugMode) {
      try {
        // Android/iOS에서 메모리 사용량 확인
        final process = ProcessInfo.currentRss;
        final memoryMB = process / 1024 / 1024;
        print('[$stage] RAM 사용량: ${memoryMB.toStringAsFixed(1)} MB');
      } catch (e) {
        print('[$stage] RAM 사용량 확인 실패: $e');
      }
    }
  }
  
  static double getCurrentMemoryUsage() {
    try {
      final process = ProcessInfo.currentRss;
      return process / 1024 / 1024; // MB 단위
    } catch (e) {
      return 0.0;
    }
  }
}
