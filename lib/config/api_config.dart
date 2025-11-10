class ApiConfig {
  // 안드로이드 에뮬레이터에서 localhost는 10.0.2.2
  // iOS 시뮬레이터에서는 localhost
  // 실제 기기에서는 PC의 IP 주소 (예: 192.168.0.10)

  // static const String baseUrl = 'http://10.0.2.2:8000';  // 안드로이드 에뮬레이터
  // static const String baseUrl = 'http://localhost:8000';  // iOS 시뮬레이터
  static const String baseUrl = 'http://192.168.62.126:8000';  // 실제 기기 (PC IP)

  static const Duration connectTimeout = Duration(seconds: 30);
  static const Duration receiveTimeout = Duration(seconds: 30);
}
