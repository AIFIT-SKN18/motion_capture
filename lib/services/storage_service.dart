import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class StorageService {
  static const FlutterSecureStorage _storage = FlutterSecureStorage();

  // 키 정의
  static const String _accessTokenKey = 'access_token';
  static const String _refreshTokenKey = 'refresh_token';
  static const String _userEmailKey = 'user_email';

  // Access Token 저장
  static Future<void> saveAccessToken(String token) async {
    await _storage.write(key: _accessTokenKey, value: token);
  }

  // Access Token 가져오기
  static Future<String?> getAccessToken() async {
    return await _storage.read(key: _accessTokenKey);
  }

  // Refresh Token 저장
  static Future<void> saveRefreshToken(String token) async {
    await _storage.write(key: _refreshTokenKey, value: token);
  }

  // Refresh Token 가져오기
  static Future<String?> getRefreshToken() async {
    return await _storage.read(key: _refreshTokenKey);
  }

  // 사용자 이메일 저장
  static Future<void> saveUserEmail(String email) async {
    await _storage.write(key: _userEmailKey, value: email);
  }

  // 사용자 이메일 가져오기
  static Future<String?> getUserEmail() async {
    return await _storage.read(key: _userEmailKey);
  }

  // 모든 토큰 저장 (로그인 시)
  static Future<void> saveTokens(String accessToken, String refreshToken) async {
    await Future.wait([
      saveAccessToken(accessToken),
      saveRefreshToken(refreshToken),
    ]);
  }

  // 모든 토큰 및 사용자 정보 삭제 (로그아웃 시)
  static Future<void> clearAll() async {
    await _storage.deleteAll();
  }

  // 로그인 여부 확인
  static Future<bool> isLoggedIn() async {
    final accessToken = await getAccessToken();
    final refreshToken = await getRefreshToken();
    return accessToken != null && refreshToken != null;
  }
}
