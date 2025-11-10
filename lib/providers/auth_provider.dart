import 'package:flutter/material.dart';
import '../models/user_model.dart';
import '../services/auth_service.dart';
import '../services/storage_service.dart';

class AuthProvider with ChangeNotifier {
  final AuthService _authService = AuthService();

  User? _user;
  bool _isLoading = true;
  bool _isAuthenticated = false;

  User? get user => _user;
  bool get isLoading => _isLoading;
  bool get isAuthenticated => _isAuthenticated;

  AuthProvider() {
    _checkLoginStatus();
  }

  // 로그인 상태 확인 (앱 시작 시)
  Future<void> _checkLoginStatus() async {
    _isLoading = true;
    notifyListeners();

    try {
      final loggedIn = await StorageService.isLoggedIn();
      if (loggedIn) {
        // 프로필 조회
        _user = await _authService.getProfile();
        _isAuthenticated = true;
      } else {
        _isAuthenticated = false;
      }
    } catch (e) {
      print('로그인 상태 확인 실패: $e');
      _isAuthenticated = false;
      await StorageService.clearAll();
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  // 회원가입 1단계: OTP 발송
  Future<String> registerStart(String email) async {
    return await _authService.registerStart(email);
  }

  // 회원가입 2단계: OTP 검증
  Future<String> registerVerify(String email, String otp) async {
    return await _authService.registerVerify(email, otp);
  }

  // 회원가입 3단계: 계정 생성
  Future<User> registerComplete({
    required String email,
    required String password,
    required String passwordConfirm,
    required String name,
    required String gender,
    required String birthDate,
  }) async {
    final user = await _authService.registerComplete(
      email: email,
      password: password,
      passwordConfirm: passwordConfirm,
      name: name,
      gender: gender,
      birthDate: birthDate,
    );

    _user = user;
    notifyListeners();

    return user;
  }

  // 로그인
  Future<void> login(String email, String password) async {
    await _authService.login(email, password);

    // 프로필 조회
    _user = await _authService.getProfile();
    _isAuthenticated = true;
    notifyListeners();
  }

  // 로그아웃
  Future<void> logout() async {
    await _authService.logout();
    _user = null;
    _isAuthenticated = false;
    notifyListeners();
  }

  // 프로필 새로고침
  Future<void> refreshProfile() async {
    try {
      _user = await _authService.getProfile();
      notifyListeners();
    } catch (e) {
      print('프로필 새로고침 실패: $e');
    }
  }

  // 비밀번호 변경
  Future<String> changePassword({
    required String oldPassword,
    required String newPassword,
    required String newPasswordConfirm,
  }) async {
    return await _authService.changePassword(
      oldPassword: oldPassword,
      newPassword: newPassword,
      newPasswordConfirm: newPasswordConfirm,
    );
  }

  // 계정 삭제
  Future<String> deleteAccount(String password) async {
    final message = await _authService.deleteAccount(password);
    _user = null;
    _isAuthenticated = false;
    notifyListeners();
    return message;
  }
}
