import 'package:dio/dio.dart';
import '../models/user_model.dart';
import 'api_service.dart';
import 'storage_service.dart';

class AuthService {
  final ApiService _apiService = ApiService();

  // 회원가입 1단계: 이메일 입력 → OTP 발송
  Future<String> registerStart(String email) async {
    try {
      final response = await _apiService.post(
        '/api/auth/register/start/',
        data: {'email': email},
      );

      return response.data['message'];
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  // 회원가입 2단계: OTP 검증
  Future<String> registerVerify(String email, String otp) async {
    try {
      final response = await _apiService.post(
        '/api/auth/register/verify/',
        data: {
          'email': email,
          'otp': otp,
        },
      );

      return response.data['message'];
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  // 회원가입 3단계: 프로필 입력 및 계정 생성
  Future<User> registerComplete({
    required String email,
    required String password,
    required String passwordConfirm,
    required String name,
    required String gender,
    required String birthDate,
  }) async {
    try {
      final response = await _apiService.post(
        '/api/auth/register/complete/',
        data: {
          'email': email,
          'password': password,
          'password_confirm': passwordConfirm,
          'name': name,
          'gender': gender,
          'birth_date': birthDate,
        },
      );

      return User.fromJson(response.data['user']);
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  // 로그인
  Future<Map<String, dynamic>> login(String email, String password) async {
    try {
      final response = await _apiService.post(
        '/api/auth/login/',
        data: {
          'email': email,
          'password': password,
        },
      );

      // 토큰 저장
      final accessToken = response.data['access'];
      final refreshToken = response.data['refresh'];

      await StorageService.saveTokens(accessToken, refreshToken);
      await StorageService.saveUserEmail(email);

      return {
        'access': accessToken,
        'refresh': refreshToken,
      };
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  // 프로필 조회
  Future<User> getProfile() async {
    try {
      final response = await _apiService.get('/api/auth/profile/');
      return User.fromJson(response.data);
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  // 로그아웃
  Future<void> logout() async {
    await StorageService.clearAll();
  }

  // 비밀번호 변경
  Future<String> changePassword({
    required String oldPassword,
    required String newPassword,
    required String newPasswordConfirm,
  }) async {
    try {
      final response = await _apiService.post(
        '/api/auth/change-password/',
        data: {
          'old_password': oldPassword,
          'new_password': newPassword,
          'new_password_confirm': newPasswordConfirm,
        },
      );

      return response.data['message'];
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  // 계정 삭제
  Future<String> deleteAccount(String password) async {
    try {
      final response = await _apiService.delete(
        '/api/auth/delete-account/',
        data: {'password': password},
      );

      // 삭제 후 로컬 데이터 클리어
      await StorageService.clearAll();

      return response.data['message'];
    } on DioException catch (e) {
      throw _handleError(e);
    }
  }

  // 에러 핸들링
  String _handleError(DioException error) {
    if (error.response != null) {
      final data = error.response!.data;

      // Django REST Framework 에러 형식 처리
      if (data is Map<String, dynamic>) {
        // 필드별 에러 처리
        if (data.containsKey('error')) {
          return data['error'];
        }

        // 첫 번째 에러 메시지 반환
        final firstError = data.values.firstWhere(
          (value) => value != null,
          orElse: () => '알 수 없는 오류가 발생했습니다.',
        );

        if (firstError is List && firstError.isNotEmpty) {
          return firstError[0].toString();
        }

        return firstError.toString();
      }

      return data.toString();
    }

    // 네트워크 오류
    if (error.type == DioExceptionType.connectionTimeout ||
        error.type == DioExceptionType.receiveTimeout) {
      return '서버 연결 시간이 초과되었습니다.';
    }

    if (error.type == DioExceptionType.connectionError) {
      return '네트워크 연결을 확인해주세요.';
    }

    return '알 수 없는 오류가 발생했습니다.';
  }
}
