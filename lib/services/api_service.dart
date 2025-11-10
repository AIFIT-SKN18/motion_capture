import 'package:dio/dio.dart';
import '../config/api_config.dart';
import 'storage_service.dart';

class ApiService {
  late final Dio _dio;

  ApiService() {
    _dio = Dio(
      BaseOptions(
        baseUrl: ApiConfig.baseUrl,
        connectTimeout: ApiConfig.connectTimeout,
        receiveTimeout: ApiConfig.receiveTimeout,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
      ),
    );

    // Interceptor 추가 (토큰 자동 첨부 및 갱신)
    _dio.interceptors.add(
      InterceptorsWrapper(
        onRequest: (options, handler) async {
          // Access Token 자동 첨부
          final token = await StorageService.getAccessToken();
          if (token != null) {
            options.headers['Authorization'] = 'Bearer $token';
          }
          return handler.next(options);
        },
        onError: (error, handler) async {
          // 401 에러 (Unauthorized) 시 토큰 갱신
          if (error.response?.statusCode == 401) {
            final refreshToken = await StorageService.getRefreshToken();
            if (refreshToken != null) {
              // 토큰 갱신 시도
              final refreshed = await _refreshToken(refreshToken);
              if (refreshed) {
                // 원래 요청 재시도
                final retryResponse = await _retry(error.requestOptions);
                return handler.resolve(retryResponse);
              }
            }
          }
          return handler.next(error);
        },
      ),
    );
  }

  // 토큰 갱신
  Future<bool> _refreshToken(String refreshToken) async {
    try {
      final response = await _dio.post(
        '/api/auth/token/refresh/',
        data: {'refresh': refreshToken},
      );

      if (response.statusCode == 200) {
        final newAccessToken = response.data['access'];
        final newRefreshToken = response.data['refresh'];

        // 새 토큰 저장
        await StorageService.saveTokens(newAccessToken, newRefreshToken);
        return true;
      }
      return false;
    } catch (e) {
      print('토큰 갱신 실패: $e');
      return false;
    }
  }

  // 요청 재시도
  Future<Response> _retry(RequestOptions requestOptions) async {
    final token = await StorageService.getAccessToken();

    final options = Options(
      method: requestOptions.method,
      headers: {
        ...requestOptions.headers,
        'Authorization': 'Bearer $token',
      },
    );

    return _dio.request(
      requestOptions.path,
      data: requestOptions.data,
      queryParameters: requestOptions.queryParameters,
      options: options,
    );
  }

  // GET 요청
  Future<Response> get(String path, {Map<String, dynamic>? queryParameters}) async {
    return await _dio.get(path, queryParameters: queryParameters);
  }

  // POST 요청
  Future<Response> post(String path, {dynamic data}) async {
    return await _dio.post(path, data: data);
  }

  // PUT 요청
  Future<Response> put(String path, {dynamic data}) async {
    return await _dio.put(path, data: data);
  }

  // DELETE 요청
  Future<Response> delete(String path, {dynamic data}) async {
    return await _dio.delete(path, data: data);
  }
}
