# Flutter 앱 회원가입/로그인 구현 완료

## ✅ 완료된 작업

### 백엔드 (Django)
- ✅ Docker Compose 환경 구성
- ✅ Django REST Framework + SimpleJWT
- ✅ Gmail SMTP 이메일 인증 (OTP)
- ✅ Argon2 비밀번호 해싱
- ✅ MySQL 데이터베이스
- ✅ API 엔드포인트 (회원가입 3단계, 로그인, 프로필)

### 프론트엔드 (Flutter)
- ✅ Dio HTTP 클라이언트 + Interceptor (자동 토큰 갱신)
- ✅ flutter_secure_storage (토큰 안전 저장)
- ✅ Provider 상태 관리
- ✅ 회원가입 3단계 UI (이메일 → OTP → 프로필)
- ✅ 로그인 UI
- ✅ 인증 가드 (자동 로그인)

---

## 🚀 실행 방법

### 1단계: 백엔드 실행

```bash
# backend 디렉토리로 이동
cd backend

# Docker Compose 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f backend
```

**Django 서버**: http://localhost:8000

---

### 2단계: Flutter 앱 실행

```bash
# 프로젝트 루트로 이동
cd c:\Users\Playdata\StudioProjects\AIfit_login

# 패키지 설치
flutter pub get

# 안드로이드 에뮬레이터 또는 실제 기기 연결 확인
flutter devices

# 앱 실행
flutter run
```

---

## 📱 앱 사용 흐름

### 1. 첫 실행 (로그인 화면)
- 로그인 화면이 표시됩니다
- 계정이 없으면 "회원가입" 버튼 클릭

### 2. 회원가입 (3단계)

**1단계: 이메일 입력**
- Gmail 주소 입력 (예: test@gmail.com)
- "인증 코드 발송" 클릭
- Gmail에서 6자리 OTP 확인

**2단계: OTP 입력**
- Gmail로 받은 6자리 숫자 입력
- "인증 확인" 클릭

**3단계: 프로필 입력**
- 비밀번호 (소문자+숫자+특수기호, 8자 이상)
- 비밀번호 확인
- 이름
- 성별 (남성/여성)
- 생년월일 (YYMMDD 형식, 예: 020124)
- "회원가입 완료" 클릭

### 3. 자동 로그인
- 회원가입 완료 후 자동으로 로그인됩니다
- 홈 화면으로 이동

### 4. 재실행 시
- 앱을 종료하고 재실행해도 로그인 유지
- 토큰이 유효하면 자동 로그인

---

## 🔧 주요 파일 구조

```
lib/
├── config/
│   └── api_config.dart              # API URL 설정
├── models/
│   └── user_model.dart              # 사용자 모델
├── services/
│   ├── api_service.dart             # Dio HTTP 클라이언트
│   ├── auth_service.dart            # 인증 API 호출
│   └── storage_service.dart         # 토큰 저장/로드
├── providers/
│   └── auth_provider.dart           # 인증 상태 관리
├── screens/
│   ├── auth/
│   │   ├── login_screen.dart        # 로그인 화면
│   │   ├── signup_email_screen.dart # 회원가입 1단계
│   │   ├── signup_otp_screen.dart   # 회원가입 2단계
│   │   └── signup_form_screen.dart  # 회원가입 3단계
│   └── home_screen.dart             # 홈 화면 (기존)
├── utils/
│   └── validators.dart              # 입력 검증
└── main.dart                         # 앱 진입점
```

---

## 🌐 API 설정 (중요!)

### 안드로이드 에뮬레이터
`lib/config/api_config.dart`:
```dart
static const String baseUrl = 'http://10.0.2.2:8000';  // 현재 설정
```

### iOS 시뮬레이터
```dart
static const String baseUrl = 'http://localhost:8000';
```

### 실제 기기
```dart
static const String baseUrl = 'http://192.168.0.10:8000';  // PC의 IP 주소
```

**PC의 IP 주소 확인:**
```bash
# Windows
ipconfig

# 무선 LAN 어댑터 Wi-Fi의 IPv4 주소 확인
```

---

## 🧪 테스트 시나리오

### 1. 회원가입 테스트
1. 앱 실행 → 로그인 화면
2. "회원가입" 버튼 클릭
3. Gmail 주소 입력 → "인증 코드 발송"
4. Gmail 앱 확인 (6자리 OTP)
5. OTP 입력 → "인증 확인"
6. 프로필 입력:
   - 비밀번호: `Test1234!`
   - 이름: `홍길동`
   - 성별: `남성`
   - 생년월일: `020124`
7. "회원가입 완료" → 자동 로그인 → 홈 화면

### 2. 로그인 테스트
1. 앱 종료 후 재실행 → 자동 로그인 (홈 화면)
2. 또는 로그아웃 후:
   - 이메일: `test@gmail.com`
   - 비밀번호: `Test1234!`
   - "로그인" 클릭 → 홈 화면

### 3. 토큰 자동 갱신 테스트
- Access Token은 15분 후 자동 만료
- API 요청 시 401 에러 발생 → 자동으로 Refresh Token으로 갱신
- 사용자는 아무것도 느끼지 못함 (자동 처리)

---

## 📊 DBeaver로 데이터 확인

1. DBeaver 실행
2. MySQL 연결:
   - Host: localhost
   - Port: 3306
   - Database: aifit_db
   - Username: root
   - Password: rootpassword123

3. 테이블 확인:
   - **users**: 회원가입한 사용자
   - **email_verifications**: OTP 발송 기록
   - **token_blacklist_outstandingtoken**: 발급된 Refresh Token
   - **token_blacklist_blacklistedtoken**: 만료된 Token

---

## ⚠️ 문제 해결

### 1. 이메일이 발송되지 않아요
- backend/.env 파일의 EMAIL_HOST_USER와 EMAIL_HOST_PASSWORD 확인
- Gmail 앱 비밀번호를 정확히 입력했는지 확인
- Docker 로그 확인: `docker-compose logs backend`

### 2. 네트워크 연결 오류
- 백엔드 서버가 실행 중인지 확인: http://localhost:8000
- `lib/config/api_config.dart`의 baseUrl 확인
- 안드로이드 에뮬레이터: `10.0.2.2:8000`
- 실제 기기: PC의 IP 주소 사용

### 3. 토큰 갱신 오류
- 토큰을 강제 삭제하고 재로그인:
  ```dart
  // 임시 테스트용
  await StorageService.clearAll();
  ```

### 4. 패키지 설치 오류
```bash
flutter clean
flutter pub get
```

---

## 🎉 다음 단계

회원가입/로그인이 완료되었으므로, 이제 다음 기능을 추가할 수 있습니다:

1. **프로필 수정** (이름, 성별, 생년월일 변경)
2. **비밀번호 변경**
3. **로그아웃 기능** (HomeScreen에 추가)
4. **운동 기록과 사용자 연동** (현재 운동 분석 기능과 통합)
5. **대시보드** (사용자별 운동 통계)

---

## 📚 참고 자료

- Django REST Framework: https://www.django-rest-framework.org/
- SimpleJWT: https://django-rest-framework-simplejwt.readthedocs.io/
- Flutter Provider: https://pub.dev/packages/provider
- Dio: https://pub.dev/packages/dio
- Flutter Secure Storage: https://pub.dev/packages/flutter_secure_storage
