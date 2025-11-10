
# AIFit - AI 운동 자세 분석 애플리케이션

AIFit은 사용자의 운동 자세를 AI를 통해 분석하고 피드백을 제공하는 Flutter 기반의 모바일 애플리케이션입니다. Django로 구축된 백엔드 서버와 연동하여 사용자 인증, 데이터 관리 등을 처리합니다.

## 주요 기능

- **사용자 인증**: 이메일 기반 회원가입, 로그인, 로그아웃, 비밀번호 변경, 회원 탈퇴 기능
- **운동 자세 분석**: 사용자가 업로드한 운동 동영상을 분석하여 자세 정확도를 평가
- **결과 확인**: 운동 자세 분석 결과 확인
- **보안**: JWT를 사용한 안전한 사용자 인증 및 API 접근 제어

## 기술 스택

### 프론트엔드 (Flutter)

- **언어**: Dart
- **프레임워크**: Flutter
- **상태 관리**: Provider
- **HTTP 통신**: Dio
- **로컬 저장소**: flutter_secure_storage, shared_preferences
- **UI**: Material Design
- **머신러닝**: google_mlkit_pose_detection, tflite_flutter (네이티브 구현)
- **기타**: camera, image_picker, video_player, permission_handler

### 백엔드 (Django)

- **언어**: Python
- **프레임워크**: Django, Django REST Framework
- **데이터베이스**: MySQL
- **인증**: djangorestframework-simplejwt (JWT)
- **비밀번호 해싱**: Argon2
- **기타**: django-cors-headers, python-dotenv, django-ratelimit

## 프로젝트 구조

### 최상위 디렉토리

```
.
├── android/            # Android 네이티브 프로젝트
├── backend/            # Django 백엔드 서버
├── build/              # 빌드 결과물
├── ios/                # iOS 네이티브 프로젝트
├── lib/                # Flutter 애플리케이션 소스 코드
├── assets/             # 모델 파일 등 정적 에셋
├── pubspec.yaml        # Flutter 프로젝트 설정 및 의존성 관리
└── README.md           # 프로젝트 설명 파일
```

### `lib` 디렉토리 구조

Flutter 애플리케이션의 핵심 로직이 담겨있습니다.

```
lib/
├── main.dart               # 앱의 시작점
├── config/                 # API 설정 등 앱의 전반적인 설정
│   └── api_config.dart
├── models/                 # 데이터 모델 (예: User)
│   └── user_model.dart
├── providers/              # 상태 관리를 위한 Provider
│   └── auth_provider.dart
├── screens/                # 각 화면을 구성하는 위젯
│   ├── auth/               # 인증 관련 화면 (로그인, 회원가입)
│   ├── home_screen.dart    # 메인 화면
│   └── ...
├── services/               # 비즈니스 로직 (API 호출, 인증 처리 등)
│   ├── api_service.dart    # 백엔드 API 통신
│   ├── auth_service.dart   # 인증 관련 로직
│   └── ...
├── utils/                  # 유틸리티 함수 (유효성 검사 등)
│   └── validators.dart
└── widgets/                # 공통적으로 사용되는 위젯
```

### `backend` 디렉토리 구조

Django 백엔드 서버의 소스 코드입니다.

```
backend/
├── accounts/           # 사용자 인증(accounts) 앱
│   ├── models.py       # Django 모델
│   ├── serializers.py  # 데이터 직렬화
│   ├── views.py        # API 뷰
│   ├── urls.py         # URL 라우팅
│   └── ...
├── config/             # Django 프로젝트 설정
│   ├── settings.py     # 메인 설정
│   └── urls.py         # 최상위 URL 라우팅
├── manage.py           # Django 관리 스크립트
└── requirements.txt    # Python 의존성 목록
```

## 시작하기

### 1. 프로젝트 클론

```bash
git clone https://github.com/your-repository/AIfit_login.git
cd AIfit_login
```

### 2. 프론트엔드 설정

- Flutter SDK 설치
- `pubspec.yaml` 파일에 명시된 의존성 설치

```bash
flutter pub get
```

- 에뮬레이터 또는 실제 기기에서 앱 실행

```bash
flutter run
```

### 3. 백엔드 설정

- Python 및 pip 설치
- 가상 환경 생성 및 활성화

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
```

- `requirements.txt` 파일에 명시된 의존성 설치

```bash
pip install -r requirements.txt
```

- `.env` 파일 설정 (DB 정보, SECRET_KEY 등)
- 데이터베이스 마이그레이션

```bash
python manage.py migrate
```

- 개발 서버 실행

```bash
python manage.py runserver
```

## API 엔드포인트

- **`POST /api/auth/register/start/`**: 회원가입 1단계 (이메일 입력 및 OTP 발송)
- **`POST /api/auth/register/verify/`**: 회원가입 2단계 (OTP 검증)
- **`POST /api/auth/register/complete/`**: 회원가입 3단계 (프로필 입력 및 계정 생성)
- **`POST /api/auth/login/`**: 로그인 (Access/Refresh 토큰 발급)
- **`POST /api/auth/token/refresh/`**: Access 토큰 갱신
- **`GET /api/auth/profile/`**: 사용자 프로필 조회
- **`POST /api/auth/change-password/`**: 비밀번호 변경
- **`DELETE /api/auth/delete-account/`**: 계정 삭제
- **`POST /api/auth/logout/`**: 로그아웃 (Refresh 토큰 블랙리스트)
