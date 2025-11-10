# AIfit Backend - Django REST API

회원가입/로그인을 위한 Django REST Framework 백엔드

---

## 🚀 실행 가이드 (사용자가 해야 할 것)

### ✅ Step 1: Gmail 앱 비밀번호 발급 (필수!)

이메일 인증 OTP를 발송하기 위해 Gmail 앱 비밀번호가 필요합니다.

1. **Gmail 계정에 로그인**
2. https://myaccount.google.com/security 접속
3. **"2단계 인증"** 활성화 (아직 안 했다면)
4. 검색창에 **"앱 비밀번호"** 검색
5. **앱 선택** → "메일" 선택
6. **기기 선택** → "기타" 입력 → "Django" 입력
7. **생성** 버튼 클릭
8. **16자리 비밀번호 복사** (예: `abcd efgh ijkl mnop`)
   - 공백 제거: `abcdefghijklmnop`

---

### ✅ Step 2: .env 파일 수정

`backend/.env` 파일을 열어서 다음 두 줄을 **실제 값으로 변경**:

```env
EMAIL_HOST_USER=your-email@gmail.com        ← 본인 Gmail 주소로 변경
EMAIL_HOST_PASSWORD=abcdefghijklmnop        ← Step 1에서 발급받은 16자리 비밀번호
```

**예시:**
```env
EMAIL_HOST_USER=hongkildong@gmail.com
EMAIL_HOST_PASSWORD=xyzw1234abcd5678
```

---

### ✅ Step 3: Docker 실행

**3-1. 터미널(PowerShell 또는 CMD)을 열고 backend 디렉토리로 이동:**

```bash
cd c:\Users\Playdata\StudioProjects\AIfit_login\backend
```

**3-2. Docker Compose로 백엔드 + MySQL 실행:**

```bash
docker-compose up -d
```

**3-3. 로그 확인 (서버가 정상 실행되는지 확인):**

```bash
docker-compose logs -f backend
```

다음과 같은 메시지가 보이면 성공:
```
Starting development server at http://0.0.0.0:8000/
```

**로그 확인을 중지하려면: `Ctrl + C`**

---

### ✅ Step 4: 서버 확인

브라우저에서 다음 URL 접속:

- **Django Admin**: http://localhost:8000/admin

---

### ✅ Step 5: 관리자 계정 생성 (Django Admin 접속용)

```bash
# Django 컨테이너 접속
docker-compose exec backend python manage.py createsuperuser

# 입력 예시:
# Email: admin@gmail.com
# Name: Admin
# Password: admin1234! (8자 이상, 소문자+숫자+특수기호)
```

생성 후 http://localhost:8000/admin 에서 로그인 가능

---

### ✅ Step 6: API 테스트 (Postman 또는 Thunder Client)

#### 6-1. 회원가입 1단계: OTP 발송

```http
POST http://localhost:8000/api/auth/register/start/
Content-Type: application/json

{
    "email": "test@gmail.com"
}
```

**→ Gmail로 6자리 OTP 수신 확인!**

---

#### 6-2. 회원가입 2단계: OTP 검증

```http
POST http://localhost:8000/api/auth/register/verify/
Content-Type: application/json

{
    "email": "test@gmail.com",
    "otp": "123456"
}
```

**→ 200 OK 응답 확인**

---

#### 6-3. 회원가입 3단계: 계정 생성

```http
POST http://localhost:8000/api/auth/register/complete/
Content-Type: application/json

{
    "email": "test@gmail.com",
    "password": "Test1234!",
    "password_confirm": "Test1234!",
    "name": "홍길동",
    "gender": "M",
    "birth_date": "020124"
}
```

**→ 201 Created 응답, users 테이블에 계정 생성**

---

#### 6-4. 로그인

```http
POST http://localhost:8000/api/auth/login/
Content-Type: application/json

{
    "email": "test@gmail.com",
    "password": "Test1234!"
}
```

**응답 예시:**
```json
{
    "access": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "refresh": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

**→ access 토큰 복사 (다음 요청에 사용)**

---

#### 6-5. 프로필 조회 (인증 필요)

```http
GET http://localhost:8000/api/auth/profile/
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

**→ 사용자 정보 반환**

---

#### 6-6. 토큰 갱신

```http
POST http://localhost:8000/api/auth/token/refresh/
Content-Type: application/json

{
    "refresh": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

**→ 새로운 access 토큰 발급**

---

## 📊 DBeaver로 MySQL 확인

### 연결 정보:
- **Host**: localhost
- **Port**: 3306
- **Database**: aifit_db
- **Username**: root
- **Password**: rootpassword123

### 확인 사항:
1. **users** 테이블: 회원가입한 계정 확인
   - `password` 컬럼: Argon2 해시로 저장되어 있어야 함 (평문 금지!)
   - `email_verified_at`: 인증 시각 기록

2. **email_verifications** 테이블: OTP 발송 기록
   - `consumed`: True로 변경되었는지 확인
   - `expires_at`: 5분 만료 시간 확인

3. **token_blacklist_outstandingtoken**: 발급된 Refresh 토큰
4. **token_blacklist_blacklistedtoken**: 만료/로그아웃된 토큰

---

## 🛠️ Docker 유용한 명령어

```bash
# 서버 시작
docker-compose up -d

# 서버 중지
docker-compose down

# 로그 실시간 보기
docker-compose logs -f backend

# MySQL 로그 보기
docker-compose logs -f db

# Django 컨테이너 셸 접속
docker-compose exec backend bash

# 데이터베이스 마이그레이션 (테이블 변경 시)
docker-compose exec backend python manage.py makemigrations
docker-compose exec backend python manage.py migrate

# 전체 재시작 (데이터베이스 초기화 포함)
docker-compose down -v
docker-compose up -d
```

---

## ⚠️ 문제 해결

### 1. 이메일이 발송되지 않아요!

**확인 사항:**
- .env 파일의 `EMAIL_HOST_USER`와 `EMAIL_HOST_PASSWORD`가 올바른지 확인
- Gmail 앱 비밀번호를 정확히 입력했는지 확인 (16자리, 공백 제거)
- 2단계 인증이 활성화되어 있는지 확인
- Gmail "보안 수준이 낮은 앱" 설정은 필요 없음 (앱 비밀번호 사용)

**로그 확인:**
```bash
docker-compose logs backend | grep "이메일"
```

---

### 2. Docker 컨테이너가 시작되지 않아요!

**확인 사항:**
- Docker Desktop이 실행 중인지 확인
- 포트 충돌 확인 (3306, 8000 포트가 다른 프로그램에서 사용 중인지)

**재시작:**
```bash
docker-compose down
docker-compose up -d
```

---

### 3. 데이터베이스 연결 오류

**MySQL 컨테이너 상태 확인:**
```bash
docker-compose ps
```

**MySQL 로그 확인:**
```bash
docker-compose logs db
```

**재시작:**
```bash
docker-compose restart db
docker-compose restart backend
```

---

### 4. 마이그레이션 오류

```bash
# 마이그레이션 초기화
docker-compose exec backend python manage.py migrate --run-syncdb

# 또는 전체 재생성
docker-compose down -v
docker-compose up -d
```

---

## 📋 API 엔드포인트 목록

| 메서드 | 엔드포인트 | 설명 | 인증 |
|--------|-----------|------|------|
| POST | `/api/auth/register/start/` | 이메일 입력 → OTP 발송 | ❌ |
| POST | `/api/auth/register/verify/` | OTP 검증 | ❌ |
| POST | `/api/auth/register/complete/` | 계정 생성 | ❌ |
| POST | `/api/auth/login/` | 로그인 (JWT 발급) | ❌ |
| POST | `/api/auth/token/refresh/` | 토큰 갱신 | ❌ |
| GET | `/api/auth/profile/` | 프로필 조회 | ✅ |

---

## 🔐 보안 체크리스트

- ✅ 비밀번호: Argon2id 해싱 (평문 저장 금지)
- ✅ OTP: SHA256 해싱 (5분 만료)
- ✅ JWT: Access 15분, Refresh 30일
- ✅ Refresh Token: 블랙리스트 + 회전(Rotation)
- ✅ 비밀번호 정책: 소문자 + 숫자 + 특수기호, 8자 이상
- ✅ CORS: 허용된 도메인만 접근

---

## 📝 다음 단계

백엔드가 정상 작동하면 Flutter 앱을 구현합니다:
1. `pubspec.yaml`에 패키지 추가 (dio, flutter_secure_storage 등)
2. API 서비스 레이어 구현
3. 회원가입/로그인 UI 구현
4. 토큰 관리 및 자동 갱신
