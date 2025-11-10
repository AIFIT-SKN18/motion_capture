## 회원가입·로그인 & LLM 기록 저장 — Flutter + **Django(DRF + SimpleJWT)** + MySQL + Gmail SMTP

---

## 필요한 기능(What to build)

#### 회원가입(Sign-up)
- 이메일(구글 메일) 입력 → **인증코드(OTP, 6자리, 5분 만료)** 전송·확인
- 비밀번호 규칙 체크(**소문자+숫자+특수기호, 8자 이상**) → **해시(Hash, Argon2id) 저장** *(평문 금지)*[^owasp-password][^argon2]
- 이름/성별/생년월일 저장(**YYMMDD → DATE** 변환)

#### 로그인(Sign-in)
- 이메일 + 비밀번호 확인 → **JWT(Access/Refresh) 발급·보관** *(SimpleJWT 사용, 응답 키: `access`, `refresh`)*[^simplejwt]

#### 보안/운영(Security/Ops)
- **OTP 5분 만료**, 6자리
- **OTP 전송/로그인 시도 횟수 제한(Rate limit)**
- 토큰은 앱 내 **Secure Storage** 보관
- 이메일 발송은 **Gmail SMTP(587/STARTTLS)** + **앱 비밀번호(App Password)**[^gmail-app-pass][^django-email]

---

## 만들 순서(How to build)

#### 1) DB 만들기(테이블 준비)
- **users** : 이메일(아이디), 비번해시, 이름, 성별(M/F), 생년월일, 이메일 인증시각
- **email_verifications** : 이메일별 **OTP(해시 저장)**, 만료시각, 사용여부(consumed)
- **refresh_tokens(선택)** : Refresh 토큰 **화이트리스트/해시**로 엄격 관리 시 사용  
  - *또는* **SimpleJWT Blacklist** 앱으로 **회전/블랙리스트** 관리[^simplejwt]
- **chat_messages** : 사용자별 **질문/답변/모델명/시간**
- **DBeaver**로 접속 → 테이블 생성·스키마를 먼저 확인

#### 2) Gmail 준비(이메일 보내기)
- Gmail **2단계 인증** 켜기 → **앱 비밀번호(App Password)** 발급[^gmail-app-pass]
- 서버에서 **SMTP**(`smtp.gmail.com`, **포트 587**, **STARTTLS**)로 메일 전송
- 메일 제목/본문: “인증코드 6자리, 5분 내 입력 안내”
- **Django Email Backend** 설정으로 `send_mail()` 사용[^django-email]

#### 3) 서버(API) 만들기 — **Django REST Framework + SimpleJWT**
- **핵심 엔드포인트**
  - `POST /auth/register/start`  
    - 이메일 중복 확인  
    - **OTP 6자리 생성 → 해시 저장(5분 만료)**  
    - **Gmail SMTP**로 코드 발송
  - `POST /auth/register/verify`  
    - 이메일 + OTP로 **검증/사용 처리(consumed)**
  - `POST /auth/register/complete`  
    - 비번 규칙 재확인 → **해시(Argon2id) 저장** → 프로필 저장
  - `POST /auth/login` *(SimpleJWT TokenObtainPairView)*  
    - 이메일 + 비번 확인 → **JWT 발급(응답 키: `access`, `refresh`)**
  - `POST /auth/token/refresh` *(TokenRefreshView)*  
    - **Refresh로 Access 재발급(회전/블랙리스트 활성화 권장)**[^simplejwt]
  - `POST /chat/save`  
    - (로그인된) 사용자 **질문/답변 저장**
- **서버 체크포인트**
  - 비밀번호는 **항상 해시**로만 저장(평문/복호화 금지)[^owasp-password][^argon2]
  - **OTP/Refresh 토큰은 해시값만** DB 저장(화이트리스트 방식 시)
  - **CORS 허용**, 에러 메시지 포맷 통일
  - **Custom User(이메일 로그인)** + **Admin**으로 운영 편의 확보

#### 4) 앱(Flutter) 만들기
- **화면 흐름(UI)**
  - **이메일 입력 화면** → “코드 보내기” → `/auth/register/start`
  - **인증코드 입력 화면**(6자리) → `/auth/register/verify`
  - **정보 입력 화면**(비밀번호/비번확인/이름/성별/생년월일(YYMMDD)) → `/auth/register/complete`
  - **로그인 화면**(이메일+비번) → `/auth/login` → **토큰 저장** → 홈 이동
- **유효성 검사(클라이언트)**
  - 이메일: `…@gmail.com`
  - OTP: **숫자 6자리**
  - 비밀번호: **소문자+숫자+특수기호, 8자 이상**
  - 생년월일: **YYMMDD**
- **네트워킹(Networking)**
  - **Dio** 사용(HTTP)
  - **Interceptor**로 Access 자동 첨부, `401`이면 `/auth/token/refresh` 후 **원요청 재시도**
- **토큰 보관**
  - **flutter_secure_storage**에 **Access/Refresh** 저장(자동 로그인)  
  - *(주의: Django(SimpleJWT) 로그인 응답 키는 `access`/`refresh`)*

---

## 실제 동작 예시(E2E 시나리오)

#### 회원가입
- 앱에서 **이메일 입력** → 서버가 **OTP 메일 발송**
- 받은 **6자리 코드 입력** → 서버가 **검증 완료**
- **비밀번호/이름/성별/생년월일 입력** → 서버가 **계정 생성**(비번은 Argon2 해시 저장)

#### 로그인
- **이메일 + 비밀번호** 입력 → 서버가 **JWT(Access/Refresh)** 2개 발급  
- 앱이 **Secure Storage**에 토큰 보관 → 이후 요청에 **Access** 자동 첨부  
- Access 만료 시 **Interceptor**가 자동으로 **Refresh** → **새 Access**로 재시도

#### LLM 저장
- 응답이 나오면 `POST /chat/save`로 **질문/답변/모델명 저장** → **DBeaver**로 적재 확인

---

## 꼭 지켜야 할 보안 포인트(Top 5)

- **비밀번호 해시 저장(Argon2id 권장)** — **평문 금지**[^owasp-password][^argon2]
- **OTP/Refresh 토큰은 해시만 저장**(유출 대비) *또는* **SimpleJWT Blacklist**로 회전 관리[^simplejwt]
- **OTP 5분 만료**, 잘못된 코드/만료/재사용 방지
- 로그인 실패·OTP 전송 **횟수 제한(Rate limit)**
- 토큰은 앱의 **Secure Storage**에만 저장(클립보드/로그 금지)

---

## DBeaver 체크리스트

- **users** : 새 계정 생성, `password_hash`가 **해시 문자열**인지, `email_verified_at`이 **기록**되는지
- **email_verifications** : OTP 행 생성 → **consumed=1**로 바뀌는지
- **refresh_tokens / (또는) SimpleJWT Blacklist 테이블** : 로그인/갱신 시 **토큰 상태**가 정상 관리되는지
- **chat_messages** : LLM **질문/답변**이 사용자별로 저장되는지

---

## 참고(Obsidian Footnotes)

[^django-email]: Django **Email**(SMTP) 공식 문서 — `send_mail`, 이메일 백엔드 설정. <https://docs.djangoproject.com/en/stable/topics/email/>  
[^argon2]: Django **Argon2 Password Hasher** 가이드. <https://docs.djangoproject.com/en/stable/topics/auth/passwords/#using-argon2-with-django>  
[^simplejwt]: **Django REST Framework SimpleJWT** — Access/Refresh, 회전, 블랙리스트. <https://django-rest-framework-simplejwt.readthedocs.io/en/latest/>  
[^owasp-password]: OWASP **Password Storage** Cheat Sheet — 안전한 비밀번호 저장 가이드(해시·솔트·파라미터). <https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html>  
[^jwt-bcp]: **JWT** 표준 및 모범 사례(RFC 7519). <https://datatracker.ietf.org/doc/html/rfc7519>  
[^gmail-app-pass]: Gmail **앱 비밀번호** 안내(2단계 인증 필요). <https://support.google.com/accounts/answer/185833>
