import secrets
import hashlib
from django.core.mail import send_mail
from django.conf import settings
from datetime import datetime, timedelta
from .models import EmailVerification


def generate_otp():
    """6자리 OTP 생성"""
    return ''.join([str(secrets.randbelow(10)) for _ in range(6)])


def hash_otp(otp):
    """OTP SHA256 해싱 (보안을 위해 평문 저장 금지)"""
    return hashlib.sha256(otp.encode()).hexdigest()


def send_otp_email(email, otp):
    """Gmail SMTP로 OTP 발송"""
    subject = '[AIfit] 이메일 인증 코드'
    message = f'''
안녕하세요, AIfit입니다!

회원가입을 위한 인증 코드입니다:

{otp}

이 코드는 5분 이내에 입력해주세요.
본인이 요청하지 않은 경우 이 이메일을 무시하셔도 됩니다.

감사합니다.
AIfit 팀
    '''

    try:
        send_mail(
            subject=subject,
            message=message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[email],
            fail_silently=False,
        )
        return True
    except Exception as e:
        print(f'이메일 발송 실패: {str(e)}')
        return False


def create_email_verification(email, otp):
    """이메일 인증 레코드 생성"""
    otp_hash = hash_otp(otp)
    expires_at = datetime.now() + timedelta(minutes=5)

    # 기존 미사용 OTP 삭제 (동일 이메일)
    EmailVerification.objects.filter(email=email, consumed=False).delete()

    verification = EmailVerification.objects.create(
        email=email,
        otp_hash=otp_hash,
        expires_at=expires_at
    )
    return verification


def verify_otp(email, otp):
    """
    OTP 검증
    - 이메일 일치
    - OTP 해시 일치
    - 만료 시간 체크
    - 사용 여부 체크
    """
    otp_hash = hash_otp(otp)

    try:
        verification = EmailVerification.objects.get(
            email=email,
            otp_hash=otp_hash,
            consumed=False,
            expires_at__gt=datetime.now()
        )
        # 사용 처리
        verification.consumed = True
        verification.save()
        return True
    except EmailVerification.DoesNotExist:
        return False
