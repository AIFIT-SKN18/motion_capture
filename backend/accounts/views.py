from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from django_ratelimit.decorators import ratelimit
from django.utils import timezone
from .serializers import (
    RegisterStartSerializer,
    RegisterVerifySerializer,
    RegisterCompleteSerializer,
    UserSerializer,
    TokenBlacklistSerializer
)
from .utils import generate_otp, send_otp_email, create_email_verification, verify_otp
from .models import EmailVerification

User = get_user_model()


@api_view(['POST'])
@permission_classes([AllowAny])
@ratelimit(key='ip', rate='5/h', method='POST')
def register_start(request):
    """
    회원가입 1단계: 이메일 입력 → OTP 발송

    Request:
        {
            "email": "user@gmail.com"
        }

    Response:
        {
            "message": "인증 코드가 이메일로 발송되었습니다.",
            "email": "user@gmail.com"
        }
    """
    serializer = RegisterStartSerializer(data=request.data)

    if serializer.is_valid():
        email = serializer.validated_data['email']

        # OTP 생성 및 발송
        otp = generate_otp()
        create_email_verification(email, otp)

        # Gmail SMTP로 발송
        if send_otp_email(email, otp):
            return Response({
                'message': '인증 코드가 이메일로 발송되었습니다.',
                'email': email
            }, status=status.HTTP_200_OK)
        else:
            return Response({
                'error': '이메일 발송에 실패했습니다. 잠시 후 다시 시도해주세요.'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([AllowAny])
def register_verify(request):
    """
    회원가입 2단계: OTP 검증

    Request:
        {
            "email": "user@gmail.com",
            "otp": "123456"
        }

    Response:
        {
            "message": "이메일 인증이 완료되었습니다."
        }
    """
    serializer = RegisterVerifySerializer(data=request.data)

    if serializer.is_valid():
        email = serializer.validated_data['email']
        otp = serializer.validated_data['otp']

        if verify_otp(email, otp):
            return Response({
                'message': '이메일 인증이 완료되었습니다.'
            }, status=status.HTTP_200_OK)
        else:
            return Response({
                'error': '인증 코드가 올바르지 않거나 만료되었습니다.'
            }, status=status.HTTP_400_BAD_REQUEST)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([AllowAny])
def register_complete(request):
    """
    회원가입 3단계: 프로필 입력 → 계정 생성

    Request:
        {
            "email": "user@gmail.com",
            "password": "Test1234!",
            "password_confirm": "Test1234!",
            "name": "홍길동",
            "gender": "M",
            "birth_date": "020124"
        }

    Response:
        {
            "message": "회원가입이 완료되었습니다.",
            "user": {
                "id": 1,
                "email": "user@gmail.com",
                "name": "홍길동",
                ...
            }
        }
    """
    serializer = RegisterCompleteSerializer(data=request.data)

    if serializer.is_valid():
        email = serializer.validated_data['email']

        # 이메일 인증 확인
        verified = EmailVerification.objects.filter(
            email=email,
            consumed=True
        ).exists()

        if not verified:
            return Response({
                'error': '이메일 인증이 완료되지 않았습니다. 먼저 인증 코드를 확인해주세요.'
            }, status=status.HTTP_400_BAD_REQUEST)

        # 사용자 생성 (비밀번호는 Argon2로 자동 해싱)
        user = User.objects.create_user(
            email=email,
            password=serializer.validated_data['password'],
            name=serializer.validated_data['name'],
            gender=serializer.validated_data['gender'],
            birth_date=serializer.validated_data['birth_date'],
            email_verified_at=timezone.now()
        )

        return Response({
            'message': '회원가입이 완료되었습니다.',
            'user': UserSerializer(user).data
        }, status=status.HTTP_201_CREATED)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def profile(request):
    """
    사용자 프로필 조회

    Header:
        Authorization: Bearer {access_token}

    Response:
        {
            "id": 1,
            "email": "user@gmail.com",
            "name": "홍길동",
            "gender": "M",
            "birth_date": "2002-01-24",
            "email_verified_at": "2025-01-09T10:30:00Z",
            "date_joined": "2025-01-09T10:30:00Z"
        }
    """
    serializer = UserSerializer(request.user)
    return Response(serializer.data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def change_password(request):
    """
    비밀번호 변경

    Request:
        {
            "old_password": "OldPassword123!",
            "new_password": "NewPassword123!",
            "new_password_confirm": "NewPassword123!"
        }

    Response:
        {
            "message": "비밀번호가 성공적으로 변경되었습니다."
        }
    """
    old_password = request.data.get('old_password')
    new_password = request.data.get('new_password')
    new_password_confirm = request.data.get('new_password_confirm')

    if not all([old_password, new_password, new_password_confirm]):
        return Response({
            'error': '모든 필드를 입력해주세요.'
        }, status=status.HTTP_400_BAD_REQUEST)

    # 현재 비밀번호 확인
    if not request.user.check_password(old_password):
        return Response({
            'error': '현재 비밀번호가 올바르지 않습니다.'
        }, status=status.HTTP_400_BAD_REQUEST)

    # 새 비밀번호 확인
    if new_password != new_password_confirm:
        return Response({
            'error': '새 비밀번호가 일치하지 않습니다.'
        }, status=status.HTTP_400_BAD_REQUEST)

    # 비밀번호 길이 확인
    if len(new_password) < 8:
        return Response({
            'error': '비밀번호는 최소 8자 이상이어야 합니다.'
        }, status=status.HTTP_400_BAD_REQUEST)

    # 비밀번호 변경
    request.user.set_password(new_password)
    request.user.save()

    return Response({
        'message': '비밀번호가 성공적으로 변경되었습니다.'
    }, status=status.HTTP_200_OK)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_account(request):
    """
    계정 삭제

    Request:
        {
            "password": "CurrentPassword123!"
        }

    Response:
        {
            "message": "계정이 성공적으로 삭제되었습니다."
        }
    """
    password = request.data.get('password')

    if not password:
        return Response({
            'error': '비밀번호를 입력해주세요.'
        }, status=status.HTTP_400_BAD_REQUEST)

    # 비밀번호 확인
    if not request.user.check_password(password):
        return Response({
            'error': '비밀번호가 올바르지 않습니다.'
        }, status=status.HTTP_400_BAD_REQUEST)

    # 계정 삭제
    user_email = request.user.email
    request.user.delete()

    return Response({
        'message': f'{user_email} 계정이 성공적으로 삭제되었습니다.'
    }, status=status.HTTP_200_OK)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout(request):
    """
    로그아웃

    Request:
        {
            "refresh": "refresh_token"
        }

    Response:
        {
            "message": "로그아웃되었습니다."
        }
    """
    serializer = TokenBlacklistSerializer(data=request.data)
    if serializer.is_valid():
        try:
            from rest_framework_simplejwt.tokens import RefreshToken
            refresh_token = serializer.validated_data['refresh']
            token = RefreshToken(refresh_token)
            token.blacklist()
            return Response({'message': '로그아웃되었습니다.'}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
