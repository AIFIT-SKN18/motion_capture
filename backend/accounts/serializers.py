from rest_framework import serializers
from django.contrib.auth import get_user_model
import re
from datetime import datetime

User = get_user_model()


class RegisterStartSerializer(serializers.Serializer):
    """회원가입 1단계: 이메일 입력"""
    email = serializers.EmailField()

    def validate_email(self, value):
        if not value.endswith('@gmail.com'):
            raise serializers.ValidationError('Gmail 주소만 사용 가능합니다.')
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError('이미 가입된 이메일입니다.')
        return value


class RegisterVerifySerializer(serializers.Serializer):
    """회원가입 2단계: OTP 검증"""
    email = serializers.EmailField()
    otp = serializers.CharField(max_length=6, min_length=6)

    def validate_otp(self, value):
        if not value.isdigit():
            raise serializers.ValidationError('OTP는 6자리 숫자여야 합니다.')
        return value


class RegisterCompleteSerializer(serializers.Serializer):
    """회원가입 3단계: 프로필 입력 및 계정 생성"""
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True, min_length=8)
    name = serializers.CharField(max_length=100)
    gender = serializers.ChoiceField(choices=['M', 'F'])
    birth_date = serializers.CharField(max_length=6)  # YYMMDD

    def validate_email(self, value):
        if not value.endswith('@gmail.com'):
            raise serializers.ValidationError('Gmail 주소만 사용 가능합니다.')
        return value

    def validate_password(self, value):
        """
        비밀번호 정책 검증:
        - 소문자 포함
        - 숫자 포함
        - 특수기호 포함
        - 8자 이상
        """
        if len(value) < 8:
            raise serializers.ValidationError('비밀번호는 8자 이상이어야 합니다.')
        if not re.search(r'[a-z]', value):
            raise serializers.ValidationError('소문자를 포함해야 합니다.')
        if not re.search(r'\d', value):
            raise serializers.ValidationError('숫자를 포함해야 합니다.')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', value):
            raise serializers.ValidationError('특수기호를 포함해야 합니다.')
        return value

    def validate_birth_date(self, value):
        """YYMMDD 형식을 DATE로 변환"""
        if not re.match(r'^\d{6}$', value):
            raise serializers.ValidationError('YYMMDD 형식으로 입력하세요. (예: 020124)')

        try:
            # YYMMDD → 20YY-MM-DD 변환
            year = int('20' + value[:2])  # 2000년대 가정
            month = int(value[2:4])
            day = int(value[4:6])
            birth_date = datetime(year, month, day).date()
            return birth_date
        except ValueError:
            raise serializers.ValidationError('올바른 날짜가 아닙니다.')

    def validate(self, data):
        """비밀번호 확인"""
        if data['password'] != data['password_confirm']:
            raise serializers.ValidationError({
                'password_confirm': '비밀번호가 일치하지 않습니다.'
            })
        return data


class UserSerializer(serializers.ModelSerializer):
    """사용자 정보 직렬화"""
    class Meta:
        model = User
        fields = ['id', 'email', 'name', 'gender', 'birth_date', 'email_verified_at', 'date_joined']
        read_only_fields = ['id', 'email_verified_at', 'date_joined']


class TokenBlacklistSerializer(serializers.Serializer):
    """로그아웃 시 리프레시 토큰을 블랙리스트에 추가"""
    refresh = serializers.CharField()

    def validate_refresh(self, value):
        try:
            # 토큰 유효성 검증
            from rest_framework_simplejwt.tokens import RefreshToken
            RefreshToken(value)
        except Exception as e:
            raise serializers.ValidationError('올바르지 않거나 만료된 토큰입니다.')
        return value
