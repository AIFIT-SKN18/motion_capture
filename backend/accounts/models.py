from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models
from django.utils import timezone


class UserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('이메일은 필수입니다')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('name', 'Admin')
        return self.create_user(email, password, **extra_fields)


class User(AbstractBaseUser, PermissionsMixin):
    GENDER_CHOICES = [
        ('M', '남성'),
        ('F', '여성'),
    ]

    email = models.EmailField(unique=True, verbose_name='이메일')
    name = models.CharField(max_length=100, verbose_name='이름')
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, null=True, blank=True, verbose_name='성별')
    birth_date = models.DateField(null=True, blank=True, verbose_name='생년월일')
    email_verified_at = models.DateTimeField(null=True, blank=True, verbose_name='이메일 인증 시각')

    is_active = models.BooleanField(default=True, verbose_name='활성 상태')
    is_staff = models.BooleanField(default=False, verbose_name='스태프 권한')
    date_joined = models.DateTimeField(default=timezone.now, verbose_name='가입일')

    objects = UserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    class Meta:
        db_table = 'users'
        verbose_name = '사용자'
        verbose_name_plural = '사용자 목록'

    def __str__(self):
        return self.email


class EmailVerification(models.Model):
    email = models.EmailField(verbose_name='이메일')
    otp_hash = models.CharField(max_length=255, verbose_name='OTP 해시')
    expires_at = models.DateTimeField(verbose_name='만료 시각')
    consumed = models.BooleanField(default=False, verbose_name='사용 여부')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='생성 시각')

    class Meta:
        db_table = 'email_verifications'
        verbose_name = '이메일 인증'
        verbose_name_plural = '이메일 인증 목록'
        indexes = [
            models.Index(fields=['email', 'expires_at']),
        ]

    def __str__(self):
        return f'{self.email} - {self.created_at}'
