from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from . import views

urlpatterns = [
    # 회원가입 3단계
    path('register/start/', views.register_start, name='register-start'),
    path('register/verify/', views.register_verify, name='register-verify'),
    path('register/complete/', views.register_complete, name='register-complete'),

    # 로그인 (SimpleJWT)
    path('login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),

    # 프로필
    path('profile/', views.profile, name='profile'),

    # 비밀번호 변경
    path('change-password/', views.change_password, name='change-password'),

    # 계정 삭제
    path('delete-account/', views.delete_account, name='delete-account'),

    # 로그아웃
    path('logout/', views.logout, name='logout'),
]
