from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User, EmailVerification


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ['email', 'name', 'gender', 'birth_date', 'email_verified_at', 'date_joined']
    list_filter = ['is_staff', 'is_active', 'gender']
    search_fields = ['email', 'name']
    ordering = ['-date_joined']

    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('개인정보', {'fields': ('name', 'gender', 'birth_date')}),
        ('권한', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('중요한 날짜', {'fields': ('email_verified_at', 'last_login', 'date_joined')}),
    )

    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'name', 'password1', 'password2'),
        }),
    )


@admin.register(EmailVerification)
class EmailVerificationAdmin(admin.ModelAdmin):
    list_display = ['email', 'consumed', 'expires_at', 'created_at']
    list_filter = ['consumed']
    search_fields = ['email']
    ordering = ['-created_at']
    readonly_fields = ['otp_hash', 'created_at']
