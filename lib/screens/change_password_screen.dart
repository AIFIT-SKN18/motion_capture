import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/auth_provider.dart';
import '../utils/validators.dart';

/// 비밀번호 변경 화면
class ChangePasswordScreen extends StatefulWidget {
  const ChangePasswordScreen({super.key});

  @override
  State<ChangePasswordScreen> createState() => _ChangePasswordScreenState();
}

class _ChangePasswordScreenState extends State<ChangePasswordScreen> {
  final _formKey = GlobalKey<FormState>();
  final _oldPasswordController = TextEditingController();
  final _newPasswordController = TextEditingController();
  final _newPasswordConfirmController = TextEditingController();

  bool _isLoading = false;
  bool _obscureOldPassword = true;
  bool _obscureNewPassword = true;
  bool _obscureNewPasswordConfirm = true;

  String _passwordMatchMessage = '';
  Color _passwordMatchColor = Colors.grey;

  @override
  void initState() {
    super.initState();
    _newPasswordConfirmController.addListener(_checkPasswordMatch);
    _newPasswordController.addListener(_checkPasswordMatch);
  }

  @override
  void dispose() {
    _oldPasswordController.dispose();
    _newPasswordController.dispose();
    _newPasswordConfirmController.dispose();
    super.dispose();
  }

  void _checkPasswordMatch() {
    final newPassword = _newPasswordController.text;
    final newPasswordConfirm = _newPasswordConfirmController.text;

    setState(() {
      if (newPasswordConfirm.isEmpty) {
        _passwordMatchMessage = '';
        _passwordMatchColor = Colors.grey;
      } else if (newPassword == newPasswordConfirm) {
        _passwordMatchMessage = '✓ 비밀번호가 일치합니다';
        _passwordMatchColor = Colors.green;
      } else {
        _passwordMatchMessage = '✗ 비밀번호가 일치하지 않습니다';
        _passwordMatchColor = Colors.red;
      }
    });
  }

  Future<void> _changePassword() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => _isLoading = true);

    try {
      final authProvider = context.read<AuthProvider>();
      final message = await authProvider.changePassword(
        oldPassword: _oldPasswordController.text,
        newPassword: _newPasswordController.text,
        newPasswordConfirm: _newPasswordConfirmController.text,
      );

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(message),
            backgroundColor: Colors.green,
          ),
        );
        Navigator.pop(context);
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(e.toString()),
            backgroundColor: Colors.red,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Change Password'),
        elevation: 0,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(24.0),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  '비밀번호 변경',
                  style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 8),
                const Text(
                  '보안을 위해 현재 비밀번호를 확인합니다',
                  style: TextStyle(fontSize: 14, color: Colors.grey),
                ),
                const SizedBox(height: 32),

                // 기존 비밀번호
                TextFormField(
                  controller: _oldPasswordController,
                  obscureText: _obscureOldPassword,
                  decoration: InputDecoration(
                    labelText: '현재 비밀번호',
                    hintText: '현재 사용 중인 비밀번호를 입력하세요',
                    border: const OutlineInputBorder(),
                    prefixIcon: const Icon(Icons.lock),
                    suffixIcon: IconButton(
                      icon: Icon(_obscureOldPassword
                          ? Icons.visibility_off
                          : Icons.visibility),
                      onPressed: () =>
                          setState(() => _obscureOldPassword = !_obscureOldPassword),
                    ),
                  ),
                  validator: (value) {
                    if (value == null || value.isEmpty) {
                      return '현재 비밀번호를 입력해주세요';
                    }
                    return null;
                  },
                  enabled: !_isLoading,
                ),
                const SizedBox(height: 16),

                // 새 비밀번호
                TextFormField(
                  controller: _newPasswordController,
                  obscureText: _obscureNewPassword,
                  decoration: InputDecoration(
                    labelText: '새 비밀번호',
                    hintText: '소문자+숫자+특수기호, 8자 이상',
                    border: const OutlineInputBorder(),
                    prefixIcon: const Icon(Icons.lock_outline),
                    suffixIcon: IconButton(
                      icon: Icon(_obscureNewPassword
                          ? Icons.visibility_off
                          : Icons.visibility),
                      onPressed: () =>
                          setState(() => _obscureNewPassword = !_obscureNewPassword),
                    ),
                  ),
                  validator: Validators.password,
                  enabled: !_isLoading,
                ),
                const SizedBox(height: 16),

                // 새 비밀번호 확인
                TextFormField(
                  controller: _newPasswordConfirmController,
                  obscureText: _obscureNewPasswordConfirm,
                  decoration: InputDecoration(
                    labelText: '새 비밀번호 확인',
                    hintText: '새 비밀번호를 다시 입력하세요',
                    border: const OutlineInputBorder(),
                    prefixIcon: const Icon(Icons.lock_outline),
                    suffixIcon: IconButton(
                      icon: Icon(_obscureNewPasswordConfirm
                          ? Icons.visibility_off
                          : Icons.visibility),
                      onPressed: () => setState(
                          () => _obscureNewPasswordConfirm = !_obscureNewPasswordConfirm),
                    ),
                  ),
                  validator: Validators.passwordConfirm(_newPasswordController.text),
                  enabled: !_isLoading,
                ),

                // 비밀번호 일치 여부 메시지
                if (_passwordMatchMessage.isNotEmpty) ...[
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      const SizedBox(width: 12),
                      Icon(
                        _passwordMatchColor == Colors.green
                            ? Icons.check_circle
                            : Icons.cancel,
                        color: _passwordMatchColor,
                        size: 16,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        _passwordMatchMessage,
                        style: TextStyle(
                          color: _passwordMatchColor,
                          fontSize: 14,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ),
                ],
                const SizedBox(height: 32),

                // 변경하기 버튼
                SizedBox(
                  width: double.infinity,
                  height: 50,
                  child: ElevatedButton(
                    onPressed: _isLoading ? null : _changePassword,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue,
                      foregroundColor: Colors.white,
                    ),
                    child: _isLoading
                        ? const CircularProgressIndicator(color: Colors.white)
                        : const Text('비밀번호 변경', style: TextStyle(fontSize: 16)),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
