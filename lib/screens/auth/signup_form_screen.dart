import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../providers/auth_provider.dart';
import '../../utils/validators.dart';
import '../main_screen.dart';

class SignupFormScreen extends StatefulWidget {
  final String email;

  const SignupFormScreen({Key? key, required this.email}) : super(key: key);

  @override
  State<SignupFormScreen> createState() => _SignupFormScreenState();
}

class _SignupFormScreenState extends State<SignupFormScreen> {
  final _formKey = GlobalKey<FormState>();
  final _passwordController = TextEditingController();
  final _passwordConfirmController = TextEditingController();
  final _nameController = TextEditingController();
  final _birthDateController = TextEditingController();

  String? _selectedGender;
  bool _isLoading = false;
  bool _obscurePassword = true;
  bool _obscurePasswordConfirm = true;
  String _passwordMatchMessage = ''; // 비밀번호 일치 메시지
  Color _passwordMatchColor = Colors.grey; // 메시지 색상

  @override
  void initState() {
    super.initState();
    // 비밀번호 확인 필드 리스너 추가
    _passwordConfirmController.addListener(_checkPasswordMatch);
    _passwordController.addListener(_checkPasswordMatch);
  }

  @override
  void dispose() {
    _passwordController.dispose();
    _passwordConfirmController.dispose();
    _nameController.dispose();
    _birthDateController.dispose();
    super.dispose();
  }

  // 비밀번호 일치 확인
  void _checkPasswordMatch() {
    final password = _passwordController.text;
    final passwordConfirm = _passwordConfirmController.text;

    setState(() {
      if (passwordConfirm.isEmpty) {
        _passwordMatchMessage = '';
        _passwordMatchColor = Colors.grey;
      } else if (password == passwordConfirm) {
        _passwordMatchMessage = '✓ 비밀번호가 일치합니다';
        _passwordMatchColor = Colors.green;
      } else {
        _passwordMatchMessage = '✗ 비밀번호가 일치하지 않습니다';
        _passwordMatchColor = Colors.red;
      }
    });
  }

  Future<void> _register() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => _isLoading = true);

    try {
      final authProvider = Provider.of<AuthProvider>(context, listen: false);
      await authProvider.registerComplete(
        email: widget.email,
        password: _passwordController.text,
        passwordConfirm: _passwordConfirmController.text,
        name: _nameController.text.trim(),
        gender: _selectedGender!,
        birthDate: _birthDateController.text.trim(),
      );

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('회원가입이 완료되었습니다!'),
            backgroundColor: Colors.green,
          ),
        );

        // 회원가입 완료 후 자동 로그인 처리 (이메일/비번으로 로그인)
        await authProvider.login(widget.email, _passwordController.text);

        // 메인 화면으로 이동
        Navigator.pushAndRemoveUntil(
          context,
          MaterialPageRoute(builder: (context) => const MainScreen()),
          (route) => false,
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(e.toString()), backgroundColor: Colors.red),
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
        title: const Text('프로필 입력'),
        backgroundColor: Colors.white,
        foregroundColor: Colors.black,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24.0),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                '프로필을 입력하세요',
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              Text(
                widget.email,
                style: const TextStyle(fontSize: 14, color: Colors.grey),
              ),
              const SizedBox(height: 32),

              // 비밀번호
              TextFormField(
                controller: _passwordController,
                obscureText: _obscurePassword,
                decoration: InputDecoration(
                  labelText: '비밀번호',
                  hintText: '소문자+숫자+특수기호, 8자 이상',
                  border: const OutlineInputBorder(),
                  prefixIcon: const Icon(Icons.lock),
                  suffixIcon: IconButton(
                    icon: Icon(_obscurePassword ? Icons.visibility_off : Icons.visibility),
                    onPressed: () => setState(() => _obscurePassword = !_obscurePassword),
                  ),
                ),
                validator: Validators.password,
                enabled: !_isLoading,
              ),
              const SizedBox(height: 16),

              // 비밀번호 확인
              TextFormField(
                controller: _passwordConfirmController,
                obscureText: _obscurePasswordConfirm,
                decoration: InputDecoration(
                  labelText: '비밀번호 확인',
                  hintText: '비밀번호를 다시 입력하세요',
                  border: const OutlineInputBorder(),
                  prefixIcon: const Icon(Icons.lock_outline),
                  suffixIcon: IconButton(
                    icon: Icon(_obscurePasswordConfirm ? Icons.visibility_off : Icons.visibility),
                    onPressed: () => setState(() => _obscurePasswordConfirm = !_obscurePasswordConfirm),
                  ),
                ),
                validator: Validators.passwordConfirm(_passwordController.text),
                enabled: !_isLoading,
              ),
              // 비밀번호 일치 여부 메시지
              if (_passwordMatchMessage.isNotEmpty) ...[
                const SizedBox(height: 8),
                Row(
                  children: [
                    const SizedBox(width: 12),
                    Icon(
                      _passwordMatchColor == Colors.green ? Icons.check_circle : Icons.cancel,
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
              const SizedBox(height: 16),

              // 이름
              TextFormField(
                controller: _nameController,
                decoration: const InputDecoration(
                  labelText: '이름',
                  hintText: '홍길동',
                  border: OutlineInputBorder(),
                  prefixIcon: Icon(Icons.person),
                ),
                validator: Validators.name,
                enabled: !_isLoading,
              ),
              const SizedBox(height: 16),

              // 성별
              DropdownButtonFormField<String>(
                value: _selectedGender,
                decoration: const InputDecoration(
                  labelText: '성별',
                  border: OutlineInputBorder(),
                  prefixIcon: Icon(Icons.wc),
                ),
                items: const [
                  DropdownMenuItem(value: 'M', child: Text('남성')),
                  DropdownMenuItem(value: 'F', child: Text('여성')),
                ],
                onChanged: _isLoading ? null : (value) => setState(() => _selectedGender = value),
                validator: Validators.gender,
              ),
              const SizedBox(height: 16),

              // 생년월일
              TextFormField(
                controller: _birthDateController,
                keyboardType: TextInputType.number,
                maxLength: 6,
                decoration: const InputDecoration(
                  labelText: '생년월일',
                  hintText: 'YYMMDD (예: 020124)',
                  border: OutlineInputBorder(),
                  prefixIcon: Icon(Icons.cake),
                ),
                validator: Validators.birthDate,
                enabled: !_isLoading,
              ),
              const SizedBox(height: 24),

              // 회원가입 버튼
              SizedBox(
                width: double.infinity,
                height: 50,
                child: ElevatedButton(
                  onPressed: _isLoading ? null : _register,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue,
                    foregroundColor: Colors.white,
                  ),
                  child: _isLoading
                      ? const CircularProgressIndicator(color: Colors.white)
                      : const Text('회원가입 완료', style: TextStyle(fontSize: 16)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
