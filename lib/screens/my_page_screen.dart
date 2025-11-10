import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../providers/auth_provider.dart';
import 'change_password_screen.dart';

/// 마이페이지 화면
class MyPageScreen extends StatefulWidget {
  const MyPageScreen({super.key});

  @override
  State<MyPageScreen> createState() => _MyPageScreenState();
}

class _MyPageScreenState extends State<MyPageScreen> {
  String? _profileImageBase64;
  final ImagePicker _picker = ImagePicker();
  int? _currentUserId;

  @override
  void initState() {
    super.initState();
  }

  // 프로필 이미지 로드
  Future<void> _loadProfileImage() async {
    // 위젯이 마운트되지 않았다면 중단
    if (!mounted) return;

    final user = context.read<AuthProvider>().user;
    if (user == null) {
      setState(() {
        _profileImageBase64 = null;
      });
      return;
    }

    final prefs = await SharedPreferences.getInstance();
    final key = 'profile_image_${user.id}';
    final base64Image = prefs.getString(key);

    // 위젯이 마운트되지 않았다면 중단
    if (!mounted) return;

    setState(() {
      _profileImageBase64 = base64Image;
    });
  }

  // 프로필 이미지 선택
  Future<void> _pickProfileImage() async {
    final user = context.read<AuthProvider>().user;
    if (user == null) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('사용자 정보가 없어 이미지를 저장할 수 없습니다.')),
        );
      }
      return;
    }

    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 512,
        maxHeight: 512,
        imageQuality: 85,
      );

      if (image == null) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('이미지 선택이 취소되었습니다')),
          );
        }
        return;
      }

      // 이미지를 base64로 변환
      final bytes = await File(image.path).readAsBytes();
      final base64Image = base64Encode(bytes);

      // SharedPreferences에 저장
      final prefs = await SharedPreferences.getInstance();
      final key = 'profile_image_${user.id}';
      await prefs.setString(key, base64Image);

      // UI 업데이트
      if (mounted) {
        setState(() {
          _profileImageBase64 = base64Image;
        });

        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('프로필 이미지가 변경되었습니다'),
            backgroundColor: Colors.green,
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('이미지 선택 오류: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  // 비밀번호 변경 화면으로 이동
  void _navigateToChangePassword() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => const ChangePasswordScreen(),
      ),
    );
  }

  // 로그아웃 다이얼로그
  void _showLogoutDialog() {
    showDialog(
      context: context,
      builder: (BuildContext dialogContext) {
        return AlertDialog(
          title: const Text('로그아웃'),
          content: const Text('정말 로그아웃할까요?'),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(dialogContext),
              child: const Text('취소'),
            ),
            TextButton(
              onPressed: () async {
                Navigator.pop(dialogContext);
                await context.read<AuthProvider>().logout();
                // 로그아웃 후 로그인 화면으로 이동 (자동으로 AuthWrapper에서 처리됨)
              },
              child: const Text(
                '확인',
                style: TextStyle(color: Colors.red),
              ),
            ),
          ],
        );
      },
    );
  }

  // 계정 삭제 다이얼로그
  void _showDeleteAccountDialog() {
    final TextEditingController confirmController = TextEditingController();
    final TextEditingController passwordController = TextEditingController();
    bool isDeleteEnabled = false;
    bool isLoading = false;

    showDialog(
      context: context,
      builder: (BuildContext dialogContext) {
        return StatefulBuilder(
          builder: (context, setDialogState) {
            return AlertDialog(
              title: const Text('계정 삭제'),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    '계정을 삭제하면 모든 데이터가 영구적으로 삭제되며 복구할 수 없습니다.',
                    style: TextStyle(color: Colors.red),
                  ),
                  const SizedBox(height: 16),
                  const Text('비밀번호를 입력하세요:'),
                  const SizedBox(height: 8),
                  TextField(
                    controller: passwordController,
                    obscureText: true,
                    decoration: const InputDecoration(
                      hintText: '비밀번호',
                      border: OutlineInputBorder(),
                    ),
                    onChanged: (value) {
                      setDialogState(() {
                        isDeleteEnabled = value.isNotEmpty && confirmController.text == 'DELETE';
                      });
                    },
                  ),
                  const SizedBox(height: 16),
                  const Text('계속하려면 "DELETE"를 입력하세요:'),
                  const SizedBox(height: 8),
                  TextField(
                    controller: confirmController,
                    decoration: const InputDecoration(
                      hintText: 'DELETE',
                      border: OutlineInputBorder(),
                    ),
                    onChanged: (value) {
                      setDialogState(() {
                        isDeleteEnabled = value == 'DELETE' && passwordController.text.isNotEmpty;
                      });
                    },
                  ),
                ],
              ),
              actions: [
                TextButton(
                  onPressed: isLoading ? null : () => Navigator.pop(dialogContext),
                  child: const Text('취소'),
                ),
                TextButton(
                  onPressed: (isDeleteEnabled && !isLoading)
                      ? () async {
                          setDialogState(() => isLoading = true);

                          try {
                            final message = await context.read<AuthProvider>().deleteAccount(
                              passwordController.text,
                            );

                            if (context.mounted) {
                              Navigator.pop(dialogContext);
                              ScaffoldMessenger.of(context).showSnackBar(
                                SnackBar(
                                  content: Text(message),
                                  backgroundColor: Colors.green,
                                ),
                              );
                              // 계정 삭제 후 로그인 화면으로 이동 (자동으로 AuthWrapper에서 처리됨)
                            }
                          } catch (e) {
                            if (context.mounted) {
                              ScaffoldMessenger.of(context).showSnackBar(
                                SnackBar(
                                  content: Text(e.toString()),
                                  backgroundColor: Colors.red),
                              );
                            }
                          } finally {
                            if (mounted) {
                              setDialogState(() => isLoading = false);
                            }
                          }
                        }
                      : null,
                  child: isLoading
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : Text(
                          '삭제',
                          style: TextStyle(
                            color: isDeleteEnabled ? Colors.red : Colors.grey,
                          ),
                        ),
                ),
              ],
            );
          },
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final authProvider = context.watch<AuthProvider>();
    final user = authProvider.user;

    // 사용자 변경 감지
    if (user != null && user.id != _currentUserId) {
      _currentUserId = user.id;
      // build 중에 상태를 변경하므로, 다음 프레임에서 실행되도록 예약
      WidgetsBinding.instance.addPostFrameCallback((_) {
        _loadProfileImage();
      });
    } else if (user == null && _currentUserId != null) {
      _currentUserId = null;
      // build 중에 상태를 변경하므로, 다음 프레임에서 실행되도록 예약
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) {
          setState(() {
            _profileImageBase64 = null;
          });
        }
      });
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('My Page'),
        centerTitle: true,
        elevation: 0,
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16.0),
          children: [
            // 프로필 섹션
            _buildProfileSection(user?.name ?? '홍길동', user?.email ?? 'user@gmail.com'),
            const SizedBox(height: 24),

            // 계정 섹션
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 4.0, vertical: 8.0),
              child: Text(
                '계정',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
            ),
            const Divider(),

            // 비밀번호 변경
            _buildActionTile(
              icon: Icons.lock_outline,
              title: 'Change Password',
              subtitle: '비밀번호 변경',
              onTap: _navigateToChangePassword,
            ),
            const Divider(),

            // 로그아웃
            _buildActionTile(
              icon: Icons.logout,
              title: 'Logout',
              subtitle: '로그아웃',
              onTap: _showLogoutDialog,
              iconColor: Colors.orange,
            ),
            const Divider(),

            // 계정 삭제
            _buildActionTile(
              icon: Icons.delete_forever,
              title: 'Delete Account',
              subtitle: '계정 삭제',
              onTap: _showDeleteAccountDialog,
              iconColor: Colors.red,
            ),
          ],
        ),
      ),
    );
  }

  // 프로필 섹션 위젯
  Widget _buildProfileSection(String name, String email) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      child: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          children: [
            // 프로필 이미지
            GestureDetector(
              onTap: _pickProfileImage,
              child: Stack(
                children: [
                  CircleAvatar(
                    radius: 50,
                    backgroundImage: _profileImageBase64 != null
                        ? MemoryImage(base64Decode(_profileImageBase64!))
                        : null,
                    child: _profileImageBase64 == null
                        ? const Icon(Icons.person, size: 50)
                        : null,
                  ),
                  Positioned(
                    bottom: 0,
                    right: 0,
                    child: Container(
                      padding: const EdgeInsets.all(4),
                      decoration: BoxDecoration(
                        color: Theme.of(context).colorScheme.primary,
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(
                        Icons.camera_alt,
                        size: 20,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // 이름
            Text(
              name,
              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 4),

            // 이메일
            Text(
              email,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Colors.grey[600],
                  ),
            ),
          ],
        ),
      ),
    );
  }

  // 액션 타일 위젯
  Widget _buildActionTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
    Color? iconColor,
  }) {
    return ListTile(
      leading: Icon(icon, color: iconColor),
      title: Text(title),
      subtitle: Text(subtitle),
      trailing: const Icon(Icons.chevron_right),
      onTap: onTap,
    );
  }
}
