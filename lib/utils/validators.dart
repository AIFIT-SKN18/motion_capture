class Validators {
  // 이메일 검증 (Gmail만)
  static String? email(String? value) {
    if (value == null || value.isEmpty) {
      return '이메일을 입력하세요';
    }
    if (!value.endsWith('@gmail.com')) {
      return 'Gmail 주소만 사용 가능합니다';
    }
    final emailRegex = RegExp(r'^[a-zA-Z0-9.]+@gmail\.com$');
    if (!emailRegex.hasMatch(value)) {
      return '올바른 이메일 형식이 아닙니다';
    }
    return null;
  }

  // OTP 검증 (6자리 숫자)
  static String? otp(String? value) {
    if (value == null || value.isEmpty) {
      return '인증 코드를 입력하세요';
    }
    if (value.length != 6) {
      return '6자리 숫자를 입력하세요';
    }
    if (!RegExp(r'^\d{6}$').hasMatch(value)) {
      return '숫자만 입력 가능합니다';
    }
    return null;
  }

  // 비밀번호 검증 (소문자+숫자+특수기호, 8자 이상)
  static String? password(String? value) {
    if (value == null || value.isEmpty) {
      return '비밀번호를 입력하세요';
    }
    if (value.length < 8) {
      return '8자 이상 입력하세요';
    }
    if (!RegExp(r'[a-z]').hasMatch(value)) {
      return '소문자를 포함하세요';
    }
    if (!RegExp(r'\d').hasMatch(value)) {
      return '숫자를 포함하세요';
    }
    if (!RegExp(r'[!@#$%^&*(),.?":{}|<>]').hasMatch(value)) {
      return '특수기호를 포함하세요';
    }
    return null;
  }

  // 비밀번호 확인
  static String? Function(String?) passwordConfirm(String password) {
    return (String? value) {
      if (value == null || value.isEmpty) {
        return '비밀번호 확인을 입력하세요';
      }
      if (value != password) {
        return '비밀번호가 일치하지 않습니다';
      }
      return null;
    };
  }

  // 이름 검증
  static String? name(String? value) {
    if (value == null || value.isEmpty) {
      return '이름을 입력하세요';
    }
    if (value.length < 2) {
      return '2자 이상 입력하세요';
    }
    return null;
  }

  // 생년월일 검증 (YYMMDD)
  static String? birthDate(String? value) {
    if (value == null || value.isEmpty) {
      return '생년월일을 입력하세요';
    }
    if (!RegExp(r'^\d{6}$').hasMatch(value)) {
      return 'YYMMDD 형식으로 입력하세요 (예: 020124)';
    }

    // 날짜 유효성 검사
    try {
      final year = int.parse('20${value.substring(0, 2)}');
      final month = int.parse(value.substring(2, 4));
      final day = int.parse(value.substring(4, 6));

      if (month < 1 || month > 12) {
        return '올바른 월을 입력하세요 (01-12)';
      }

      if (day < 1 || day > 31) {
        return '올바른 일을 입력하세요 (01-31)';
      }

      // DateTime으로 변환하여 유효성 재확인
      DateTime(year, month, day);

      return null;
    } catch (e) {
      return '올바른 날짜가 아닙니다';
    }
  }

  // 성별 검증
  static String? gender(String? value) {
    if (value == null || value.isEmpty) {
      return '성별을 선택하세요';
    }
    return null;
  }
}
