class User {
  final int id;
  final String email;
  final String name;
  final String? gender;
  final DateTime? birthDate;
  final DateTime? emailVerifiedAt;
  final DateTime dateJoined;

  User({
    required this.id,
    required this.email,
    required this.name,
    this.gender,
    this.birthDate,
    this.emailVerifiedAt,
    required this.dateJoined,
  });

  factory User.fromJson(Map<String, dynamic> json) {
    return User(
      id: json['id'],
      email: json['email'],
      name: json['name'],
      gender: json['gender'],
      birthDate: json['birth_date'] != null
          ? DateTime.parse(json['birth_date'])
          : null,
      emailVerifiedAt: json['email_verified_at'] != null
          ? DateTime.parse(json['email_verified_at'])
          : null,
      dateJoined: DateTime.parse(json['date_joined']),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'email': email,
      'name': name,
      'gender': gender,
      'birth_date': birthDate?.toIso8601String(),
      'email_verified_at': emailVerifiedAt?.toIso8601String(),
      'date_joined': dateJoined.toIso8601String(),
    };
  }
}
