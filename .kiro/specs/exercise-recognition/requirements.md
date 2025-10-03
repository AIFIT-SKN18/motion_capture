# Requirements Document

## Introduction

이 애플리케이션은 float16 형식의 TensorFlow Lite 모델(weights_8_744_simplified_float16.tflite)을 사용하여 실시간 및 동영상 기반 운동 동작을 인지하고 분류하는 Flutter 기반 Android 애플리케이션입니다. MediaPipe를 통해 33개의 포즈 랜드마크를 추출하고 이를 NTU RGB+D 25개 관절점 형식으로 변환하여 모델에 입력합니다. 사용자는 갤러리에서 동영상을 선택하거나 실시간 카메라를 통해 운동 동작을 수행하고, 애플리케이션이 6개 클래스(benchpress, deadlift, lunges, side_lateral_raise, squat, others) 중 하나로 분류하여 결과를 표시합니다.

## Requirements

### Requirement 1: 홈 화면 네비게이션

**User Story:** 사용자로서, 실시간 동작 인식과 동영상 분석 기능 중 원하는 기능을 선택할 수 있도록 홈 화면에서 명확한 네비게이션 버튼을 보고 싶습니다.

#### Acceptance Criteria

1. WHEN 애플리케이션이 시작되면 THEN 시스템은 홈 화면을 표시해야 합니다
2. WHEN 홈 화면이 표시되면 THEN 시스템은 "실시간 동작 인식" 버튼을 표시해야 합니다
3. WHEN 홈 화면이 표시되면 THEN 시스템은 "동영상 분석" 버튼을 표시해야 합니다
4. WHEN 사용자가 "실시간 동작 인식" 버튼을 탭하면 THEN 시스템은 실시간 동작 인식 화면으로 이동해야 합니다
5. WHEN 사용자가 "동영상 분석" 버튼을 탭하면 THEN 시스템은 동영상 분석 화면으로 이동해야 합니다
6. WHEN 홈 화면이 표시되면 THEN 시스템은 애플리케이션 제목과 간단한 설명을 표시해야 합니다

### Requirement 2: TensorFlow Lite 모델 로딩 및 초기화

**User Story:** 개발자로서, float16 형식의 TensorFlow Lite 모델(weights_8_744_simplified_float16.tflite)이 애플리케이션 시작 시 정상적으로 로드되어 동작 분류에 사용될 수 있기를 원합니다.

#### Acceptance Criteria

1. WHEN 애플리케이션이 시작되면 THEN 시스템은 assets/models 폴더에서 weights_8_744_simplified_float16.tflite 모델을 로드해야 합니다
2. IF 모델 로딩이 실패하면 THEN 시스템은 사용자에게 오류 메시지를 표시해야 합니다
3. WHEN 모델이 성공적으로 로드되면 THEN 시스템은 tflite_flutter 패키지를 사용하여 인터프리터를 구성해야 합니다
4. WHEN 인터프리터가 구성되면 THEN 시스템은 float16 데이터 타입을 지원하도록 설정해야 합니다
5. WHEN 모델이 로드되면 THEN 시스템은 입력 텐서의 shape [1, 3, 64, 25, 1]을 확인하고 저장해야 합니다
6. WHEN 모델이 로드되면 THEN 시스템은 출력 클래스 수(6개: benchpress, deadlift, lunges, side_lateral_raise, squat, others)를 확인해야 합니다
7. WHEN 모델이 사용 가능하면 THEN 시스템은 추론(inference)을 수행할 준비가 되어야 합니다

### Requirement 3: 동영상 기반 동작 분류

**User Story:** 사용자로서, 갤러리에서 운동 동영상을 선택하여 TensorFlow Lite 모델이 동작을 올바르게 분류하는지 테스트하고 결과를 확인하고 싶습니다.

#### Acceptance Criteria

1. WHEN 사용자가 동영상 분석 화면에 진입하면 THEN 시스템은 "동영상 선택" 버튼을 표시해야 합니다
2. WHEN 사용자가 "동영상 선택" 버튼을 탭하면 THEN 시스템은 갤러리 선택 인터페이스를 표시해야 합니다
3. WHEN 사용자가 동영상을 선택하면 THEN 시스템은 선택된 동영상을 화면에 표시해야 합니다
4. WHEN 동영상이 선택되면 THEN 시스템은 "분석 시작" 버튼을 활성화해야 합니다
5. WHEN 사용자가 "분석 시작" 버튼을 탭하면 THEN 시스템은 동영상에서 프레임을 추출하고 포즈 데이터를 생성해야 합니다
6. WHEN 포즈 데이터가 생성되면 THEN 시스템은 TensorFlow Lite 모델을 사용하여 동작을 분류해야 합니다
7. WHEN 분류가 완료되면 THEN 시스템은 분류된 동작 클래스와 신뢰도(confidence score)를 화면에 표시해야 합니다
8. WHEN 분석이 진행 중이면 THEN 시스템은 진행 상태를 나타내는 로딩 인디케이터를 표시해야 합니다
9. IF 동영상 처리 중 오류가 발생하면 THEN 시스템은 사용자에게 오류 메시지를 표시해야 합니다
10. WHEN 분석 결과가 표시되면 THEN 사용자는 다른 동영상을 선택하여 다시 테스트할 수 있어야 합니다

### Requirement 4: 실시간 동작 인식 및 분류

**User Story:** 사용자로서, 카메라 앞에서 운동 동작을 실시간으로 수행하고, 시스템이 내 동작을 즉시 인식하고 분류하는 것을 확인하고 싶습니다.

#### Acceptance Criteria

1. WHEN 사용자가 실시간 동작 인식 화면에 진입하면 THEN 시스템은 카메라 권한을 요청해야 합니다
2. IF 카메라 권한이 거부되면 THEN 시스템은 권한이 필요하다는 메시지를 표시하고 설정으로 이동할 수 있는 옵션을 제공해야 합니다
3. WHEN 카메라 권한이 허용되면 THEN 시스템은 카메라 프리뷰를 화면에 표시해야 합니다
4. WHEN 카메라 프리뷰가 표시되면 THEN 시스템은 "시작" 버튼을 표시해야 합니다
5. WHEN 사용자가 "시작" 버튼을 탭하면 THEN 시스템은 3초 카운트다운을 표시해야 합니다
6. WHEN 카운트다운이 "3, 2, 1"로 진행되면 THEN 시스템은 각 숫자를 화면에 크게 표시해야 합니다
7. WHEN 3초 카운트다운이 완료되면 THEN 시스템은 실시간 포즈 감지를 시작해야 합니다
8. WHEN 실시간 포즈 감지가 활성화되면 THEN 시스템은 카메라 프레임에서 포즈 랜드마크를 추출해야 합니다
9. WHEN 포즈 데이터가 수집되면 THEN 시스템은 TensorFlow Lite 모델을 사용하여 실시간으로 동작을 분류해야 합니다
10. WHEN 동작이 분류되면 THEN 시스템은 현재 인식된 동작 클래스와 신뢰도를 화면에 실시간으로 표시해야 합니다
11. WHEN 실시간 인식이 진행 중이면 THEN 시스템은 카메라 프리뷰 위에 포즈 랜드마크를 오버레이로 표시해야 합니다
12. WHEN 사용자가 "중지" 버튼을 탭하면 THEN 시스템은 실시간 인식을 중지하고 초기 상태로 돌아가야 합니다
13. WHEN 실시간 인식이 활성화되면 THEN 시스템은 최소 15 FPS로 프레임을 처리해야 합니다

### Requirement 5: 포즈 데이터 전처리 및 모델 입력 준비

**User Story:** 개발자로서, 카메라나 동영상에서 추출한 포즈 데이터가 TensorFlow Lite 모델의 입력 형식 [1, 3, 64, 25, 1]에 맞게 전처리되어 정확한 분류가 이루어지기를 원합니다.

#### Acceptance Criteria

1. WHEN 시스템이 포즈 랜드마크를 감지하면 THEN 시스템은 MediaPipe의 33개 포즈 랜드마크를 추출해야 합니다
2. WHEN 포즈 랜드마크가 추출되면 THEN 시스템은 MediaPipe 랜드마크를 NTU RGB+D 25개 관절점 형식으로 변환해야 합니다
3. WHEN 좌표 변환이 완료되면 THEN 시스템은 이미지 크기를 기준으로 좌표를 정규화(0~1 범위)해야 합니다
4. WHEN 포즈 데이터가 수집되면 THEN 시스템은 64개 프레임의 시간 시퀀스 데이터를 준비해야 합니다
5. WHEN 시퀀스 데이터가 준비되면 THEN 시스템은 (T, V, C) 형식을 (C, T, V, M) 형식으로 transpose해야 합니다
6. WHEN 데이터가 변환되면 THEN 시스템은 [1, 3, 64, 25, 1] shape의 Float32List로 변환해야 합니다
7. IF 수집된 프레임이 64개 미만이면 THEN 시스템은 마지막 프레임으로 패딩을 수행해야 합니다
8. WHEN 전처리가 완료되면 THEN 시스템은 NaN 및 무한값을 검증하고 제거해야 합니다
9. WHEN 데이터 검증이 완료되면 THEN 시스템은 TensorFlow Lite 인터프리터를 통해 추론을 실행해야 합니다

### Requirement 6: 사용자 인터페이스 및 사용자 경험

**User Story:** 사용자로서, 직관적이고 반응성이 좋은 인터페이스를 통해 애플리케이션의 모든 기능을 쉽게 사용하고 결과를 명확하게 확인하고 싶습니다.

#### Acceptance Criteria

1. WHEN 애플리케이션이 실행되면 THEN 시스템은 Material Design 가이드라인을 따르는 UI를 표시해야 합니다
2. WHEN 분류 결과가 표시되면 THEN 시스템은 클래스 이름을 한글로 표시해야 합니다
3. WHEN 신뢰도가 표시되면 THEN 시스템은 백분율(%)로 표시해야 합니다
4. WHEN 로딩 상태가 발생하면 THEN 시스템은 명확한 로딩 인디케이터와 상태 메시지를 표시해야 합니다
5. WHEN 오류가 발생하면 THEN 시스템은 사용자 친화적인 오류 메시지를 표시해야 합니다
6. WHEN 사용자가 화면 간 이동하면 THEN 시스템은 부드러운 전환 애니메이션을 제공해야 합니다
7. WHEN 실시간 인식 화면에서 THEN 시스템은 뒤로 가기 버튼을 제공하여 홈 화면으로 돌아갈 수 있어야 합니다
8. WHEN 동영상 분석 화면에서 THEN 시스템은 뒤로 가기 버튼을 제공하여 홈 화면으로 돌아갈 수 있어야 합니다

### Requirement 7: 성능 및 안정성

**User Story:** 사용자로서, 애플리케이션이 빠르고 안정적으로 동작하여 끊김 없이 운동 동작을 인식하고 분류할 수 있기를 원합니다.

#### Acceptance Criteria

1. WHEN 모델 추론이 실행되면 THEN 시스템은 단일 추론을 200ms 이내에 완료해야 합니다
2. WHEN 실시간 인식이 진행되면 THEN 시스템은 메모리 사용량을 효율적으로 관리해야 합니다
3. IF 메모리 부족 상황이 발생하면 THEN 시스템은 적절히 리소스를 해제하고 사용자에게 알려야 합니다
4. WHEN 카메라가 사용 중이면 THEN 시스템은 화면이 꺼지지 않도록 설정해야 합니다
5. WHEN 애플리케이션이 백그라운드로 이동하면 THEN 시스템은 카메라와 모델 리소스를 적절히 해제해야 합니다
6. WHEN 애플리케이션이 포그라운드로 복귀하면 THEN 시스템은 필요한 리소스를 다시 초기화해야 합니다
7. WHEN 예외가 발생하면 THEN 시스템은 크래시 없이 적절히 처리하고 사용자에게 알려야 합니다

### Requirement 8: Android 플랫폼 호환성

**User Story:** 개발자로서, 애플리케이션이 Android Studio에서 빌드되고 다양한 Android 기기에서 정상적으로 동작하기를 원합니다.

#### Acceptance Criteria

1. WHEN 프로젝트가 빌드되면 THEN 시스템은 Android API 레벨 21(Lollipop) 이상을 지원해야 합니다
2. WHEN 애플리케이션이 설치되면 THEN 시스템은 필요한 권한(카메라, 저장소)을 AndroidManifest.xml에 선언해야 합니다
3. WHEN 빌드가 실행되면 THEN 시스템은 필요한 Flutter 플러그인(camera, google_mlkit_pose_detection, image_picker, tflite_flutter 등)을 포함해야 합니다
4. WHEN 애플리케이션이 실행되면 THEN 시스템은 다양한 화면 크기와 해상도를 지원해야 합니다
5. WHEN 기기가 회전하면 THEN 시스템은 세로 방향(portrait)으로 고정되어야 합니다
6. WHEN 애플리케이션이 배포되면 THEN 시스템은 release 모드에서 정상적으로 동작해야 합니다
7. WHEN TensorFlow Lite 모델이 로드되면 THEN 시스템은 float16 데이터 타입을 지원해야 합니다
