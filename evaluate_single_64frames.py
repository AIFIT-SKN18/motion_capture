import os
import numpy as np
import tensorflow as tf
from pose_extractor import PoseExtractor

# 추론을 위한 함수 정의
def predict_exercise_tflite(video_path, tflite_model_path, pose_extractor):
    """
    TFLite 모델로 새로운 비디오에 대해 덤벨/바벨 운동 분류 수행
    Colab과 동일한 전처리 사용 (64프레임, 정규화 없음)
    """
    class_names = [
        'benchpress',       # 0
        'deadlift',         # 1
        'lunges',           # 2
        'side_lateral_raise', # 3
        'squat',            # 4
        'others'            # 5
    ]

    # TFLite 인터프리터 로드
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # 입력 및 출력 텐서 정보 가져오기
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"입력 shape: {input_details[0]['shape']}")
    print(f"입력 dtype: {input_details[0]['dtype']}")

    # 비디오에서 스켈레톤 추출 (Colab과 동일)
    skeleton_seq = pose_extractor.extract_from_video(video_path)
    print(f"추출된 스켈레톤 shape: {skeleton_seq.shape}")  # (T, 25, 3)

    # Colab과 동일한 전처리: (T, 25, 3) -> (3, T, 25, 1)
    skeleton_data = skeleton_seq.transpose(2, 0, 1)  # (3, T, 25)
    skeleton_data = np.expand_dims(skeleton_data, axis=-1)  # (3, T, 25, 1)

    # 윈도우 크기 조정 (Colab과 동일: 64프레임)
    window_size = 64
    C, T, V, M = skeleton_data.shape
    print(f"현재 shape: (C={C}, T={T}, V={V}, M={M}), window_size: {window_size}")

    if T > window_size:
        indices = np.linspace(0, T - 1, window_size, dtype=int)
        skeleton_data = skeleton_data[:, indices, :, :]
    elif T < window_size:
        pad_width = ((0, 0), (0, window_size - T), (0, 0), (0, 0))
        skeleton_data = np.pad(skeleton_data, pad_width, mode='edge')

    print(f"윈도우 조정 후 shape: {skeleton_data.shape}")  # (3, 64, 25, 1)

    # TFLite 형식으로 변환: (C, T, V, M) -> (T, V, M, C)
    skeleton_data = skeleton_data.transpose(1, 2, 3, 0)  # (64, 25, 1, 3)

    # 배치 차원 추가: (64, 25, 1, 3) -> (1, 64, 25, 1, 3)
    input_data = np.expand_dims(skeleton_data, axis=0).astype(np.float32)
    print(f"최종 입력 shape: {input_data.shape}")

    # TFLite 모델 입력 설정
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 추론 실행
    interpreter.invoke()

    # 출력 가져오기
    output = interpreter.get_tensor(output_details[0]['index'])

    print(f"\n모델 원시 출력 (logits): {output[0]}")

    # Softmax 적용
    probabilities = np.exp(output[0]) / np.sum(np.exp(output[0]))
    predicted_class = np.argmax(probabilities)

    return predicted_class, probabilities, class_names


if __name__ == "__main__":
    # PoseExtractor import
    pose_extractor = PoseExtractor()

    # 설정
    test_folder = "C:\\Users\\Playdata\\Desktop\\tflite_test\\predict_mp4"
    tflite_model_path = "C:\\Users\\Playdata\\Desktop\\tflite_test\\weights-8-744_simplified_float32.tflite"

    # 테스트 비디오 리스트
    test_videos = [
        ("test1.mp4", "side_lateral_raise"),
        ("test2.mp4", "deadlift"),
        ("test3.mp4", "lunges"),
        ("test4.mp4", "squat"),
        ("test5.mp4", "bench_press"),
        ("00251201.mp4", "bench_press"),
        ("00321201.mp4", "deadlift"),
        ("00431201.mp4", "squat"),
    ]

    print(f"\n총 {len(test_videos)}개 테스트 예정\n")

    correct = 0
    total = 0

    for i, (video_name, expected_label) in enumerate(test_videos, 1):
        video_path = os.path.join(test_folder, video_name)

        if not os.path.exists(video_path):
            print(f"[{i}/{len(test_videos)}] 건너뛰기: {video_name} (파일 없음)\n")
            continue

        print(f"\n{'='*70}")
        print(f"[{i}/{len(test_videos)}] 테스트: {video_name}")
        print(f"기대 결과: {expected_label}")
        print('='*70)

        try:
            predicted_class, probs, class_names = predict_exercise_tflite(
                video_path, tflite_model_path, pose_extractor
            )

            predicted_label = class_names[predicted_class]
            print(f"\n예측 결과: {predicted_label}")

            # 정확도 체크 (간단한 문자열 매칭)
            is_correct = expected_label.split('(')[0].lower() in predicted_label.lower()
            if is_correct:
                correct += 1
                print("CORRECT!")
            else:
                print("WRONG!")

            total += 1

            print("\n각 클래스별 확률:")
            for j, (class_name, prob) in enumerate(zip(class_names, probs)):
                marker = "<<<" if j == predicted_class else ""
                print(f"  {class_name:20s}: {prob:.4f} ({prob*100:5.2f}%) {marker}")

        except Exception as e:
            print(f"예측 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"전체 정확도: {correct}/{total} = {100*correct/total:.1f}%" if total > 0 else "테스트 없음")
    print('='*70)
