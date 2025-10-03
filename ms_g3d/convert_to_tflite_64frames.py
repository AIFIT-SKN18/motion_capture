"""
64프레임용 TFLite 모델 변환
Colab 테스트와 동일한 입력 형식 사용
"""
import sys
import os

# UTF-8 출력 설정
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')

# MS-G3D 경로 추가
sys.path.insert(0, r'C:\Users\Playdata\Desktop\tflite_test\MS-G3D')

import torch
import numpy as np

print("=" * 80)
print("64프레임용 TFLite 변환")
print("=" * 80)

# 모델 import
from model.msg3d import Model

# 설정
PT_MODEL_PATH = r"C:\Users\Playdata\Desktop\tflite_test\weights-8-744.pt"
OUTPUT_ONNX = r"C:\Users\Playdata\Desktop\tflite_test\weights-8-744.onnx"
OUTPUT_DIR = r"C:\Users\Playdata\Desktop\tflite_test"

# 1. PyTorch 모델 로드
print("\n[1] PyTorch 모델 로드")
model_args = {
    'num_class': 6,
    'num_point': 25,
    'num_person': 1,
    'num_gcn_scales': 13,
    'num_g3d_scales': 6,
    'graph': 'graph.ntu_rgb_d.AdjMatrixGraph'
}

model = Model(**model_args)
checkpoint = torch.load(PT_MODEL_PATH, map_location='cpu')

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()
print("모델 로드 완료")

# 2. 테스트 입력 생성 (64프레임!)
print("\n[2] 테스트 입력 생성 (64프레임)")
dummy_input = torch.randn(1, 3, 64, 25, 1)  # window_size=64
print(f"입력 shape: {dummy_input.shape}")

# 3. PyTorch 추론 테스트
print("\n[3] PyTorch 추론 테스트")
with torch.no_grad():
    output = model(dummy_input)
print(f"출력 shape: {output.shape}")
print(f"출력값: {output[0].numpy()}")

# 4. ONNX 변환
print("\n[4] ONNX 변환")
torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_ONNX,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=None,
    verbose=False,
    training=torch.onnx.TrainingMode.EVAL,
)
print(f"ONNX 저장: {OUTPUT_ONNX}")

# 5. ONNX 검증
print("\n[5] ONNX 검증")
import onnx
import onnxruntime as ort

onnx_model = onnx.load(OUTPUT_ONNX)
onnx.checker.check_model(onnx_model)
print("ONNX 구조 검증 통과")

session = ort.InferenceSession(OUTPUT_ONNX)
onnx_output = session.run(None, {'input': dummy_input.numpy()})[0]
print(f"ONNX 추론 성공")
print(f"출력값: {onnx_output[0]}")

# 차이 확인
diff = np.abs(output.numpy() - onnx_output)
print(f"PyTorch vs ONNX 차이: 평균={diff.mean():.8f}, 최대={diff.max():.8f}")

# 6. ONNX 단순화
print("\n[6] ONNX 단순화")
from onnxsim import simplify

simplified_model, check = simplify(onnx_model)
if check:
    simplified_path = OUTPUT_ONNX.replace('.onnx', '_simplified.onnx')
    onnx.save(simplified_model, simplified_path)
    print(f"단순화 완료: {simplified_path}")
    onnx_input = simplified_path
else:
    print("단순화 실패, 원본 사용")
    onnx_input = OUTPUT_ONNX

# 7. TFLite 변환
print("\n[7] TFLite 변환 (onnx2tf)")
import subprocess

cmd = [
    sys.executable,
    '-m', 'onnx2tf',
    '-i', onnx_input,
    '-o', OUTPUT_DIR,
    '-osd',
    '-dgc',
    '-nuo'
]

print(f"명령어: {' '.join(cmd)}")

env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'

result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env, encoding='utf-8', errors='replace')

if result.returncode == 0:
    print("TFLite 변환 성공")

    # 생성된 파일 찾기
    tflite_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.tflite') and '64frames' in f]
    if tflite_files:
        print(f"생성된 파일: {tflite_files}")

        # Float32 파일만 이름 변경
        for f in tflite_files:
            if 'float32' in f:
                src = os.path.join(OUTPUT_DIR, f)
                dst = os.path.join(OUTPUT_DIR, "weights-64frames-float32.tflite")
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)
                print(f"→ {dst}")
else:
    print("TFLite 변환 실패")
    print(result.stderr)

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
print("\n다음 단계:")
print("1. weights-64frames-float32.tflite 파일 확인")
print("2. evaluate_single.py에서 모델 경로 변경")
print("3. 재테스트 실행")
