# motion_capture

이 저장소는 **MediaPipe Pose → NTU RGB+D 25관절 포맷 → 2s‑AGCN(2‑Stream Adaptive GCN)** 흐름으로 운동 동작을 분류하는 프로젝트입니다. 로컬 MP4를 스켈레톤/NPY로 변환하고, 2s‑AGCN으로 학습·추론·앙상블까지 이어지는 파이프라인을 포함합니다.

## 폴더 구조
- `Toyproject-motionclassification/`
  - `2s-AGCN-0.0/`: 2s‑AGCN 모델 코드(원본 리포지토리 기반)
  - `yaml/`: 학습/테스트 설정(YAML)
  - `mp4_preprocess.py`: MP4 → joint NPY 변환
  - `data_mp4.py`: MP4 → NTU `.skeleton` 및/또는 joint NPY 변환
  - `ms_g3d_basic_preprocessing.py`: 기본 전처리/데이터셋 생성 스크립트
  - `motion.ipynb`: 실험/분석용 노트북
  - `fine-tuning*`, `fine-tuning_outdir/`: 파인튜닝 실험 및 결과 산출물
  - `npy_data/`, `single_npy/`: 전처리 산출물/샘플

## 개발/처리 파이프라인 요약
1) **MP4 → 스켈레톤 추출**
- `data_mp4.py` 또는 `mp4_preprocess.py`에서 MediaPipe Pose로 관절을 추출합니다.
- MediaPipe 33관절을 NTU RGB+D 25관절 포맷으로 매핑합니다.

2) **포맷 변환**
- `.skeleton` (NTU RGB+D 텍스트 포맷) 또는
- `*_data_joint.npy` (형태: `(C,T,V,M) = (3,T,25,1)`)로 저장합니다.

3) **학습/추론(2s‑AGCN)**
- `2s-AGCN-0.0/main.py`를 YAML 설정과 함께 실행합니다.
- joint/bone 스트림을 각각 학습한 뒤 `ensemble.py`로 앙상블합니다.

4) **파인튜닝/평가**
- `yaml/` 폴더에 커스텀 6클래스용 설정이 포함되어 있으며, 결과는 `fine-tuning_outdir/`에 저장됩니다.

## 데이터/전처리/학습 상세
### 1) 입력 데이터 구조
- 비디오는 클래스별 폴더 구조로 준비합니다(예: `benchpress/`, `deadlift/`, `lunges/`, `side_lateral_raise/`, `squat/`).
- `ms_g3d_basic_preprocessing.py`는 **알려진 5개 클래스**를 고정으로 사용하고, 나머지 폴더는 전부 `others`로 묶어 **총 6클래스**로 라벨링합니다.

### 2) 포즈 추출 및 관절 매핑
- MediaPipe Pose(33관절)를 사용해 프레임별 관절을 추출합니다.
- 33관절을 **NTU RGB+D 25관절 포맷**으로 매핑합니다.
  - hip center/shoulder center 계산 등은 스크립트 내부에서 직접 계산합니다.
  - 인식 실패 프레임은 직전 프레임 복사(없으면 0으로 채움)로 보정합니다.

### 3) 시퀀스 길이 보정(T)
- `mp4_preprocess.py`와 `data_mp4.py`는 **고정 길이 T**를 사용합니다(기본 300).
- 길이가 짧으면 `repeat_last` 또는 `zeros` 패딩으로 보정합니다.
- 길이가 길면 앞에서 T프레임만 사용합니다.

### 4) 출력 포맷
- `.skeleton` (NTU RGB+D 텍스트 포맷) 또는
- `*_data_joint.npy` (형태: `(C,T,V,M) = (3,T,25,1)`)로 저장합니다.
- `data_mp4.py`는 `.skeleton`/`.npy`/둘 다를 선택 가능합니다(`--format`).

### 5) 학습 데이터셋 구성
- `ms_g3d_basic_preprocessing.py`는 아래 파일을 생성합니다.
  - `exercise_data_joint.npy`: 샘플별 스켈레톤 시퀀스 딕셔너리
  - `exercise_train_label.pkl`: 샘플 이름 + 라벨 리스트
  - `exercise_class_info.pkl`: 클래스 이름/인덱스 매핑
- 이 데이터는 2s‑AGCN의 feeder에서 로드할 수 있도록 `(C,T,V,M)` 형태로 저장됩니다.

### 6) 학습 설정(2s‑AGCN)
- 실제 학습은 `2s-AGCN-0.0/main.py` + YAML 설정으로 실행합니다.
- `yaml/train_joint.yaml` 기준 주요 설정:
  - `num_class: 6`, `num_point: 25`, `num_person: 1`
  - `graph: graph.ntu_rgb_d.Graph` (NTU RGB+D 관절 그래프 사용)
  - batch size 64, epoch 50, step LR 스케줄 사용
- 학습 데이터 경로와 로그 저장 경로는 YAML에 지정되어 있으며, **절대 경로**를 로컬 환경에 맞게 수정해야 합니다.

### 7) 테스트/평가
- `yaml/test_joint.yaml`에 테스트 데이터 경로를 지정합니다.
- 평가 결과는 `fine-tuning_outdir/` 등에 저장되며, 필요 시 `ensemble.py`로 joint/bone 스트림 앙상블이 가능합니다.

## motion.ipynb 기반 학습/확률 처리 상세
이 섹션은 `Toyproject-motionclassification/motion.ipynb`의 실행 흐름을 정리한 것입니다(Colab 경로 기반).

### 1) 데이터 준비/전처리 (노트북)
- **입력 구조**: 클래스/영상.mp4 폴더 구조를 가정합니다.
- **MP4 → NTU .skeleton 변환**
  - MediaPipe Pose로 프레임별 관절을 추출하고 NTU 25관절로 매핑합니다.
  - 원본 FPS와 무관하게 **TARGET_FPS=30**으로 리샘플합니다.
  - 길이는 **TARGET_T=300**으로 고정(길면 균등 샘플링, 짧으면 0-body 패딩)합니다.
  - `.skeleton` 파일명은 NTU 형식(SxxxCxxxPxxxRxxxAxxx_*)으로 생성합니다.
- **.skeleton → joint NPY**
  - `data_gen.gen_ntu_batch`를 사용해 joint NPY로 변환합니다.
  - 고정 라벨(`label_mode fixed`) 또는 파일별 라벨 저장 옵션을 사용합니다.
- **중복 제거**
  - NPY 파일 해시 비교로 중복을 찾아 제거합니다(플립 데이터는 우선적으로 제거 대상으로 분류).
- **Train/Test 분리**
  - **다중 클래스 stratified split**(예: 0.7/0.3)로 분할합니다.
  - 라벨은 sidecar `.pkl` 또는 파일명 `_y{label}`을 우선 사용합니다.
- **CTVM 정규화 및 M 보정**
  - `ctvm_sanitize.py`로 `(C,T,V,M)` 형식을 정규화합니다.
  - `FORCE_M=2`로 사람 수 차원을 보정합니다.
- **Bone 데이터 생성**
  - `folder_bone_generator.py`로 joint → bone 변환을 수행합니다.

### 2) 학습 방법 (노트북 실행 커맨드 기반)
- **파인튜닝 스크립트**
  - `train_full_backbone_fc_finetune.patched.patched2.py` 또는
    `train_full_backbone_fc_finetune_plus_head_calib.py` 사용.
- **기본 설정**
  - `num_class=6`, `num_point=25`, `num_person=2`
  - `graph: graph.ntu_rgb_d.Graph` (spatial labeling)
  - 입력 길이 `window_T=300`
- **학습 전략**
  - 사전학습 체크포인트(`ntu_cs_agcn_*`)에서 시작
  - fine-tuning 대상 파라미터 prefix: `fc., classifier., head.`
  - early stopping 사용 (metric: acc 또는 f1, patience 지정)
  - label smoothing, cosine head, per-class margin, bias calibration 등을 상황에 따라 적용

### 3) 확률/점수 처리 (노트북 실행 커맨드 기반)
- **평가 스크립트**
  - `eval_full_model_recursive3.py` 및 `eval_full_model_recursive_3_patched_with_scores.py`
  - per-sample score CSV 덤프(`--dump_scores`) 지원
- **단일 샘플 추론**
  - `predict_single_npy_centerpad.py`, `predict_single_npy_leftpad*.py` 사용
  - 출력은 클래스별 score(확률/점수)와 top-k를 출력하도록 구성
- **기타 게이트 옵션**
  - `apply_other_gate`, `other_top1_min`, `other_valid_first/last` 같은 옵션으로
    **others 클래스 판정 게이트**를 추가 적용할 수 있도록 설계되어 있습니다.

### 4) 확률 점수 계산 방식
- **MS‑G3D TFLite 추론**(`evaluate_single_100frames.py`)
  - 모델 출력은 logits이며, 아래처럼 **softmax**로 확률을 계산합니다.
    - `prob = exp(logit) / sum(exp(logit))`
  - 이 확률로 `argmax`를 취해 최종 클래스를 결정합니다.
- **2s‑AGCN 노트북 추론**
  - `predict_single_npy_*` 스크립트에서 클래스별 **score/확률**을 출력합니다.
  - 필요 시 `eval_full_model_recursive_3_patched_with_scores.py`로 per-sample score를 CSV로 저장합니다.
- **others gate (선택)**  
  - `apply_other_gate=1`일 때, top-1 확률이 임계값(`other_top1_min`) 미만이면 **others로 보정**합니다.
  - `other_valid_first/last`로 정상 클래스 범위를 제한해 **others 판정 조건**을 강화할 수 있습니다.

## 핵심 스크립트 설명
- `Toyproject-motionclassification/mp4_preprocess.py`
  - MP4 → `(3,T,25,1)` NPY 변환
  - 길이 보정(T) 및 pad 방식 선택 가능

- `Toyproject-motionclassification/data_mp4.py`
  - MP4 → NTU `.skeleton` 또는 NPY(또는 둘 다) 생성
  - 입력은 단일 파일/폴더/글롭 패턴 지원

- `Toyproject-motionclassification/ms_g3d_basic_preprocessing.py`
  - 비디오 폴더 구조를 읽어 스켈레톤 시퀀스를 생성
  - `exercise_data_joint.npy`, `exercise_train_label.pkl`, `exercise_class_info.pkl` 저장
  - 알려진 5개 클래스 + `others` 클래스로 라벨 구성

## 설정(YAML)
- `Toyproject-motionclassification/yaml/train_*.yaml`, `test_*.yaml`
  - 학습/테스트 데이터 경로, 클래스 수(6), 그래프 구조 등 정의
  - 일부 파일에 Colab/Google Drive 절대 경로가 들어있으므로 로컬 환경에 맞게 수정 필요

## 데이터 포맷
- 스켈레톤 NPY: `N x C x T x V x M`
  - C=3(x,y,z), V=25, M=1
- `.skeleton` 텍스트: NTU RGB+D 포맷 호환

## 실행 예시
```bash
# MP4 → NPY 변환
python Toyproject-motionclassification/mp4_preprocess.py \
  --videos "./videos/*.mp4" --out_dir ./npy_data --T 300

# MP4 → .skeleton 또는 NPY
python Toyproject-motionclassification/data_mp4.py \
  --videos "./videos/*.mp4" --out_dir ./npy_data --format both

# 학습 (YAML 기반)
python Toyproject-motionclassification/2s-AGCN-0.0/main.py \
  --config Toyproject-motionclassification/yaml/train_joint.yaml
```

## 주의사항
- YAML과 일부 스크립트에 절대 경로가 존재합니다. 로컬 경로로 수정 후 사용하세요.
- `2s-AGCN-0.0`는 원본 코드 기반이므로, 필요한 데이터 구조를 맞춰야 정상 동작합니다.
