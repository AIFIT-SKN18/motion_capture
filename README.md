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
