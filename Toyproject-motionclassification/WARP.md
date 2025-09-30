# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a motion classification project based on **Two-Stream Adaptive Graph Convolutional Networks (2s-AGCN)** for skeleton-based action recognition. The implementation supports both joint data and bone data processing for improved accuracy through ensemble methods.

### Key Architecture Components

- **2s-AGCN Model** (`2s-AGCN-0.0/model/agcn.py`): Main graph convolutional network with adaptive temporal and spatial graph learning
- **Data Feeders** (`2s-AGCN-0.0/feeders/`): Handles skeleton data loading with augmentation options (random shift, choose, move)  
- **Graph Definitions** (`2s-AGCN-0.0/graph/`): NTU RGB+D and Kinetics skeleton topology definitions
- **Data Preprocessing** (`2s-AGCN-0.0/data_gen/`): Scripts to generate joint and bone data from raw skeleton files

The model processes skeleton sequences as graphs where joints are nodes and bones are edges, using learnable adjacency matrices for adaptive spatial-temporal feature extraction.

## Common Commands

### Training
```bash
# Train on joint data
python 2s-AGCN-0.0/main.py --config 2s-AGCN-0.0/config/nturgbd-cross-view/train_joint.yaml

# Train on bone data  
python 2s-AGCN-0.0/main.py --config 2s-AGCN-0.0/config/nturgbd-cross-view/train_bone.yaml

# Custom training with toy project config (6 classes)
python 2s-AGCN-0.0/main.py --config yaml/train_joint.yaml
```

### Testing
```bash  
# Test joint model and save scores
python 2s-AGCN-0.0/main.py --config 2s-AGCN-0.0/config/nturgbd-cross-view/test_joint.yaml

# Test bone model and save scores
python 2s-AGCN-0.0/main.py --config 2s-AGCN-0.0/config/nturgbd-cross-view/test_bone.yaml
```

### Ensemble Results
```bash
# Combine joint and bone predictions
python 2s-AGCN-0.0/ensemble.py --datasets ntu/xview --alpha 1
```

### Data Preprocessing
```bash
# Generate processed data from raw NTU RGB+D
python 2s-AGCN-0.0/data_gen/ntu_gendata.py

# Generate bone data from joint data  
python 2s-AGCN-0.0/data_gen/gen_bone_data.py
```

### Single Sample Prediction
```bash
# Predict on individual .npy files with different padding strategies
python 2s-AGCN-0.0/predict_single_npy_centerpad.py
python 2s-AGCN-0.0/predict_single_npy_leftpad.py
```

## Configuration Structure

YAML config files control all training parameters:
- **Model args**: `num_class` (60 for NTU, 6 for toy project), `num_point` (25 joints), `num_person`
- **Data paths**: Separate configs for joint/bone data, train/test splits
- **Training**: Learning rate scheduling, batch sizes, device assignments
- **Graph topology**: NTU RGB+D skeleton structure with 25 joints

Key insight: The project uses a two-stream approach where joint coordinates and bone vectors are processed separately, then ensembled for final predictions.

## Fine-tuning Setup

The `yaml/` directory contains configs for fine-tuning pre-trained models on smaller custom datasets:
- Reduced `num_class` from 60 to 6 actions
- Single person setup (`num_person: 1`)
- Custom data paths for fine-tuning datasets

## Data Format

Skeleton data format: `N × C × T × V × M`
- N: batch size  
- C: coordinate dimensions (3 for x,y,z)
- T: temporal frames
- V: vertices/joints (25 for NTU)
- M: maximum persons (1 or 2)