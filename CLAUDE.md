# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GARF (Learning Generalizable 3D Reassembly for Real-World Fractures) is a deep learning project for 3D object reassembly using PyTorch and PyTorch Lightning. It's an academic research project focused on reconstructing fractured objects using flow matching and diffusion techniques.

## Essential Commands

### Environment Setup
```bash
# Install dependencies using uv package manager
uv sync
uv sync --extra post
source .venv/bin/activate
```

### Training Commands
```bash
# Stage 1: Fracture-aware Pretraining
python train.py experiment=pretraining_frac_seg data.categories=$DATA_CATEGORIES trainer.num_nodes=$NUM_NODES data.data_root=$DATA_ROOT

# Stage 2: Flow-Matching Training
python train.py experiment=denoiser_flow_matching data.categories=$DATA_CATEGORIES trainer.num_nodes=$NUM_NODES data.data_root=$DATA_ROOT model.feature_extractor_ckpt=$FEATURE_EXTRACTOR_CKPT

# Stage 3 (Optional): LoRA Fine-tuning
python train.py experiment=finetune data.categories="['egg']" data.data_root=./finetune_egg.hdf5 finetuning=true
```

### Evaluation
```bash
# Run evaluation with flow matching
HYDRA_FULL_ERROR=1 python eval.py seed=42 experiment=denoiser_flow_matching experiment_name=$EVAL_NAME data.data_root=$DATA_ROOT data.categories=$DATA_CATEGORIES ckpt_path=$CHECKPOINT_PATH

# Run evaluation with diffusion variant
HYDRA_FULL_ERROR=1 python eval.py seed=42 experiment=denoiser_diffusion experiment_name=$EVAL_NAME data.data_root=$DATA_ROOT data.categories=$DATA_CATEGORIES ckpt_path=$CHECKPOINT_PATH
```

### Demo
```bash
# Launch interactive Gradio web interface
python app.py
```

## Architecture Overview

### Core Components

1. **Feature Extraction**: PointTransformerV3 backbone in `assembly/backbones/`
2. **Denoising Models**: Flow matching and diffusion implementations in `assembly/models/denoiser/`
3. **Data Pipeline**: Breaking Bad dataset processing in `assembly/data/breaking_bad/`
4. **Configuration System**: Hydra-based configs in `configs/` with hierarchical overrides

### Training Pipeline

1. **Pretraining**: Fracture-aware feature learning using segmentation objectives
2. **Main Training**: Flow matching (primary) or diffusion-based reassembly training
3. **Fine-tuning**: Optional LoRA-based adaptation for specific datasets

### Key Design Patterns

- **Lightning Modules**: All models inherit from `LightningModule` for distributed training
- **Hydra Configs**: Nested YAML configurations with command-line overrides
- **HDF5 Data Format**: Efficient storage for point cloud fragments and transformations
- **Modular Architecture**: Clear separation between backbones, denoising models, and evaluation metrics

## Important Notes

- **No Traditional Tests**: Project uses evaluation scripts (`eval.py`) rather than unit tests
- **Single GPU Evaluation**: Multi-GPU evaluation may produce incomplete `json_results`
- **CUDA Requirements**: Flash attention requires CUDA 12.0+, PyTorch3D needs GPU during installation
- **Config Precedence**: Command-line args override experiment configs which override default configs