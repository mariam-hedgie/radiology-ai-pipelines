# Radiology AI Pipelines
Segmentation • Classification • Report Generation

This repository is a research-oriented monorepo containing three end-to-end radiology ML pipelines built to explore foundation models for medical imaging.

The goal of this repo is clarity and reproducibility, not turnkey deployment.

--------------------------------------------------

Pipelines Included

1) Segmentation
Task: Lung segmentation from Chest X-ray
Model: Frozen ViT-B encoder + UPerNet decoder

Encoder options:
- RAD-DINO (DINOv2 ViT-B/14)
- RAD-JEPA (ViT-B trained with JEPA self-supervised learning)

Decoder: UPerNet
Training: decoder-only (encoder frozen)
Metrics: Dice, IoU
Visual outputs: ground-truth overlay vs prediction overlay

Location:
segmentation/

--------------------------------------------------

2) Classification
Task: Pneumonia vs Normal classification
Model: Frozen ViT encoder + linear probe

Encoder: RAD-DINO
Head: linear classifier
Evaluation: Accuracy, Balanced Accuracy, Macro F1, ROC-AUC
Designed as a clean linear-probe baseline

Location:
classification/

--------------------------------------------------

3) Report Generation
Task: Chest X-ray → radiology Findings text

Architecture:
- Vision encoder: RAD-DINO (frozen)
- Projector: MLP mapping vision tokens to LLM embedding space
- LLM: Vicuna-7B with QLoRA + LoRA
- Training: projector + LoRA weights only
- Dataset: image + Findings text pairs

Location:
report_generation/

--------------------------------------------------

What This Repo Contains

- Full source code (src/)
- Training, evaluation, and inference scripts
- Visualization utilities (overlay images, comparison grids)
- Modular model definitions
- Reproducible experiment structure

--------------------------------------------------

What Is Intentionally NOT Included

To keep the repository lightweight and shareable, the following are not committed:

- Datasets (data/)
- Trained checkpoints (checkpoints/, checkpoints_jepa/)
- Experiment logs (wandb/)
- Virtual environments (.venv/)
- Confidential RAD-JEPA pretrained weights

You are expected to download datasets and configure paths locally.

--------------------------------------------------

Datasets Used

Segmentation:
- Chest X-ray lung segmentation datasets
- Public / Kaggle lung mask datasets
- Images resized for ViT compatibility

Classification:
- Chest X-ray Pneumonia dataset
- Binary labels: Normal vs Pneumonia

Report Generation:
- MIMIC-CXR
- Frontal images only
- Findings section used as target text

Refer to individual pipeline folders for expected directory structure.

--------------------------------------------------

Path Configuration (Important)

Some scripts contain hard-coded paths for simplicity and readability.

You are expected to modify these locally.

Common variables to search for:
DATA_ROOT
CKPT
JEPA_CKPT

Example:
DATA_ROOT = "data/lung_seg"
JEPA_CKPT = "/absolute/path/to/best_jepa_weights.pth.tar"

Environment-variable based configuration is intentionally not enforced.

--------------------------------------------------

RAD-JEPA Weights Notice

The file:
best_jepa_weights.pth.tar

is not distributed with this repository.

To run JEPA-based experiments, you must:
- Obtain the weights separately (private or internal source)
- Update the JEPA_CKPT path in segmentation scripts

The code assumes the JEPA checkpoint contains:
- An encoder state dict
- ViT-B compatible patch embedding and transformer blocks

--------------------------------------------------

Example Usage (Segmentation)

Training:
python -m src.train_full

Inference:
python scripts/infer_test.py

Evaluation:
python scripts/eval_test_dice_iou.py --ckpt checkpoints_jepa/best.pth --split test

Visualization grid:
python scripts/make_overlay_grid.py --epoch_dir outputs_jepa/test --n 24 --cols 4

--------------------------------------------------

Design Philosophy

- Prefer explicit, readable code over heavy abstractions
- Use frozen foundation models with lightweight task-specific heads
- Keep data flow visible and inspectable
- Make scripts easy to modify for ablation studies
- Optimize for research iteration, not production deployment

--------------------------------------------------

Notes

- Segmentation performance is sensitive to input resolution
- JEPA vs DINO comparisons are meaningful only when resolution and training schedules match
- Report generation code assumes familiarity with HuggingFace, PEFT, and large language models

--------------------------------------------------

License and Use

This repository is intended for research and educational use.
Datasets and pretrained weights are subject to their original licenses.