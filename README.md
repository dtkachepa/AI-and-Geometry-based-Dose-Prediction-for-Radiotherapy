# RANDose

Reproduction and extension of **RANDose**, a deep learning framework for 3D radiotherapy dose prediction using the OpenKBP head-and-neck dataset.

The project reproduces the baseline RANDose training and evaluation pipeline and investigates modifications to the model architecture and loss functions to improve dose prediction performance.

## Key Features

* 3D radiotherapy dose prediction with **PyTorch**
* Training and evaluation on the **OpenKBP** dataset
* Experiments with alternative model architectures and custom loss functions
* GPU-accelerated model training
* Experiment tracking with **Weights & Biases**
* Evaluation using dose and DVH-based metrics
* Dependency and environment management with `uv`

## Results

The reproduced baseline achieved a Dose Score of:

```text
2.937
```

Subsequent model and loss-function experiments improved the Dose Score to:

```text
2.900
```

This corresponds to an improvement of approximately **1.26%** over the reproduced baseline.

## Getting Started

### Clone the repository

```bash
git clone https://github.com/dtkachepa/AI-and-Geometry-based-Dose-Prediction-for-Radiotherapy.git
cd AI-and-Geometry-based-Dose-Prediction-for-Radiotherapy
```

### Install dependencies

```bash
uv sync
```

### Run training

```bash
uv run python train.py \
    --model Model_MTASP \
    --loss Loss_DC_PTV \
    --batch_size 2 \
    --list_GPU_ids 0 \
    --max_iter 80000 \
    --project_name RANDose_reproduction
```

## Evaluation

Model performance is evaluated using the dose and dose-volume histogram (DVH) metrics used in the OpenKBP evaluation pipeline.

## Tech Stack

**Python · PyTorch · Weights & Biases · uv · CUDA**
