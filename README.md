# SSTT: Interaction-Aware Trajectory Prediction Method Based on Sparse Spatial-Temporal Transformer for Internet of Vehicles

**Sparse Spatial-Temporal Transformer (SSTT)** for vehicle trajectory prediction.

This repository contains the official implementation of the paper **"Interaction-Aware Trajectory Prediction Method Based on Sparse Spatial-Temporal Transformer for Internet of Vehicles"** published in *IEEE Transactions on Intelligent Transportation Systems*.

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20TITS-blue)](https://ieeexplore.ieee.org/)

## :newspaper: Overview

We propose a novel **Sparse Spatial-Temporal Transformer (SSTT)** model for interaction-aware trajectory prediction in Internet of Vehicles (IoV) scenarios. The model leverages sparse attention mechanisms to efficiently capture both spatial interactions among neighboring vehicles and temporal dependencies in trajectory sequences, achieving state-of-the-art performance on the NGSIM dataset.

## :rocket: Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- Ubuntu 18.04 or later

### Installation

1. Clone the repository:
```bash
git clone https://github.com/lxhy9799/SSTT.git
cd SSTT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - Download from [Google Drive](https://drive.google.com/drive/folders/1yiNNOMC1twnzxCn4HclaSlMzBoKKSz5Q?usp=drive_link)
   - Place the downloaded files in the `data/ngsim/` directory

### Project Structure

```
SSTT/
├── data/
│   └── ngsim/
│       └── TestSet.mat          # Test dataset
├── trained_models/
│   └── SSTT_ngsim.pth          # Pre-trained model
├── evaluate.py                  # Evaluation script
├── loader.py                    # Data loader
├── model.py                     # Model architecture
├── requirements.txt             # Dependencies
└── README.md
```

## :computer: Usage

### Evaluation

Run the evaluation script with the pre-trained model:

```bash
python evaluate.py
```

### Custom Configuration

You can customize the evaluation parameters:

```bash
python evaluate.py \
    --batch_size 128 \
    --num_workers 4 \
    --test_set data/ngsim/TestSet.mat
```

## :trophy: Results

Based on our pre-trained model, you can reproduce the prediction results presented in our paper:

![Prediction Results](img.png)

### Performance Metrics

The model is evaluated using:
- **RMSE (Root Mean Square Error)**: Measured in meters, converted from feet using the factor 0.3048
- **FDE (Final Displacement Error)**: Error at the final prediction timestep
- **Horizontal Evaluation**: Performance across different prediction horizons

## :construction: Development

The repository now includes the complete source code implementation, including:
- Model architecture (`model.py`)
- Training and evaluation scripts (`evaluate.py`)
- Data loading utilities (`loader.py`)
- Pre-trained models for NGSIM dataset

## :handshake: Acknowledgements

We sincerely appreciate the following open-source projects for their excellent codebases:

- [CS-LSTM](https://github.com/nachiket92/conv-social-pooling): Convolutional Social Pooling
- [STDAN](https://github.com/xbchen82/stdan): Spatial-Temporal Attention Network
- [HLTP](https://github.com/Petrichor625/HLTP): Teacher-student Trajectory Prediction

## :page_facing_up: Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{li2025interaction,
  title={Interaction-Aware Trajectory Prediction Method Based on Sparse Spatial-Temporal Transformer for Internet of Vehicles},
  author={Li, Xunhao and Zhang, Jian and Chen, Jun and Qian, Pinzheng},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025},
  publisher={IEEE}
}
```
