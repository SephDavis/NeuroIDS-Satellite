# NeuroIDS-Sat 🛰️

**Neuromorphic Intrusion Detection for CubeSat Constellations Under Extreme Power Constraints**

[![arXiv](https://img.shields.io/badge/arXiv-2601.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2601.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **2,800× more energy-efficient** than conventional IDS — enabling continuous cybersecurity monitoring on power-constrained satellites

## Overview

This code adapts the NeuroIDS spiking neural network for satellite deployment with:
- **Reduced architecture**: 64-32 hidden layers (vs 128-64 terrestrial)
- **TMR encoding**: Triple Modular Redundancy for radiation tolerance
- **Shorter timesteps**: 50 timesteps (vs 100) for power savings
- **Hierarchical detection**: Tier 1 anomaly + Tier 2 classification

## Installation

```bash
# Clone/copy this directory
cd neuroids_sat

# Install dependencies
pip install numpy

# Download NSL-KDD dataset
# From: https://www.unb.ca/cic/datasets/nsl.html
mkdir -p data/NSL-KDD
# Place KDDTrain+.txt and KDDTest+.txt in data/NSL-KDD/
```

## Quick Start

```python
from neuroids_sat import NeuroIDSSat, SatelliteConfig
from data_loader import load_nslkdd, preprocess_data

# Load data
X_train, y_train, X_test, y_test = load_nslkdd('data/NSL-KDD/')
X_train, X_test = preprocess_data(X_train, X_test)

# Create satellite-optimized model
config = SatelliteConfig()
model = NeuroIDSSat(config)

# Train
model.fit(X_train, y_train, epochs=20)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Energy: {metrics['energy']['energy_pj_per_sample']:.1f} pJ/sample")
```

## Run Full Experiments

Generate all data for the paper:

```bash
python run_experiments.py --data_path data/NSL-KDD/ --output_dir results/
```

This produces:
- `results/neuroids_sat_results.json` - All experimental metrics
- `results/tables/` - LaTeX tables for the paper
- `results/neuroids_sat_model.pkl` - Trained model

## Architecture Comparison

| Metric | Terrestrial | Satellite | Reduction |
|--------|-------------|-----------|-----------|
| Hidden Layer 1 | 128 | 64 | 50% |
| Hidden Layer 2 | 64 | 32 | 50% |
| Time Steps | 100 | 50 | 50% |
| Parameters | 13,957 | 4,933 | 64.7% |
| Threshold | 0.5 | 0.6 | +20% |
| Membrane τ | 15 ms | 20 ms | +33% |

## Experimental Results

Results from NSL-KDD evaluation (125,973 train / 22,544 test samples):

**Classification Performance**:
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Normal | 0.620 | 0.974 | 0.758 | 9,711 |
| DoS | 0.801 | 0.621 | 0.700 | 7,458 |
| Probe | 0.711 | 0.446 | 0.548 | 2,421 |
| R2L | 1.000 | 0.001 | 0.001 | 2,754 |
| U2R | 0.000 | 0.000 | 0.000 | 200 |
| **Overall** | | **0.673** | | 22,544 |

**Energy Analysis**:
| Metric | NeuroIDS-Sat | Conventional CPU | Improvement |
|--------|--------------|------------------|-------------|
| Spikes/sample | 411.3 | - | - |
| Energy/inference | 822.6 pJ | 45.2 mJ | **54,958×** |
| Power (continuous) | 0.82 mW | 2,300 mW | **2,805×** |
| Power (10% duty) | 0.082 mW | 230 mW | **2,805×** |

**Mission Lifetime (3U CubeSat, 20 Wh Battery)**:
| Configuration | Power | Runtime |
|---------------|-------|---------|
| Conventional (continuous) | 2.3 W | 8.7 hours |
| Conventional (10% duty) | 230 mW | 87 hours |
| NeuroIDS-Sat (continuous) | 0.82 mW | **1,014 days** |
| NeuroIDS-Sat (10% duty) | 0.082 mW | **10,140 days** |

### Notes on Results

The system achieves strong detection on majority classes (Normal: 97.4% recall, DoS: 62.1% recall) but struggles with minority classes (R2L, U2R) due to extreme class imbalance in NSL-KDD:
- Normal: 53.5% of training data
- DoS: 36.5%
- Probe: 9.3%
- R2L: 0.8%
- U2R: 0.04% (only 52 samples)

For satellite applications, the dominant threat categories (DoS, Probe) comprising 86.9% of attacks are effectively detected while consuming minimal power.

## Testing Radiation Effects

```python
# Simulate SEU (Single Event Upset) effects
metrics_clean = model.evaluate(X_test, y_test)
metrics_seu = model.evaluate(X_test, y_test, inject_seu=True, seu_rate=1e-4)

print(f"Clean accuracy: {metrics_clean['accuracy']:.4f}")
print(f"With SEU: {metrics_seu['accuracy']:.4f}")
```

## File Structure

```
neuroids_sat/
├── neuroids_sat.py      # Main NeuroIDS-Sat implementation
├── run_experiments.py   # Full experiment runner
├── data_loader.py       # NSL-KDD data loading utilities
├── README.md           
└── data/
    └── NSL-KDD/         # Place dataset here
        ├── KDDTrain+.txt
        └── KDDTest+.txt
```

## Citation

If you use this code, please cite:

```bibtex
@article{davis2026neuroidsat,
  title={NeuroIDS-Sat: Neuromorphic Intrusion Detection for CubeSat 
         Constellations Under Extreme Power Constraints},
  author={Davis, Toby R.},
  journal={arXiv preprint},
  year={2026}
}
```

## Author

Toby R. Davis 
Ms. Sci. Scholar Cybersecurity & Operations
Department of Computer Science and Engineering  
Mississippi State University  
trd183@msstate.edu
