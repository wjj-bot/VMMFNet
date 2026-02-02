# VMMFNet: A Lightweight Dual-Branch Multispectral Fusion Network for PWD Detection

**VMMFNet: A Lightweight Dual-Branch Multispectral Fusion Network for Multi-Stage Detection of Pine Wilt Disease**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

---

---

## 🛰️ Overall Research Design and Workflow

To provide a clear understanding of the study, the complete experimental pipeline—from biological inoculation to model evaluation—is summarized in the figure below:

![VMMFNet Research Workflow](images/image_f4943e.png)

### Workflow Description:
The study is organized into five systematic phases:
1. **Phase 1: PWN Inoculation & Monitoring**: Continuous monitoring of pine trees inoculated with Pine Wood Nematode (PWN) from April to July 2025 to identify sensitive spectral bands and vegetation indices.
2. **Phase 2: Data Acquisition**: UAV flight planning and 3D reconstruction to generate comprehensive visible (RGB) and multispectral (MS) maps.
3. **Phase 3: Dataset Construction**: Manual annotation of infection stages (Early, Mid, Late) and partitioning into training, validation, and test sets (8:1:1 ratio).
4. **Phase 4: Architecture Design**: Implementation of **VMMFNet**, featuring the **LWANet** backbone for feature extraction and **HMAFNet** neck for multi-scale fusion.
5. **Phase 5: Performance Evaluation**: Comprehensive testing including comparative experiments with different datasets, fusion strategies, and ablation studies.

---
## 🌟 Highlights

* **Experimental Grounding**: Conducted controlled PWN inoculation to identify early-stage sensitive spectral bands and vegetation indices.
* **Annotated Dataset**: Constructed a multi-stage dataset integrating synchronized UAV visible and multispectral imagery with precise annotations.
* **Lightweight Fusion (LWANet)**: Proposed a dual-branch backbone for efficient cross-modality feature extraction and lightweight fusion.
* **Noise Suppression (HMAFNet)**: Designed a specialized neck to mitigate background noise and enhance multi-scale feature representation.
* **Micro-target Detection**: Achieved robust four-scale detection optimized for tiny (**4×4 pixels**, ~3–25 cm) to large-scale PWD targets.



---

## 📁 Repository Structure

```text
├── configs/                # Model configuration files (AdamW, lr=0.01, etc.)
├── data_processing/        # Scripts for tiling and spatial anti-leakage checks
│   ├── uav_dataset_construction.py
│   └── spatial_anti_leakage_check.py
├── models/                 # LWANet backbone and HMAFNet implementation
├── train.py                # Main training script (AMP & EMA enabled)
├── val.py                  # Evaluation and inference script
├── metadata.pdf            # Technical specs (M3M bands, flight parameters)
└── README.md
