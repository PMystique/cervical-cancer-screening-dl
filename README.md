# Deep Learning-Based Cervical Cancer Screening

> **Course:** MSB7216 — Deep Learning for Health Data, Makerere University  
> **Component:** Final Examination Practical Project Submission  
> **Author:** Apio Patricia Mystica | Reg No: 2025/HD07/25981U  
> **Date:** May 30, 2026  
> **Supervisor:** Mr. Solomon Nsumba

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PMystique/cervical-cancer-screening-dl/blob/main/notebooks/03_Transfer_Learning.ipynb)
[![Live Demo](https://img.shields.io/badge/🤗%20Gradio-Live%20Demo-blue)](https://huggingface.co/spaces/PMystique/Automated_CaCx_Screening)
[![Dataset](https://img.shields.io/badge/Dataset-CPSMI2025%20Mendeley-orange)](https://data.mendeley.com/datasets/rnfyhrtfkw/1)

---

## Overview

An automated end-to-end computer-aided screening pipeline for cervical cytology classification from conventional Pap smear microscopy images. The pipeline benchmarks three deep learning models — a custom 4-block CNN baseline, fine-tuned **ResNet18**, and fine-tuned **EfficientNet_V2_S** — across four diagnostic classes: `cancer`, `lesion`, `normal`, and `others`.

| Model | Test Accuracy | Macro F1 | Cancer Recall | Lesion Recall | Macro AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| Custom Baseline CNN | 70.4% | 0.64 | 80% | 54% | — |
| ResNet18 (fine-tuned) | 85.0% | **0.79** | 88% | 76% | 0.94 |
| EfficientNet_V2_S (fine-tuned) | 85.0% | 0.78 | **89%** | **83%** | 0.93 |

**EfficientNet_V2_S** is the recommended deployment model due to its superior recall on malignant and precancerous classes.

---

## Project Structure

```text
project/
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb     # Dataset exploration, flattening, augmentation
│   ├── 02_Baseline_Model.ipynb            # Custom 4-block CNN from scratch
│   ├── 03_Transfer_Learning.ipynb         # ResNet18 & EfficientNet_V2_S fine-tuning
│   └── 04_GradCAM_Explainability.ipynb    # Grad-CAM heatmap generation & analysis
├── src/
│   └── app.py                             # Gradio deployment demo
├── figures/                               # All generated plots and visualizations
├── reports/
│   ├── Final_Project_Report_Deep_Learning_Patricia_Mystica_Apio.pdf     # Academic report (PDF)
│   └── Final_Project_Presentation_Deep_Learning_Patricia_Apio.pptx      # Presentation slides
├── models/                                # Saved .pth checkpoints (see note below)
├── References/                            # Reference PDFs
├── requirements.txt                       # Python dependencies
├── .gitignore
└── README.md
```

---

## Dataset

**CPSMI2025** — Curated Dataset of Conventional Pap Smear Microscopy Images  
- **Source:** [Mendeley Data](https://data.mendeley.com/datasets/rnfyhrtfkw/1) (DOI: 10.17632/rnfyhrtfkw.1)  
- **Images:** 2,169 conventional microscopy slides  
- **Classes:** `cancer` (92 test), `lesion` (102 test), `normal` (112 test), `others` (22 test)  
- **Split:** 70% Train / 15% Val / 15% Test (stratified, seed = 42)

> **To use the dataset:** Download from Mendeley Data and place the extracted folders into `data/`. Then run `01_EDA_and_Preprocessing.ipynb` which flattens the nested structure automatically.

---

## Setup & Usage

### Google Colab (Recommended)

1. Open any notebook directly in Colab via the badge above.
2. Mount your Google Drive (`drive.mount('/content/drive')`).
3. Place the dataset in Drive and update the `DATA_DIR` path variable at the top of each notebook.
4. Run cells sequentially. The T4 GPU runtime is recommended.

### Local Environment

```bash
# 1. Clone the repository
git clone https://github.com/PMystique/cervical-cancer-screening-dl.git
cd cervical-cancer-screening-dl

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks in order
jupyter lab notebooks/
```

### Run the Gradio Demo Locally

```bash
python src/app.py
```

Or visit the live hosted demo: **[HuggingFace Spaces](https://huggingface.co/spaces/PMystique/Automated_CaCx_Screening)**

---

## Notebooks Guide

| Notebook | Description |
|:---|:---|
| `01_EDA_and_Preprocessing.ipynb` | Flattens nested CPSMI2025 folders, stratified split, EDA plots, augmentation visualization |
| `02_Baseline_Model.ipynb` | Trains custom 4-block CNN from scratch; confusion matrix, ROC curve |
| `03_Transfer_Learning.ipynb` | Fine-tunes ResNet18 & EfficientNet_V2_S; 30-epoch training with checkpointing |
| `04_GradCAM_Explainability.ipynb` | Generates Grad-CAM heatmaps for all classes; biological interpretation |

---

## Key Results

- **+14.6 pp accuracy gain** with transfer learning (70.4% → 85.0%)  
- **EfficientNet_V2_S**: 89% cancer recall, 83% lesion recall — minimizes dangerous false negatives  
- **Grad-CAM validated**: models attend to hyperchromatic nuclei and koilocytic vacuoles, not background artefacts  
- **Macro AUC 0.93–0.94**: robust discriminative ability at all decision thresholds  
- **`others` class retained** as a clinical quality-control flag for indeterminate slides

---

## Model Checkpoints

Trained `.pth` checkpoints are **not committed** to this repository due to file size (50–82 MB each).

| Model | Val Accuracy (best epoch) | File |
|:---|:---:|:---|
| Baseline CNN | — | `models/baseline_cnn_best.pth` |
| ResNet18 | 83.08% (epoch 19) | `models/resnet18_best.pth` |
| EfficientNet_V2_S | 87.08% (epoch 28) | `models/efficientnet_best.pth` |

To obtain the checkpoints, re-run `03_Transfer_Learning.ipynb` — they will be saved to `models/` automatically.

---

## Deliverables

| Deliverable | Location |
|:---|:---|
| Academic Report (PDF) | `reports/Final_Project_Report_Deep_Learning_Patricia_Mystica_Apio.pdf` |
| Presentation Slides | `reports/Final_Project_Presentation_Deep_Learning_Patricia_Apio.pptx` |
| Live Gradio Demo | [HuggingFace Spaces](https://huggingface.co/spaces/PMystique/Automated_CaCx_Screening) |

---

## References

1. Ocampo-López-Escalera et al. (2025). *Frontiers in Medical Technology*, 7, 1531817.  
2. He et al. (2016). Deep Residual Learning. *CVPR*.  
3. Tan & Le (2019). EfficientNet. *ICML*.  
4. Tan et al. (2021). EfficientNetV2. *ICML*.  

Full references are in the academic report PDF.

---

## License

This project is for academic research and educational use only. The CPSMI2025 dataset is licensed under the terms specified by Mendeley Data.
