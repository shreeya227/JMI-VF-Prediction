# Fairness-aware deep learning for predicting visual field loss from optical coherence tomography


This repository contains the code used for the manuscript:

**Fairness-aware deep learning for predicting visual field loss from optical coherence tomography**  
Journal of Medical Imaging submission: **JMI-260093**

The project implements a fairness-aware volumetric deep learning framework for predicting 52-point Humphrey 24-2 visual field total deviation values from optical coherence tomography (OCT) B-scan volumes. The framework combines 3D OCT feature learning, demographic embedding, subgroup-specific calibration, and Adaptive Fairness Feedback (AFF).

## Repository Status

This repository corresponds to the revised JMI submission after major revision.

The revised experiments use:

- Patient-level train, validation, and test splitting
- Internal validation-based checkpoint selection
- Held-out test evaluation only after final model selection
- Ablation experiments comparing:
  - 3D ResNet without demographic embedding or AFF
  - 3D ResNet with demographic embedding only
  - 3D ResNet with demographic embedding and AFF

## Repository Structure

```text
JMI-VF-Prediction/
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_handler.py
│   └── model.py
└── scripts/
    └── train_vf_fair.py
