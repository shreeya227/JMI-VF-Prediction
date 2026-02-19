# JMI-VF-Prediction
## Environment Setup

Python 3.10
PyTorch 2.x
CUDA 11.x

### Each .npz file must contain:
'oct_bscans'  → 3D volume (200 × 200 × 200)
'tds'         → 52-dim visual field vector
'race'        → categorical label
'hispanic'    → categorical label

The model consists of:

A 3D ResNet-18 backbone for volumetric feature extraction

Channel-wise attention on pooled features

MLP-based demographic embedding

Group-specific calibration layers

A shared regression head for 52-point TD prediction
