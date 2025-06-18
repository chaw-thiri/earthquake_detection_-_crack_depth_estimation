# Earthquake Prediction Model


**Date**: June 19, 2025  
 

## Overview
This project develops a neural network model to predict earthquake `Magnitude` (Richter scale, ~0–10) and `Depth` (~0–700 km) using spatial-temporal features (`Timestamp`, `Latitude`, `Longitude`). The model aims to support seismic hazard assessment, a critical task in geophysics.
![Screenshot 2025-06-19 000932](https://github.com/user-attachments/assets/c9be64f3-406d-412a-b01b-21c301f34f19)

### Objectives
- Predict `Magnitude` and `Depth` as continuous variables (regression task).
- Achieve target performance:  
  - Standardized: MSE < 0.3, R² > 0.7.
  - Original scale: MSE < 0.04 for `Magnitude` (~0.2 units error), < 2500 for `Depth` (~50 km error).
- Visualize prediction errors spatially to guide improvements.

## Model Description
The model is a deep neural network implemented in TensorFlow/Keras, with separate architectures for `Magnitude` and `Depth`:
- **Architecture**: Functional API with three hidden layers (128 → 64 → 32 neurons, ReLU activation), batch normalization, dropout (0.2), and a linear output layer (1 neuron).
- **Input Features**: `Timestamp` (normalized to [0,1]), `Latitude`, `Longitude`.
- **Preprocessing**: 
  - Features standardized with `StandardScaler`.
  - Targets scaled separately for `Magnitude` and `Depth`.
- **Training**:
  - Optimizer: Adam.
  - Loss: Mean squared error (MSE).
  - Hyperparameters tuned via `GridSearchCV` (neurons=128, batch_size=64, epochs=100).
  - Callbacks: Early stopping (patience=10), learning rate reduction (factor=0.5, patience=5).
- **Evaluation Metrics**: MSE and R² in standardized space; MSE in original scale.

## Installation
```bash
pip install -r requirements.txt
```
## Dataset
Data: Earthquake dataset (database.csv) 

## Usage
1. Clone the repository:

```
git clone https://github.com/chaw-thiri/earthquake_detection_-_crack_depth_estimation.git
```
2. Place database.csv in the specified path (C:\Users\chawt\Desktop\earthquake detection\).
3. Run the script:
```
python main.py
```

## Performance 
### Magnitude:
Best Negative MSE (validation): -0.9909 (MSE ≈ 0.9909 standardized).
Test MSE (standardized): 1.0252
Test R² (standardized): 0.0227
MSE (original): 0.1817 (~0.426 magnitude units error)
### Depth:
Best Negative MSE (validation): -0.3474 (MSE ≈ 0.3474 standardized).
Test MSE (standardized): 0.2481
Test R² (standardized): 0.7541
MSE (original): 3722.4633 (~61.0 km error)


### Comparison with Previous Iterations
| Iteration | Magnitude MSE (std) | Magnitude R² (std) | Magnitude MSE (orig) | Depth MSE (std) | Depth R² (std) | Depth MSE (orig) |
|-----------|---------------------|--------------------|----------------------|-----------------|----------------|------------------|
| Initial (3-layer, GridSearchCV) | 0.9233 | N/A | N/A | 0.9233 | N/A | N/A |
| Deeper (3-layer, GridSearchCV) | 0.8769 | 0.1504 | 0.1834 | 0.8769 | 0.1504 | 10789.8740 |
| 4-layer (GridSearchCV) | 1.0276 | 0.0203 | 0.1822 | 0.4116 | 0.5920 | 6176.4036 |
| Simple (2-layer, Sequential) | 1.0252 | 0.0227 | 0.1817 | 0.3126 | 0.6901 | 4690.6951 |
| **Current (4-layer, GridSearchCV)** | **1.0252** | **0.0227** | **0.1817** | **0.2481** | **0.7541** | **3722.4633** |


Magnitude: Persistently poor performance (MSE ≈ 1.0, R² ≈ 0.02–0.15), no better than predicting the mean. Original MSE (~0.18, ~0.42 units error) is acceptable but far from the target (<0.04).
Depth: Significant improvement, surpassing standardized targets (MSE < 0.3, R² > 0.7) and approaching original MSE target (<2500). Outperforms the simpler Sequential model (MSE: 0.3126, R²: 0.6901).

## Analysis

### Strengths
Depth Prediction: Excellent performance (MSE ≈ 0.25, R² ≈ 0.75) due to spherical coordinates capturing spatial patterns (e.g., subduction zones). The deeper architecture with batch normalization and dropout generalizes well.
Preprocessing: Normalizing Timestamp, using spherical coordinates, and separate scalers ensured stability and improved Depth results.
Training: GridSearchCV, early stopping, and learning rate scheduling optimized Depth convergence.

### Weaknesses

Magnitude Prediction: Near-zero R² (0.0227) and high MSE (1.0252) indicate Timestamp and spherical coordinates lack predictive power for Magnitude, requiring geological features (e.g., fault stress).
Depth Limitation: Original MSE (3722.4633, ~61.0 km error) is slightly above the target (<2500, ~50 km).
Feature Insufficiency: Lack of geological context limits Magnitude and constrains further Depth improvement.

### Comparison to Targets

Magnitude: Far below targets (MSE < 0.3, R² > 0.7, original MSE < 0.04).
Depth: Meets standardized targets (MSE < 0.3, R² > 0.7), close to original MSE target (<2500).

