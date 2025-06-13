---
title: BrainVista – Stroke Detection using Hybrid ML & DL Models
app_file: BrainVista_combined.py
sdk: python
sdk_version: 3.10+
---

# 🧠 BrainVista – Stroke Classification from CT Brain Images

Stroke remains a leading cause of death and disability globally, and timely detection is critical. **BrainVista** is a unified research framework developed for stroke detection using both traditional **machine learning (ML)** and **deep learning (DL)** models on grayscale CT images. It integrates various classification algorithms, model optimization strategies, and feature extraction pipelines to compare performance across methods.

---

## 📊 Project Highlights

- **Dataset Input**: Grayscale CT brain scans of normal vs. stroke patients.
- **Preprocessing**: Image resizing, grayscale normalization, flattening, and PCA/feature selection.
- **Model Diversity**: Includes both classic ML (SVM, LR, DT, RF, NB) and DL (ANN, DNN, RNN, CNN, AlexNet, InceptionV3, NASNet, ShuffleNet, VGG19).
- **Evaluation**: 5-Fold Cross Validation with metrics like Accuracy, Precision, Recall, F1, and AUC.
- **Deployment Ready**: Modular code structure for both Google Colab & local systems.

---

## 🔬 Methodology

### 1. 📁 Data Loading & Preparation
- Loads images from structured folders (`Normal`, `Stroke`) in Google Drive.
- Images are resized (227x227 / 224x224 / 299x299 / 331x331 depending on the model).
- Converted to grayscale or RGB as required by the model.
- Labels are binary: `0` for Normal, `1` for Stroke.

### 2. 🧠 Models Implemented

| Model Type | Model Name           | Input Size   | Feature Engineering     |
|------------|----------------------|--------------|--------------------------|
| ML         | SVM (RBF Kernel)     | 227×227      | Standard Scaling         |
| ML         | Logistic Regression  | 227×227      | MinMax + Gradient Descent |
| ML         | Decision Tree        | 227×227      | Raw Pixels               |
| ML         | Random Forest        | 227×227      | Scaled & Tuned Trees     |
| ML         | Naive Bayes          | 227×227      | Variance Threshold + PCA |
| DL         | ANN, DNN             | 227×227×1    | Normalized               |
| DL         | RNN                  | 227×227      | Sequence-like format     |
| DL         | AlexNet (Modified)   | 227×227×1    | CNN + Feature Extractor  |
| DL         | InceptionV3          | 299×299×3    | Transfer Learning        |
| DL         | VGG19-GAP            | 224×224×3    | CNN + Global Avg Pooling |
| DL         | NASNetLarge          | 331×331×3    | Transfer Learning        |
| DL         | ShuffleNetV2         | 224×224×3    | CNN + Channel Shuffle    |

---

## 📂 File Overview

| Filename                    | Description                                         |
|-----------------------------|-----------------------------------------------------|
| `BrainVista_combined.py`    | Main unified script for training & evaluation       |
| `*.npy`                     | Saved feature vectors from CNN models               |
| `*.h5`, `*.keras`           | Saved trained DL models                             |
| `Normal/`, `Stroke/`        | Input image folders in Drive                        |

---

## 🧪 Evaluation Metrics

Each model is evaluated using:

- **Accuracy**
- **Precision / Recall**
- **F1 Score**
- **AUC (Area Under ROC Curve)**
- **Classification Report**
- **ROC Curve Visualization (for DL models)**

---

## 🛠 Setup & Requirements

Install all required packages:

```bash
pip install -r requirements.txt


