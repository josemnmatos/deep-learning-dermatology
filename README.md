# Multiclass Skin Lesion Classification using Deep Learning on DermaMNIST

**Author:** José Matos  ([Github](https://github.com/josemnmatos)) ([LinkedIn](https://linkedin.com/in/josemnmatos))

<!-- Optional Badges: Add more as needed -->
![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg) <!-- Choose your license -->
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)

## Overview

This project explores the automated classification of 7 types of skin lesions using deep learning models on the low-resolution DermaMNIST dataset. Accurate and automated classification of skin lesions is crucial for aiding clinical diagnosis, but presents challenges due to visual similarities between different conditions.

This work compares the performance of three distinct neural network architectures:
*   Multi-Layer Perceptrons (MLP)
*   Convolutional Neural Networks (CNN)
*   Vision Transformers (ViT)

The models were optimized through rigorous hyperparameter tuning, focusing on the Macro-F1 score to effectively handle the inherent class imbalance in the dataset. Techniques like data augmentation and early stopping were employed to improve generalization and prevent overfitting. A custom CNN architecture was developed and evaluated against the other models and established benchmarks.

## Key Features & Highlights

*   **Comparative Analysis:** In-depth performance comparison of MLP, CNN, and ViT on a challenging, low-resolution medical imaging task.
*   **Custom CNN Optimization:** Development and tuning of a tailored CNN architecture demonstrating strong performance even with low-resolution images.
*   **Class Imbalance Handling:** Explicit focus on addressing class imbalance using the Macro-averaged F1 score (MAF1) as the primary evaluation metric during tuning.
*   **Rigorous Evaluation:** Comprehensive evaluation using Accuracy (ACC), Area Under the ROC Curve (AUC - Micro and Macro), and Confusion Matrices, validated over multiple runs with confidence intervals.
*   **Reproducibility:** Clear pipeline structure (`pipeline.py`) and dependency management (`requirements.txt`) for easier replication of results.
*   **Competitive Performance:** The optimized custom CNN achieved competitive results, notably outperforming standard ResNet benchmarks on the 28x28 version of the dataset and achieving a high AUC score.

## Visual Insights

### Dataset Samples
A glimpse into the 7 classes of skin lesions present in the DermaMNIST dataset (28x28 resolution):

![DermaMNIST Sample Images](figures/dermamnist_samples.png)
*Fig 1: Samples from each class of the DermaMNIST dataset (Source: Project Report)*

### Model Performance

**Average Loss Curves (CNN Example):** Demonstrates training stability and prevention of overfitting using data augmentation and early stopping.
![Average Loss Curves](figures/cnn_avg_loss_curves.png) <!-- Make sure you save this plot -->
*Fig 2: Average Train and Validation Loss for the optimized CNN over 5 runs.*

**Final ROC Curves (Mean AUC over 5 runs):** Shows the discriminative ability of the optimized models on the test set. ViT achieves the highest overall AUC, but CNN shows strong performance across classes relevant for MAF1/ACC.
![Final ROC Curves](figures/final_roc_curves.png) <!-- Make sure you save this plot from final_model_eval -->
*Fig 3: Mean ROC curves for MLP, CNN, and ViT evaluated on the test set.*

**Average Confusion Matrix (Optimized CNN):** Provides insight into class-specific performance and common misclassifications for the best-performing model (CNN).
![Average Confusion Matrix](figures/cnn_avg_confusion_matrix.png) <!-- Make sure you save this plot -->
*Fig 4: Average Confusion Matrix for the optimized CNN over 5 runs on the test set.*

## Dataset: DermaMNIST

*   **Source:** A component of the [MedMNIST v2](https://medmnist.com/) collection, derived from the [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).
*   **Content:** 10,015 dermatoscopic images of 7 skin lesion types.
*   **Format:** Images resized to a low resolution of 28x28 pixels with 3 color channels (RGB).
*   **Split:** Standardized 7:1:2 ratio for training, validation, and test sets.
*   **Classes (7):**
    0.  Actinic keratoses and intraepithelial carcinoma (akiec)
    1.  Basal cell carcinoma (bcc)
    2.  Benign keratosis-like lesions (bkl)
    3.  Dermatofibroma (df)
    4.  Melanoma (mel)
    5.  Melanocytic nevi (nv)
    6.  Vascular lesions (vasc)

## Results Summary

The final performance of the optimized models was evaluated on the unseen DermaMNIST test set, averaged over 5 independent runs (Mean ± 95% CI).

| Model | Test MAF1             | Test ACC              | Test AUC (Micro)      |
| :---- | :-------------------- | :-------------------- | :-------------------- |
| MLP   | 0.3593 ± 0.0458       | 0.7165 ± 0.0046       | 0.9438 ± 0.0016       |
| **CNN** | **0.5309 ± 0.0166**   | **0.7664 ± 0.0071**   | 0.9477 ± 0.0046       |
| ViT   | 0.4804 ± 0.0396       | 0.7500 ± 0.0052       | **0.9514 ± 0.0017**   |

* **CNN:** Achieved the highest Macro-F1 and Accuracy, indicating the best balance in classifying across all, including minority, classes. Its performance suggests tailored CNNs can be very effective even for low-resolution medical images.
* **ViT:** Achieved the highest Micro-AUC, showing excellent overall discriminative power, but slightly lower MAF1/ACC suggests potential struggles with specific minority classes compared to the CNN in this setup.
* **MLP:** Performed poorly, highlighting the importance of spatial feature extraction (Convolution or Attention) for image tasks.

Notably, all developed models surpassed the highest AUC (0.920) reported in the original MedMNIST v2 paper (using ResNet-18 on 224x224 images). The custom CNN's accuracy (0.766) nearly matched the benchmark set by Google AutoML Vision (0.768) while achieving a significantly higher AUC.

## Technology Stack

*   Python 3.8+
*   PyTorch
*   MedMNIST
*   Scikit-learn
*   NumPy
*   Matplotlib
*   Seaborn
*   Pandas
*   [vit-pytorch](https://github.com/lucidrains/vit-pytorch) (for ViT experiments)

## Directory Structure

```└── josemnmatos-ml-dermatology-project/
    ├── cnn_experiments.ipynb         # Jupyter Notebook for CNN experiments (also .py)
    ├── final_model_eval.ipynb        # Jupyter Notebook for final model evaluation (also .py)
    ├── helper_methods.py             # Utility functions (data loading, plotting, etc.)
    ├── mlp_experiments.ipynb         # Jupyter Notebook for MLP experiments (also .py)
    ├── pipeline.py                   # Core training and evaluation pipeline class
    ├── vit_experiments.ipynb         # Jupyter Notebook for ViT experiments (also .py)
    ├── requirements.txt              # Project dependencies
    ├── README.md                     # This file
    ├── LICENSE                       # Project License file
    ├── custom_models/                # Directory for custom model definitions
    │   ├── __init__.py
    │   ├── cnn.py                    # Custom CNN architecture
    │   └── mlp.py                    # Custom MLP architecture
    └── figures/                      # Directory for storing plots and figures
        ├── dermamnist_samples.png
        ├── cnn_avg_loss_curves.png
        ├── final_roc_curves.png
        └── cnn_avg_confusion_matrix.png
        └── <!-- Add other relevant figures -->
```

## Full Report

For a detailed description of the methodology, experiments, and in-depth analysis, please refer to the full project report:
[Link to Full PDF Report](https://github.com/josemnmatos/ml-dermatology-project/blob/main/report.pdf)
