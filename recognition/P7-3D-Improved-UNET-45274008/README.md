# 3D Prostate Segmentation using UNet Architectures

**Problem: Task 7 - Segmenting Prostate 3D data set.**

**Author: Alan Ravikumar - 45274008**

## Table of Contents

### 1. Overview: Problem and Algorithm

The Problem

This project tackles the challenge of volumetric medical image segmentation. The goal is to take a 3D MRI scan of a prostate and automatically generate an accurate segmentation mask that identifies different classes of tissue (6 classes in total, including background and 5 distinct foreground regions). Automated segmentation is critical for streamlining clinical workflows, aiding in diagnosis, and planning treatments.

### 2. Dependences

This project was constructed with the following key dependencies:

* **Python 3.x**: Core programming language.
* **PyTorch**: Deep learning framework for training and evaluation of models.
* **NumPy**: Numerical computation and array operations.
* **nibabel**: For loading and handling NIfTI (`.nii.gz`) medical image files.
* **matplotlib**: For plotting training history and visualizing 2D/3D image slices.
* **scikit-learn**: For data splitting (`train_test_split`) and performance metrics.
* **tqdm**: To display progress bars during training and evaluation.

### 3. Project Structure

The project consists of the following files:

* `modules.py`: Contains the PyTorch class definitions for both the baseline `UNet3D` and the `ImprovedUNet3D`, along with their respective building blocks (`DoubleConv`, `ImprovedConvBlock`, etc.).
* `dataset.py`: Contains the `ProstateDataset` class, a custom PyTorch `Dataset` that loads, pre-processes, and serves 3D NIfTI files for the `DataLoader`.
* `train.py`: The main training script. It handles data splitting, model initialization (Standard or Improved), the training/validation loop, and saving the best-performing model based on validation Dice score.
* `predict.py`: The evaluation and visualization script. It loads a trained model, runs it on the entire validation set to calculate the final average and per-class Dice scores, and then provides a 3D slice-by-slice visualization of a sample prediction.

### 4. Model and Data

### 5. Implementation

### 6. Results

### 7. Future Work

### 8. References