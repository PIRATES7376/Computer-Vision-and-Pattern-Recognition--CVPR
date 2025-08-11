# k-NN Image Classifier (Dogs/Cats/Pandas) — Colab
This repository contains a from-scratch implementation of a k-Nearest Neighbors (k-NN) image classifier in NumPy, developed for a CVPR lab assignment. It loads exactly 100 images per class (dog, cat, panda) from Google Drive, preprocesses them to 32×32 grayscale float32 vectors, and evaluates k=1...20 using stratified 5-fold cross-validation, reporting mean±std accuracy and an accuracy-vs-k plot. The project deliberately avoids scikit-learn’s KNeighborsClassifier (using scikit-learn only for StratifiedKFold) to make the distance computation, neighbor selection, and majority-vote mechanics explicit.

Goal: From-scratch k-Nearest Neighbors (NumPy) baseline on a 3-class dataset (dog, cat, panda) with accuracy-vs-k analysis.

# 🎯 Objectives
*Loads 100 images per class from Google Drive (folders containing dog, cat, panda).

*Preprocesses to 32×32 grayscale, float32, [0,1], then flattens.

*Runs Stratified 5-Fold Cross-Validation for k = 2…20.

*Plots accuracy vs k and prints the best k.

# How to run (Colab)
Open the notebook in Colab.

drive.mount('/content/drive')

Set your dataset path:
dataset_dir = "/content/drive/MyDrive/...."

# Outputs
Plot: Accuracy vs k (mean ± std)

Console: Best k
