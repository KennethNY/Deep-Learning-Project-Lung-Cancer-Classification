# Lung Cancer Image Classification using Deep Learning

## Project Overview

This repository contains a **group project** developed as part of a **university Deep Learning course**. The project focuses on **lung cancer image classification** using convolutional neural networks (CNN) and **transfer learning with MobileNetV2**.

The objective of this project is to apply deep learning techniques to medical image data, understand the full training pipeline, and evaluate model performance in a multi-class classification setting.

**Academic Notice**
This project was developed **collaboratively as a course assignment** and is intended strictly for **educational and academic purposes**. It is **not intended for clinical or real-world medical use**.

---

## Objectives

* Apply deep learning concepts learned in class to a real dataset
* Implement an end-to-end image classification pipeline
* Utilize **transfer learning** for medical image analysis
* Train, validate, and evaluate a CNN model using TensorFlow
* Visualize model performance using accuracy and loss curves

---

## Model Architecture

The model uses **MobileNetV2** pretrained on ImageNet as a feature extractor.

```
Input Image (224 × 224 × 3)
        ↓
MobileNetV2 (pretrained, frozen weights)
        ↓
Global Average Pooling
        ↓
Dropout (0.2)
        ↓
Dense Layer + Softmax (Multi-class Classification)
```

**Why MobileNetV2?**

* Efficient and lightweight CNN architecture
* Strong feature extraction capability
* Suitable for limited-sized medical image datasets

---

## Dataset Structure

The dataset follows a directory-based class structure compatible with TensorFlow utilities:

```
lung_image_sets/
├── lung_aca/   # Lung Adenocarcinoma
├── lung_n/     # Normal Lung Tissue
└── lung_scc/   # Lung Squamous Cell Carcinoma
```

* Each subdirectory represents one class
* Images are automatically labeled based on folder names
* Dataset split:

  * **80% Training**
  * **20% Validation**

---

## Training Configuration

| Parameter         | Value                           |
| ----------------- | ------------------------------- |
| Image Size        | 224 × 224                       |
| Batch Size        | 100                             |
| Epochs            | 5                               |
| Optimizer         | Adam                            |
| Loss Function     | Sparse Categorical Crossentropy |
| Evaluation Metric | Accuracy                        |

---

## Training Pipeline

1. Load and preprocess images using `image_dataset_from_directory`
2. Automatically resize and label images
3. Split dataset into training and validation sets
4. Extract image features using pretrained MobileNetV2
5. Train classification layers on top of extracted features
6. Validate model performance during training
7. Evaluate final model on validation data

---

## Evaluation & Visualization

During training, the following metrics are visualized:

* Training vs Validation Accuracy
* Training vs Validation Loss

These visualizations help in:

* Monitoring training convergence
* Detecting overfitting or underfitting
* Evaluating model generalization performance

---

## Model Output

The trained model is saved using the Keras native format:

```
models/lung_cancer_model.keras
```

This model can be reused for:

* Model evaluation
* Further experimentation
* Fine-tuning exercises

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Transfer Learning (MobileNetV2)

---

## Group Members

* **Akmal Hendrian Malik** – 2702352383
* **Kenneth Nathanael Yuwono** – 2702224062
* **Nicholas Tristan** – 2702286072

---

## Notes & Future Work

Possible future improvements:

* Fine-tuning deeper layers of MobileNetV2
* Confusion matrix and classification report
* Grad-CAM visualization for interpretability
* Hyperparameter optimization

---

## Disclaimer

This project is developed **for educational purposes only** and should **not be used for medical diagnosis or clinical decision-making**.
