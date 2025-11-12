# 🎧 AudioMNIST Classification — Warm-up 2  

**Author:** Aqsa Mohsin  
**Course:** Project Data Science & Artificial Intelligence WiSe 25/26  

---

## 📘 Overview

This project is part of the **Warm-up 2 assignment** for the Data Science & AI course.  
The objective is to work with the [AudioMNIST dataset](https://github.com/soerenab/AudioMNIST),  
which contains spoken digits (0 – 9) from 60 different speakers.

Two related classification tasks are performed:

1. **Digit Classification** — predict *what* digit was spoken.  
2. **Speaker Classification** — predict *who* spoke the digit.

Each audio file is converted into a compact set of **Mel-Frequency Cepstral Coefficients (MFCC)** features  
to represent the frequency characteristics of speech.  
Both **classical machine-learning** and **deep-learning** approaches are implemented.

---

## 🧠 Tasks & Pipeline

| Step | Description |
|------|--------------|
| 1️⃣ Data Acquisition | Download AudioMNIST dataset (30 000 audio samples). |
| 2️⃣ Feature Extraction | Compute 13 MFCC coefficients per sample using Librosa. |
| 3️⃣ Train/Test Split | 80 % train / 20 % test, stratified by class. |
| 4️⃣ Baseline Model | Support Vector Machine (SVM) with RBF kernel. |
| 5️⃣ Neural Network | Simple feed-forward network (PyTorch) with early stopping. |
| 6️⃣ Evaluation | Accuracy, F1-score, confusion matrix, per-class accuracy plots. |

---

## 📊 Results Summary

| Task | Model | Accuracy | Notes |
|------|--------|-----------|-------|
| Digit Classification | **SVM** | 0.76 | Reasonable baseline but confuses similar digits. |
| Digit Classification | **Neural Network** | **0.921** | Learns non-linear patterns in MFCC features. |
| Speaker Classification | **SVM** | 0.337 | Hard task (60 classes), underfits. |
| Speaker Classification | **Neural Network** | **0.86** | Captures speaker-specific voice signatures. |



---

## ⚙️ Technologies

| Category | Tools |
|-----------|--------|
| Programming | Python 3.10 + Google Colab |
| Libraries | `torch`, `librosa`, `numpy`, `scikit-learn`, `tqdm`, `matplotlib`, `seaborn` |
| ML Approaches | SVM (RBF kernel), Feed-Forward Neural Network (PyTorch) |

---

## 🧩 Key Learnings

- Understanding audio preprocessing and MFCC feature engineering.  
- Comparing classical and neural approaches on the same data.  
- Implementing **early stopping** to avoid overfitting.  
- Visual interpretation of model performance using confusion matrices and bar plots.  

