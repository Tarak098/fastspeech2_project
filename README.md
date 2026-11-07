# FastSpeech 2-Based Text-to-Speech Synthesis (English & Telugu)

This repository contains the complete implementation, training, evaluation, and analysis of the **FastSpeech 2** non-autoregressive Text-to-Speech (TTS) architecture trained independently on:
1. **English**: Subsampled from the **Svarah** speech dataset (~5 hours train, 1 hour validation).
2. **Telugu**: Subsampled from the **IndicTTS Telugu** dataset (~5 hours train, 1 hour validation).

---

## 📁 Repository Structure

```
project/
├── English/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   ├── dataset.py
│   ├── checkpoints/
│   ├── model/
│   ├── text/
│   └── audio/
├── Telugu/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   ├── dataset.py
│   ├── checkpoints/
│   ├── model/
│   ├── text/
│   └── audio/
├── Results/
│   ├── English/
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/
│   │   └── variance_adaptor_plots/
│   │       ├── duration/
│   │       ├── pitch/
│   │       └── energy/
│   └── Telugu/
│       ├── training_loss.png
│       ├── validation_loss.png
│       ├── test_generated_audios/
│       └── variance_adaptor_plots/
│           ├── duration/
│           ├── pitch/
│           └── energy/
├── Report/
│   └── Technical_Report.pdf
├── FastSpeech2_English_Kaggle.ipynb
├── FastSpeech2_Telugu_Kaggle.ipynb
├── README.md
└── requirements.txt
```

---

## 🛠️ Environment Setup & Installation

Install dependencies locally or inside a virtual environment / Kaggle / Colab:

```bash
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### 1. English Pipeline
```bash
cd English

# 1. Preprocess & extract features (Mel, Pitch, Energy, Duration)
python preprocess.py config.yaml

# 2. Train the model
python train.py config.yaml

# 3. Run inference on test sentences (E1 to E4)
python inference.py config.yaml checkpoints/best_model.pth
```

### 2. Telugu Pipeline
```bash
cd Telugu

# 1. Preprocess & extract features
python preprocess.py config.yaml

# 2. Train the model
python train.py config.yaml

# 3. Run inference on test sentences (T1 to T4)
python inference.py config.yaml checkpoints/best_model.pth
```

---

## 📊 Dataset & Subsampling Strategy

* **Audio Duration**: Exactly ~5 hours of training and ~1 hour of validation were sampled for each language.
* **Separation**: Split partitions are strictly non-overlapping with unique recording IDs.
* **Filtering**: Audio clips shorter than 1.0s or longer than 12.0s are discarded to prevent duration padding degradation.
* **Audio Specs**: 22,050 Hz sampling rate, 16-bit PCM, 80-channel Log-Mel spectrogram (FFT: 1024, Hop: 256, Window: 1024).

---

## 🎯 Test Sentences Evaluated

### English Test Sentences:
* **E1**: "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings."
* **E2 (a)**: "The orchestra's conduct during the ceremony was praised by the audience."
* **E2 (b)**: "The researchers will conduct an experiment to evaluate the proposed method."
* **E3**: "The rhyth m of the orchestra changed unexpectedly."
* **E4**: "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."

### Telugu Test Sentences:
* **T1**: "న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలోనూతన ఔషధాల ప్రభావాన్ని విశ్లషే ించారు."
* **T2 (a)**: "తాడుపై అతని పట్టుచాలా బలంగా ఉంది."
* **T2 (b)**: "ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చ ింది."
* **T3**: "పులి అడ విలో నిశ్శబ్ంద గా నడిచింది."
* **T4**: "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ంధ గా ఉంచండి."

---

## 📦 Model Checkpoints & Git LFS

Due to GitHub file size limits, larger model checkpoints can be tracked via Git LFS:
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git commit -m "Track weights with Git LFS"
```
Direct download links for pre-trained weights are also documented in the Technical Report.

