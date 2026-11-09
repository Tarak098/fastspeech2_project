# FastSpeech 2-Based Text-to-Speech Synthesis (English & Telugu)

This repository contains an end-to-end implementation, training pipeline, and neural acoustic synthesis evaluation of the **FastSpeech 2** non-autoregressive Text-to-Speech (TTS) architecture trained independently on:
1. **English**: Subsampled from the **Svarah** speech dataset (~5 hours train, 1 hour validation).
2. **Telugu**: Subsampled from the **IndicTTS Telugu** dataset (~5 hours train, 1 hour validation).

---

## 🚀 Key Engineering Upgrades & Defect Resolutions

This codebase has undergone a complete Senior Speech AI Engineer audit and repair:
- **Genuine Phoneme Forced Alignment**: Replaced artificial uniform duration assignment with the **Meta MMS Multilingual Forced Aligner (`torchaudio.pipelines.MMS_FA`)** and Dravidian phonetic segmentation.
- **Telugu Transcript Extraction Fix**: Corrected IndicTTS metadata parsing to properly extract Telugu sentences (`parts[2]`) rather than speaker IDs (`parts[1]`).
- **HiFi-GAN Compatible Mel Spectrogram**: Standardized mel extraction to linear magnitude STFT ($\text{power}=1.0$, $80$ mels, $f_{\min}=0$, $f_{\max}=8000\text{ Hz}$) strictly matching NVIDIA HiFi-GAN expectations.
- **Variance Feature Standardization**: Standardized continuous fundamental frequency ($F_0$) and STFT energy to zero-mean and unit-variance ($\mu=0, \sigma=1$), preventing scale domination over spectral loss.
- **Masked Multi-Task Losses**: Implemented strict temporal padding masking on mel, duration, pitch, and energy losses, alongside decoder attention masking.
- **Dynamic Positional Encoding**: Dynamic sinusoidal generation supporting arbitrary sequence lengths without static buffer overflow.
- **Arbitrary Text CLI Synthesis**: Full command-line interface supporting arbitrary user text, duration scaling, pitch shifting, and neural HiFi-GAN waveform inversion.

---

## 📁 Repository Structure

```
project/
├── English/
│   ├── aligner.py          # MMS Forced Aligner & phonetic segmenter
│   ├── train.py            # Fully-masked FastSpeech 2 training pipeline
│   ├── preprocess.py       # Audio resampling, feature extraction, stats.json
│   ├── inference.py        # CLI synthesis & benchmark evaluation
│   ├── config.yaml         # Acoustic and architectural configuration
│   ├── dataset.py          # PyTorch Dataset & collate_fn with boolean masks
│   ├── checkpoints/        # Model checkpoints (best_model.pth, epoch_*.pth)
│   ├── model/              # FastSpeech 2, Variance Adaptor, PostNet, FFT blocks
│   ├── text/               # ARPAbet G2P, normalization, symbols
│   └── audio/              # HiFi-GAN vocoder, mel, pitch, energy tools
├── Telugu/
│   ├── aligner.py          # Telugu MMS Aligner & Akshara segmenter
│   ├── train.py            # Telugu training pipeline
│   ├── preprocess.py       # IndicTTS preprocessing & transcript parser
│   ├── inference.py        # Telugu CLI synthesis & benchmark evaluation
│   ├── config.yaml         # Telugu model configuration
│   ├── dataset.py          # Telugu dataset loader
│   ├── checkpoints/        # Telugu checkpoints
│   ├── model/              # Model modules
│   ├── text/               # Unicode Akshara decomposition, symbols
│   └── audio/              # Audio & vocoder tools
├── Results/
│   ├── English/            # Loss curves, WAV audios, Variance Adaptor plots
│   └── Telugu/             # Loss curves, WAV audios, Variance Adaptor plots
├── Report/
│   ├── Technical_Report.pdf # Final academic PDF report
│   └── Technical_Report.tex # Complete clean LaTeX source code
├── test_pipeline.py        # Automated 5-stage diagnostic test suite
├── FastSpeech2_English_Kaggle.ipynb # End-to-end Kaggle/Colab notebook
├── FastSpeech2_Telugu_Kaggle.ipynb  # End-to-end Kaggle/Colab notebook
├── README.md
└── requirements.txt
```

---

## 🛠️ Installation & Environment Setup

```bash
git clone https://github.com/Tarak098/fastspeech2_project.git
cd fastspeech2_project

# Install required dependencies
pip install -r requirements.txt

# Download required NLTK resources (for English G2P)
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## 🎙️ Command-Line Inference (Arbitrary Text)

### 1. English Synthesis
```bash
cd English
python inference.py --text "Hello! FastSpeech 2 neural speech synthesis is now working cleanly." --output output_en.wav
```

### 2. Telugu Synthesis
```bash
cd Telugu
python inference.py --text "నమస్కారం! తెలుగు ఫాస్ట్‌స్పీచ్ 2 సంశ్లేషణ విజయవంతంగా పనిచేస్తోంది." --output output_te.wav
```

### 3. Prosody Controls
You can control speaking speed, pitch, and energy directly from the command line:
```bash
# Slow down speech (1.2x duration) and raise pitch by 10%
python inference.py --text "The weather is beautiful." --d_control 1.2 --p_control 1.1 --output expressive.wav
```

---

## 🧪 Automated Diagnostic Verification Suite

Run the comprehensive 5-stage validation suite to verify text normalization, phoneme mappings, duration variance, mel numerical integrity, and vocoder output:

```bash
python test_pipeline.py
```

---

## 📊 Preprocessing & Training from Scratch

### English Pipeline:
```bash
cd English
# 1. Preprocess dataset & extract genuine alignments
python preprocess.py config.yaml

# 2. Train FastSpeech 2
python train.py config.yaml

# 3. Evaluate on the 5 benchmark test sentences
python inference.py --config config.yaml --checkpoint checkpoints/best_model.pth
```

### Telugu Pipeline:
```bash
cd Telugu
# 1. Preprocess IndicTTS Telugu dataset
python preprocess.py config.yaml

# 2. Train FastSpeech 2
python train.py config.yaml

# 3. Evaluate on the 5 benchmark test sentences
python inference.py --config config.yaml --checkpoint checkpoints/best_model.pth
```

---

## 📜 Academic Technical Report
The full technical report covering mathematical formulations, Dravidian orthographic adaptations, loss convergence analysis, and linguistic evaluations is available in `Report/Technical_Report.pdf` (and `Report/Technical_Report.tex`).
