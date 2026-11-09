"""
Audio processing utilities for FastSpeech 2:
1. HiFi-GAN compatible Log-Mel Spectrogram computation (power=1.0, magnitude-based).
2. Continuous Fundamental Frequency (F0 / Pitch) extraction with linear interpolation.
3. Frame-level STFT energy extraction.
4. Griffin-Lim inversion (matched power=1.0) for CPU fallback.
5. Neural Vocoder (HiFi-GAN) loader and synthesis pipeline.
"""

import os
import numpy as np
import torch
import librosa

try:
    import pyworld as pw
except ImportError:
    pw = None

def get_mel_spectrogram(audio_np, sr=22050, n_fft=1024, hop_length=256, win_length=1024,
                        n_mels=80, fmin=0.0, fmax=8000.0):
    """
    Computes log-mel spectrogram from audio array strictly matching the HiFi-GAN
    training specifications (power=1.0, linear magnitude STFT).
    Returns:
        log_mel: [T_mel, n_mels] float32 numpy array.
    """
    # 1. Linear STFT with Hann window and reflect padding
    stft = librosa.stft(
        audio_np,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        center=True,
        pad_mode='reflect'
    )
    magnitudes = np.abs(stft)  # Power = 1.0 (magnitude)

    # 2. Mel Filterbank projection
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    mel = np.dot(mel_basis, magnitudes)

    # 3. Dynamic range compression (log-scale with 1e-5 clamp)
    log_mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))

    # Transpose to [T_mel, 80] for FastSpeech 2 temporal sequence processing
    return log_mel.T.astype(np.float32)

def extract_pitch(audio_np, sr=22050, hop_length=256):
    """
    Extracts continuous fundamental frequency (F0) contour.
    Uses PyWorld Harvest if installed, falling back to Librosa YIN.
    Unvoiced segments (F0 == 0) are linearly interpolated to preserve smooth gradients.
    Returns:
        f0_interp: [T_mel] float32 numpy array.
    """
    if pw is not None:
        audio_double = audio_np.astype(np.float64)
        frame_period = (hop_length / sr) * 1000.0
        f0, _ = pw.harvest(audio_double, sr, frame_period=frame_period)
        f0 = f0.astype(np.float32)
    else:
        # Fallback to librosa YIN with human speech vocal range (65Hz - 500Hz)
        f0 = librosa.yin(audio_np, fmin=65, fmax=500, sr=sr, hop_length=hop_length)
        f0 = np.nan_to_num(f0).astype(np.float32)

    # Continuous linear interpolation across unvoiced frames
    voiced_idx = np.where(f0 > 0)[0]
    if len(voiced_idx) > 0:
        f0_interp = np.interp(np.arange(len(f0)), voiced_idx, f0[voiced_idx])
    else:
        f0_interp = f0

    return f0_interp.astype(np.float32)

def extract_energy(audio_np, n_fft=1024, hop_length=256, win_length=1024):
    """
    Extracts frame-level energy using L2 norm of STFT magnitudes.
    Returns:
        energy: [T_mel] float32 numpy array.
    """
    stft = librosa.stft(
        audio_np,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        center=True
    )
    mag = np.abs(stft)
    energy = np.linalg.norm(mag, axis=0)
    return energy.astype(np.float32)

def griffin_lim(log_mel, sr=22050, n_fft=1024, hop_length=256, win_length=1024, n_iter=60):
    """
    Reconstructs audio waveform using Griffin-Lim.
    Properly matches power=1.0 linear magnitude inversion.
    """
    mel = np.exp(log_mel.T)  # [n_mels, T]
    linear_spec = librosa.feature.inverse.mel_to_stft(
        mel,
        sr=sr,
        n_fft=n_fft,
        power=1.0  # Matched with magnitude extraction
    )
    wav = librosa.griffinlim(
        linear_spec,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length
    )
    return wav.astype(np.float32)

def load_hifigan_vocoder(device="cpu"):
    """
    Loads pretrained NVIDIA HiFi-GAN neural vocoder.
    Attempts TorchHub first, with clean error handling and CPU/GPU fallback.
    """
    try:
        vocoder = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_hifigan')
        vocoder = vocoder.to(device).eval()
        try:
            vocoder.remove_weight_norm()
        except Exception:
            pass
        return vocoder
    except Exception as e:
        print(f"[Warning] Could not load HiFi-GAN from TorchHub: {e}")
        return None

def vocode_mel(mel, vocoder=None, sr=22050, hop_length=256, device="cpu"):
    """
    Synthesizes audio waveform from a mel spectrogram.
    Uses neural HiFi-GAN if provided/available; gracefully falls back to Griffin-Lim.
    Args:
        mel: torch.Tensor or np.ndarray of shape [T, 80] or [1, T, 80] or [1, 80, T].
    Returns:
        wav: float32 numpy array, peak normalized to 0.95.
    """
    if isinstance(mel, np.ndarray):
        mel_t = torch.from_numpy(mel).float()
    else:
        mel_t = mel.float()

    # Ensure shape is [1, 80, T] for vocoder
    if mel_t.dim() == 2:
        # [T, 80] -> [1, 80, T]
        mel_t = mel_t.transpose(0, 1).unsqueeze(0)
    elif mel_t.dim() == 3:
        if mel_t.shape[1] != 80 and mel_t.shape[2] == 80:
            mel_t = mel_t.transpose(1, 2)

    mel_t = mel_t.to(device)

    if vocoder is not None:
        with torch.inference_mode():
            # Clamp mel values to realistic acoustic bounds to prevent saturation
            mel_clamped = torch.clamp(mel_t, min=-11.51, max=2.5)
            wav = vocoder(mel_clamped).squeeze().cpu().numpy()
    else:
        # Fallback to Griffin-Lim
        mel_np = mel_t.squeeze(0).transpose(0, 1).cpu().numpy()  # [T, 80]
        wav = griffin_lim(mel_np, sr=sr, hop_length=hop_length)

    # Peak normalization to prevent clipping distortion
    max_val = np.max(np.abs(wav))
    if max_val > 1e-4:
        wav = (wav / max_val) * 0.95

    return wav.astype(np.float32)
