"""
Audio processing utilities: Mel-spectrogram computation, Pitch (F0) extraction,
Energy extraction, and audio reconstruction for FastSpeech 2.
"""

import numpy as np
import torch
import torch.nn.functional as F
import librosa
try:
    import pyworld as pw
except ImportError:
    pw = None

def get_mel_spectrogram(audio_np, sr=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmin=0.0, fmax=8000.0):
    """Computes log-mel spectrogram from numpy audio array."""
    mel = librosa.feature.melspectrogram(
        y=audio_np,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    log_mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    return log_mel.T  # Shape: [T_mel, 80]

def extract_pitch(audio_np, sr=22050, hop_length=256):
    """
    Extracts fundamental frequency (F0) contour.
    Uses PyWorld harvest if available, otherwise falls back to Librosa YIN.
    """
    if pw is not None:
        audio_double = audio_np.astype(np.float64)
        frame_period = (hop_length / sr) * 1000.0
        f0, timeaxis = pw.harvest(audio_double, sr, frame_period=frame_period)
        f0 = f0.astype(np.float32)
    else:
        # Fallback to librosa YIN
        f0 = librosa.yin(audio_np, fmin=65, fmax=500, sr=sr, hop_length=hop_length)
        f0 = np.nan_to_num(f0).astype(np.float32)

    # Linearly interpolate unvoiced (0-pitch) frames
    nonzero_idx = np.where(f0 > 0)[0]
    if len(nonzero_idx) > 0:
        f0_interp = np.interp(np.arange(len(f0)), nonzero_idx, f0[nonzero_idx])
    else:
        f0_interp = f0
    return f0_interp.astype(np.float32)

def extract_energy(audio_np, n_fft=1024, hop_length=256, win_length=1024):
    """Extracts frame-level energy using L2 norm of STFT magnitudes."""
    stft = librosa.stft(audio_np, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(stft)
    energy = np.linalg.norm(mag, axis=0)
    return energy.astype(np.float32)

def griffin_lim(mel_spec_db, sr=22050, n_fft=1024, hop_length=256, win_length=1024, n_iter=60):
    """Reconstructs audio using Griffin-Lim vocoder algorithm."""
    mel_spec = np.exp(mel_spec_db.T)
    linear_spec = librosa.feature.inverse.mel_to_stft(
        mel_spec,
        sr=sr,
        n_fft=n_fft,
        power=1.0
    )
    wav = librosa.griffinlim(linear_spec, n_iter=n_iter, hop_length=hop_length, win_length=win_length)
    return wav

