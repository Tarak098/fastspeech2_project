"""
FastSpeech 2 End-to-End Non-Autoregressive TTS Architecture:
Phoneme Encoder, Variance Adaptor, Mel Decoder, Residual Post-Net,
and Fully-Masked Multi-Task Objective Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import FFTBlock, PositionalEncoding
from .variance_adaptor import VarianceAdaptor

class PostNet(nn.Module):
    """5-layer 1D convolutional Post-Net predicting residual mel refinements."""
    def __init__(self, n_mels=80, postnet_dim=256, kernel_size=5, dropout=0.1):
        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(n_mels, postnet_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
                nn.BatchNorm1d(postnet_dim),
                nn.Tanh(),
                nn.Dropout(dropout)
            )
        )

        for _ in range(3):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(postnet_dim, postnet_dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
                    nn.BatchNorm1d(postnet_dim),
                    nn.Tanh(),
                    nn.Dropout(dropout)
                )
            )

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(postnet_dim, n_mels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
                nn.Dropout(dropout)
            )
        )

    def forward(self, x):
        # x: [B, T_mel, n_mels] -> [B, n_mels, T_mel]
        out = x.transpose(1, 2)
        for i, conv in enumerate(self.convolutions):
            out = conv(out)
        return out.transpose(1, 2)  # [B, T_mel, n_mels]

class FastSpeech2(nn.Module):
    def __init__(self, vocab_size=120, d_model=256, n_heads=2, d_inner=1024,
                 enc_layers=4, dec_layers=4, n_mels=80, dropout=0.1):
        super(FastSpeech2, self).__init__()
        self.d_model = d_model
        self.n_mels = n_mels

        # 1. Phoneme Encoder
        self.src_word_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            FFTBlock(d_model, n_heads, d_inner, dropout=dropout) for _ in range(enc_layers)
        ])

        # 2. Variance Adaptor
        self.variance_adaptor = VarianceAdaptor(d_model=d_model, dropout=dropout)

        # 3. Mel Decoder
        self.decoder_layers = nn.ModuleList([
            FFTBlock(d_model, n_heads, d_inner, dropout=dropout) for _ in range(dec_layers)
        ])
        self.mel_linear = nn.Linear(d_model, n_mels)
        self.postnet = PostNet(n_mels=n_mels, dropout=dropout)

    def forward(self, src_seq, duration_target=None, pitch_target=None, energy_target=None,
                max_duration_len=None, p_control=1.0, e_control=1.0, d_control=1.0):
        # 1. Phoneme Encoding with padding mask
        src_mask = (src_seq != 0)  # [B, T_text]
        enc_mask = src_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T_text]

        enc_out = self.src_word_emb(src_seq)
        enc_out = self.pos_enc(enc_out)
        for enc_layer in self.encoder_layers:
            enc_out = enc_layer(enc_out, mask=enc_mask)

        # 2. Variance Adaptation
        mel_mask = None
        if duration_target is not None:
            mel_lens = duration_target.sum(dim=1)
            max_mel_len = int(mel_lens.max().item()) if max_duration_len is None else max_duration_len
            mel_mask = torch.arange(max_mel_len, device=src_seq.device).unsqueeze(0) < mel_lens.unsqueeze(1)

        adapted, d_pred, p_pred, e_pred, durations, mel_mask = self.variance_adaptor(
            enc_out,
            src_mask=src_mask,
            mel_mask=mel_mask,
            duration_target=duration_target,
            pitch_target=pitch_target,
            energy_target=energy_target,
            max_duration_len=max_duration_len,
            p_control=p_control,
            e_control=e_control,
            d_control=d_control
        )

        # 3. Mel Decoding with decoder attention mask
        dec_mask = mel_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T_mel]
        dec_out = self.pos_enc(adapted)
        for dec_layer in self.decoder_layers:
            dec_out = dec_layer(dec_out, mask=dec_mask)

        mel_before = self.mel_linear(dec_out)
        mel_residual = self.postnet(mel_before)
        mel_after = mel_before + mel_residual

        # Mask outputs
        mel_before = mel_before.masked_fill(~mel_mask.unsqueeze(-1), 0.0)
        mel_after = mel_after.masked_fill(~mel_mask.unsqueeze(-1), 0.0)

        return mel_before, mel_after, d_pred, p_pred, e_pred, durations

class FastSpeech2Loss(nn.Module):
    """
    Fully-masked multi-task loss for FastSpeech 2:
    - Mel Linear & Post-Net Loss: L1 on valid mel frames
    - Duration Loss: MSE in log(1 + d) space on valid phonemes
    - Pitch Loss: MSE in standardized space on valid frames
    - Energy Loss: MSE in standardized space on valid frames
    """
    def __init__(self):
        super(FastSpeech2Loss, self).__init__()

    def forward(self, mel_before, mel_after, d_pred, p_pred, e_pred,
                mel_target, d_target, p_target, e_target,
                src_mask, mel_mask):
        # 1. Truncate/Align temporal frames between prediction and target
        min_mel_len = min(mel_before.size(1), mel_target.size(1))
        mel_before = mel_before[:, :min_mel_len, :]
        mel_after = mel_after[:, :min_mel_len, :]
        mel_target = mel_target[:, :min_mel_len, :]
        mel_mask = mel_mask[:, :min_mel_len]

        min_f_len = min(p_pred.size(1), p_target.size(1), min_mel_len)
        p_pred = p_pred[:, :min_f_len]
        p_target = p_target[:, :min_f_len]
        e_pred = e_pred[:, :min_f_len]
        e_target = e_target[:, :min_f_len]
        f_mask = mel_mask[:, :min_f_len]

        # 2. Masked Mel Spectrogram L1 Loss
        m_mask_3d = mel_mask.unsqueeze(-1).float()
        valid_mel_elements = m_mask_3d.sum() * mel_target.size(-1)
        valid_mel_elements = torch.clamp(valid_mel_elements, min=1.0)

        mel_loss_before = (torch.abs(mel_before - mel_target) * m_mask_3d).sum() / valid_mel_elements
        mel_loss_after = (torch.abs(mel_after - mel_target) * m_mask_3d).sum() / valid_mel_elements

        # 3. Masked Duration MSE Loss in log1p space
        min_text_len = min(d_pred.size(1), d_target.size(1))
        d_pred = d_pred[:, :min_text_len]
        d_target = d_target[:, :min_text_len]
        src_mask = src_mask[:, :min_text_len]

        d_target_log = torch.log(d_target.float() + 1.0)
        valid_text_tokens = src_mask.sum().float().clamp(min=1.0)
        d_loss = (((d_pred - d_target_log) ** 2) * src_mask.float()).sum() / valid_text_tokens

        # 4. Masked Pitch & Energy MSE Losses
        valid_frames = f_mask.sum().float().clamp(min=1.0)
        p_loss = (((p_pred - p_target) ** 2) * f_mask.float()).sum() / valid_frames
        e_loss = (((e_pred - e_target) ** 2) * f_mask.float()).sum() / valid_frames

        # Total balanced loss
        total_loss = mel_loss_before + mel_loss_after + d_loss + p_loss + e_loss

        return total_loss, mel_loss_before, mel_loss_after, d_loss, p_loss, e_loss
