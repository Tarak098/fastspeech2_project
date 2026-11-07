"""
FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (Telugu).
Combines Feed-Forward Transformer Encoder, Variance Adaptor, Decoder, and Post-Net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import PositionalEncoding, FFTBlock
from .variance_adaptor import VarianceAdaptor

class PostNet(nn.Module):
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

    def forward(self, mel):
        x = mel.transpose(1, 2)
        for conv in self.convolutions:
            x = conv(x)
        return x.transpose(1, 2)

class FastSpeech2(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=2, d_inner=1024,
                 enc_layers=4, dec_layers=4, n_mels=80, dropout=0.1):
        super(FastSpeech2, self).__init__()
        self.d_model = d_model
        
        self.src_word_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            FFTBlock(d_model, n_heads, d_inner, dropout=dropout) for _ in range(enc_layers)
        ])
        
        self.variance_adaptor = VarianceAdaptor(d_model=d_model, dropout=dropout)
        
        self.decoder_layers = nn.ModuleList([
            FFTBlock(d_model, n_heads, d_inner, dropout=dropout) for _ in range(dec_layers)
        ])
        self.mel_linear = nn.Linear(d_model, n_mels)
        self.postnet = PostNet(n_mels=n_mels, dropout=dropout)

    def forward(self, src_seq, duration_target=None, pitch_target=None, energy_target=None,
                max_duration_len=None, p_control=1.0, e_control=1.0, d_control=1.0):
        src_mask = (src_seq != 0).unsqueeze(1).unsqueeze(2)
        enc_out = self.src_word_emb(src_seq)
        enc_out = self.pos_enc(enc_out)
        for enc_layer in self.encoder_layers:
            enc_out = enc_layer(enc_out, mask=src_mask)
            
        adapted, d_pred, p_pred, e_pred, durations = self.variance_adaptor(
            enc_out,
            duration_target=duration_target,
            pitch_target=pitch_target,
            energy_target=energy_target,
            max_duration_len=max_duration_len,
            p_control=p_control,
            e_control=e_control,
            d_control=d_control
        )
        
        dec_out = self.pos_enc(adapted)
        for dec_layer in self.decoder_layers:
            dec_out = dec_layer(dec_out)
            
        mel_before = self.mel_linear(dec_out)
        mel_residual = self.postnet(mel_before)
        mel_after = mel_before + mel_residual
        
        return mel_before, mel_after, d_pred, p_pred, e_pred, durations

class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super(FastSpeech2Loss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, mel_before, mel_after, d_pred, p_pred, e_pred,
                mel_target, d_target, p_target, e_target):
        min_mel_len = min(mel_before.size(1), mel_target.size(1))
        mel_before = mel_before[:, :min_mel_len, :]
        mel_after = mel_after[:, :min_mel_len, :]
        mel_target = mel_target[:, :min_mel_len, :]

        min_f_len = min(p_pred.size(1), p_target.size(1))
        p_pred = p_pred[:, :min_f_len]
        p_target = p_target[:, :min_f_len]
        e_pred = e_pred[:, :min_f_len]
        e_target = e_target[:, :min_f_len]

        mel_loss_before = self.l1_loss(mel_before, mel_target)
        mel_loss_after = self.l1_loss(mel_after, mel_target)
        
        d_target_log = torch.log(d_target.float() + 1.0)
        d_loss = self.mse_loss(d_pred, d_target_log)
        
        p_loss = self.mse_loss(p_pred, p_target)
        e_loss = self.mse_loss(e_pred, e_target)
        
        total_loss = mel_loss_before + mel_loss_after + d_loss + p_loss + e_loss
        return total_loss, mel_loss_before, mel_loss_after, d_loss, p_loss, e_loss

