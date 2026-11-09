"""
Variance Adaptor for FastSpeech 2 (Telugu):
Implements Duration Predictor, Length Regulator, Pitch Predictor, and Energy Predictor
with proper sequence masking, standardized feature projections, and inference controls.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VariancePredictor(nn.Module):
    """Variance Predictor: 2 Conv1d layers with LayerNorm, ReLU, Dropout and Linear projection."""
    def __init__(self, d_model=256, conv_filter_size=256, kernel_size=3, dropout=0.1):
        super(VariancePredictor, self).__init__()
        self.conv1 = nn.Conv1d(d_model, conv_filter_size, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.norm1 = nn.LayerNorm(conv_filter_size)
        self.conv2 = nn.Conv1d(conv_filter_size, conv_filter_size, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.norm2 = nn.LayerNorm(conv_filter_size)
        self.linear = nn.Linear(conv_filter_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        out = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        out = self.norm1(F.relu(out))
        out = self.dropout(out)

        out = self.conv2(out.transpose(1, 2)).transpose(1, 2)
        out = self.norm2(F.relu(out))
        out = self.dropout(out)

        out = self.linear(out).squeeze(-1)
        if mask is not None:
            out = out.masked_fill(~mask, 0.0)
        return out

class LengthRegulator(nn.Module):
    """Expands encoder hidden states according to phoneme durations."""
    def __init__(self):
        super(LengthRegulator, self).__init__()

    def forward(self, x, duration, max_len=None):
        output = []
        for batch_idx in range(x.size(0)):
            h = x[batch_idx]
            d = duration[batch_idx].clamp(min=0)
            expanded = torch.repeat_interleave(h, d.long(), dim=0)
            output.append(expanded)

        if max_len is None:
            max_len = max(max([o.size(0) for o in output]), 1)

        padded_output = []
        for o in output:
            if o.size(0) < max_len:
                pad = torch.zeros(max_len - o.size(0), o.size(1), device=o.device)
                padded_output.append(torch.cat([o, pad], dim=0))
            else:
                padded_output.append(o[:max_len])

        return torch.stack(padded_output, dim=0)

class VarianceAdaptor(nn.Module):
    """FastSpeech 2 Variance Adaptor combining Duration, Pitch, and Energy."""
    def __init__(self, d_model=256, conv_filter_size=256, dropout=0.1):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(d_model, conv_filter_size, dropout=dropout)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(d_model, conv_filter_size, dropout=dropout)
        self.energy_predictor = VariancePredictor(d_model, conv_filter_size, dropout=dropout)

        self.pitch_projection = nn.Linear(1, d_model)
        self.energy_projection = nn.Linear(1, d_model)

    def forward(self, x, src_mask=None, mel_mask=None,
                duration_target=None, pitch_target=None, energy_target=None,
                max_duration_len=None, p_control=1.0, e_control=1.0, d_control=1.0):
        duration_pred = self.duration_predictor(x, mask=src_mask)

        if duration_target is not None:
            durations = duration_target
        else:
            d_exp = torch.exp(duration_pred) - 1.0
            durations = torch.clamp(torch.round(d_exp * d_control), min=1.0, max=50.0).long()
            if src_mask is not None:
                durations = durations.masked_fill(~src_mask, 0)

        x_expanded = self.length_regulator(x, durations, max_len=max_duration_len)

        if mel_mask is None:
            mel_lens = durations.sum(dim=1)
            max_mel_len = x_expanded.size(1)
            mel_mask = torch.arange(max_mel_len, device=x.device).unsqueeze(0) < mel_lens.unsqueeze(1)

        pitch_pred = self.pitch_predictor(x_expanded, mask=mel_mask)
        if pitch_target is not None:
            pitch_feature = pitch_target
        else:
            pitch_feature = pitch_pred * p_control

        pitch_embed = self.pitch_projection(pitch_feature.unsqueeze(-1))

        energy_pred = self.energy_predictor(x_expanded, mask=mel_mask)
        if energy_target is not None:
            energy_feature = energy_target
        else:
            energy_feature = energy_pred * e_control

        energy_embed = self.energy_projection(energy_feature.unsqueeze(-1))

        adapted = x_expanded + pitch_embed + energy_embed

        if mel_mask is not None:
            adapted = adapted.masked_fill(~mel_mask.unsqueeze(-1), 0.0)

        return adapted, duration_pred, pitch_pred, energy_pred, durations, mel_mask
