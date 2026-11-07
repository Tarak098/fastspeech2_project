"""
Variance Adaptor for FastSpeech 2 (Telugu).
Implements Duration Predictor, Length Regulator, Pitch Predictor, and Energy Predictor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VariancePredictor(nn.Module):
    def __init__(self, d_model=256, conv_filter_size=256, kernel_size=3, dropout=0.1):
        super(VariancePredictor, self).__init__()
        self.conv1 = nn.Conv1d(d_model, conv_filter_size, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.norm1 = nn.LayerNorm(conv_filter_size)
        self.conv2 = nn.Conv1d(conv_filter_size, conv_filter_size, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.norm2 = nn.LayerNorm(conv_filter_size)
        self.linear = nn.Linear(conv_filter_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        out = self.norm1(F.relu(out))
        out = self.dropout(out)
        out = self.conv2(out.transpose(1, 2)).transpose(1, 2)
        out = self.norm2(F.relu(out))
        out = self.dropout(out)
        out = self.linear(out)
        return out.squeeze(-1)

class LengthRegulator(nn.Module):
    def __init__(self):
        super(LengthRegulator, self).__init__()

    def forward(self, x, duration, max_len=None):
        output = []
        for batch_idx in range(x.size(0)):
            h = x[batch_idx]
            d = duration[batch_idx]
            expanded = torch.repeat_interleave(h, d.long(), dim=0)
            output.append(expanded)
            
        if max_len is None:
            max_len = max([o.size(0) for o in output])
            
        padded_output = []
        for o in output:
            if o.size(0) < max_len:
                pad = torch.zeros(max_len - o.size(0), o.size(1), device=o.device)
                padded_output.append(torch.cat([o, pad], dim=0))
            else:
                padded_output.append(o[:max_len])
                
        return torch.stack(padded_output, dim=0)

class VarianceAdaptor(nn.Module):
    def __init__(self, d_model=256, conv_filter_size=256, dropout=0.1):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(d_model, conv_filter_size, dropout=dropout)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(d_model, conv_filter_size, dropout=dropout)
        self.energy_predictor = VariancePredictor(d_model, conv_filter_size, dropout=dropout)
        
        self.pitch_projection = nn.Linear(1, d_model)
        self.energy_projection = nn.Linear(1, d_model)

    def forward(self, x, duration_target=None, pitch_target=None, energy_target=None,
                max_duration_len=None, p_control=1.0, e_control=1.0, d_control=1.0):
        duration_pred = self.duration_predictor(x)
        
        if duration_target is not None:
            durations = duration_target
        else:
            durations = torch.clamp(torch.round(torch.exp(duration_pred) - 1) * d_control, min=1.0, max=30.0)
            
        x_expanded = self.length_regulator(x, durations, max_len=max_duration_len)
        
        pitch_pred = self.pitch_predictor(x_expanded)
        if pitch_target is not None:
            pitch_feature = pitch_target.unsqueeze(-1)
        else:
            pitch_feature = (pitch_pred * p_control).unsqueeze(-1)
        pitch_embed = self.pitch_projection(pitch_feature)
        
        energy_pred = self.energy_predictor(x_expanded)
        if energy_target is not None:
            energy_feature = energy_target.unsqueeze(-1)
        else:
            energy_feature = (energy_pred * e_control).unsqueeze(-1)
        energy_embed = self.energy_projection(energy_feature)
        
        adapted = x_expanded + pitch_embed + energy_embed
        return adapted, duration_pred, pitch_pred, energy_pred, durations

