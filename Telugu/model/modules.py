"""
Core building blocks for FastSpeech 2: Multi-Head Attention,
Feed-Forward Transformer (FFT) blocks, and Positional Encodings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            pe = torch.zeros(x.size(1), x.size(2), device=x.device)
            position = torch.arange(0, x.size(1), dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, x.size(2), 2, device=x.device).float() * (-math.log(10000.0) / x.size(2)))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return x + pe.unsqueeze(0)
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.q_proj(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.out_proj(context)

class ConvFFN(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size=(9, 1), dropout=0.1):
        super(ConvFFN, self).__init__()
        self.conv1 = nn.Conv1d(d_model, d_inner, kernel_size=kernel_size[0], padding=(kernel_size[0] - 1) // 2)
        self.conv2 = nn.Conv1d(d_inner, d_model, kernel_size=kernel_size[1], padding=(kernel_size[1] - 1) // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x.transpose(1, 2))
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        return out.transpose(1, 2)

class FFTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_inner, kernel_size=(9, 1), dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.ffn = ConvFFN(d_model, d_inner, kernel_size=kernel_size, dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.slf_attn(x, x, x, mask=mask)
        x = self.layer_norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + self.dropout(ffn_out))
        return x

