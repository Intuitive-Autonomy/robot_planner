#!/usr/bin/env python3
"""
Neural network model for motion prediction
- Input: [seq_len, skeleton_dim] skeleton sequences
- Output: [pred_len, output_dim] predicted motion
- Supports teacher forcing for training
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class MotionTransformer(nn.Module):
    """Transformer-based motion prediction model"""
    def __init__(self,
                 skeleton_dim=63,
                 output_dim=12,
                 hidden_dim=256,
                 num_layers=4,
                 num_heads=8,
                 dropout=0.1):
        super().__init__()

        self.skeleton_dim = skeleton_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(skeleton_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Decoder for autoregressive generation
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Projection from output back to hidden for teacher forcing
        self.output_to_hidden = nn.Linear(output_dim, hidden_dim)

        # Learnable start token for decoder
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, pred_len, teacher_forcing_ratio=0.0, y_gt=None):
        """
        Args:
            src: [batch_size, seq_len, skeleton_dim] - input skeleton sequence
            pred_len: int - number of frames to predict
            teacher_forcing_ratio: float - ratio for teacher forcing during training
            y_gt: [batch_size, pred_len, output_dim] - ground truth targets (for teacher forcing)
        Returns:
            output: [batch_size, pred_len, output_dim] - predicted motion
        """
        batch_size = src.size(0)
        device = src.device

        # Encode input sequence
        src_embedded = self.input_proj(src)  # [B, seq_len, hidden_dim]
        src_embedded = self.pos_encoding(src_embedded.transpose(0, 1)).transpose(0, 1)
        src_embedded = self.dropout(src_embedded)

        # Encode with transformer
        memory = self.transformer_encoder(src_embedded)  # [B, seq_len, hidden_dim]

        # Initialize decoder input with start token
        decoder_input = self.start_token.expand(batch_size, 1, -1)  # [B, 1, hidden_dim]
        outputs = []

        # Autoregressive generation
        for t in range(pred_len):
            # Add positional encoding to decoder input
            decoder_input_pos = self.pos_encoding(decoder_input.transpose(0, 1)).transpose(0, 1)
            decoder_input_pos = self.dropout(decoder_input_pos)

            # Decode
            decoder_output = self.transformer_decoder(
                decoder_input_pos, memory
            )  # [B, current_len, hidden_dim]

            # Project to output dimension
            current_output = self.output_proj(decoder_output[:, -1:, :])  # [B, 1, output_dim]
            outputs.append(current_output)

            # Prepare next decoder input
            if self.training and y_gt is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing: use ground truth (project from output_dim to hidden_dim)
                gt_embedded = self._project_output_to_hidden(y_gt[:, t:t+1, :])
                decoder_input = torch.cat([decoder_input, gt_embedded], dim=1)
            else:
                # Use own prediction
                pred_embedded = self._project_output_to_hidden(current_output)
                decoder_input = torch.cat([decoder_input, pred_embedded], dim=1)

        # Concatenate all outputs
        output = torch.cat(outputs, dim=1)  # [B, pred_len, output_dim]
        return output

    def _project_output_to_hidden(self, output):
        """Project output back to hidden dimension for next decoder input"""
        return self.output_to_hidden(output)


def get_model(skeleton_dim=63, output_dim=12, hidden_dim=256, num_layers=4, num_heads=8, dropout=0.1):
    """Factory function to create the model"""
    return MotionTransformer(
        skeleton_dim=skeleton_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    )