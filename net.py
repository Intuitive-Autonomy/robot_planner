#!/usr/bin/env python3
"""
Neural network model for motion prediction
- Input: [seq_len, skeleton_dim] skeleton sequences
- Output: [pred_len, output_dim] predicted motion
- Stage 1-3: Parallel prediction (one-shot prediction)
- Stage 4: Autoregressive prediction with predicted robot poses
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
                 skeleton_dim=39,
                 output_dim=12,
                 hidden_dim=256,
                 num_layers=4,
                 num_heads=8,
                 dropout=0.1,
                 max_pred_len=10):
        super().__init__()

        self.skeleton_dim = skeleton_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.max_pred_len = max_pred_len

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

        # Learnable query tokens for parallel prediction (Stage 1-3)
        self.query_tokens = nn.Parameter(torch.randn(1, max_pred_len, hidden_dim))

        # Decoder for attending to encoder output
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection for robot arms
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Phase prediction head
        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)  # 3 classes: 0=approaching, 1=assisting, 2=leaving
        )

        # For Stage 4 autoregressive: project robot output back to hidden
        self.output_to_hidden = nn.Linear(output_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, pred_len, autoregressive=False, y_gt=None, return_phase=False):
        """
        Args:
            src: [batch_size, seq_len, skeleton_dim] - input skeleton sequence
            pred_len: int - number of frames to predict
            autoregressive: bool - whether to use autoregressive mode (Stage 4)
            y_gt: [batch_size, pred_len, output_dim] - ground truth robot poses (for Stage 4)
            return_phase: bool - whether to return phase predictions
        Returns:
            if return_phase:
                (robot_output, phase_logits):
                    robot_output: [batch_size, pred_len, output_dim]
                    phase_logits: [batch_size, pred_len, 3]
            else:
                robot_output: [batch_size, pred_len, output_dim]
        """
        batch_size = src.size(0)
        device = src.device

        # Encode input sequence
        src_embedded = self.input_proj(src)  # [B, seq_len, hidden_dim]
        src_embedded = self.pos_encoding(src_embedded.transpose(0, 1)).transpose(0, 1)
        src_embedded = self.dropout(src_embedded)

        # Encode with transformer
        memory = self.transformer_encoder(src_embedded)  # [B, seq_len, hidden_dim]

        if not autoregressive:
            # Stage 1-3: Parallel prediction (one-shot)
            # Use learnable query tokens
            queries = self.query_tokens[:, :pred_len, :].expand(batch_size, -1, -1)  # [B, pred_len, hidden_dim]
            queries = self.pos_encoding(queries.transpose(0, 1)).transpose(0, 1)
            queries = self.dropout(queries)

            # Decode
            decoder_output = self.transformer_decoder(queries, memory)  # [B, pred_len, hidden_dim]

            # Project to outputs
            robot_output = self.output_proj(decoder_output)  # [B, pred_len, output_dim]
            phase_logits = self.phase_head(decoder_output)   # [B, pred_len, 3]

        else:
            # Stage 4: Sliding window autoregressive prediction
            # Network predicts 10 frames each time, but only uses the 1st prediction
            # Then slides the window forward by 1 frame
            seq_len = src.size(1)  # 30

            robot_outputs = []
            phase_outputs = []

            # Current input window starts as the original src
            current_src = src  # [B, 30, 39]

            for t in range(pred_len):
                # Re-encode current input window
                src_embedded_t = self.input_proj(current_src)
                src_embedded_t = self.pos_encoding(src_embedded_t.transpose(0, 1)).transpose(0, 1)
                src_embedded_t = self.dropout(src_embedded_t)
                memory_t = self.transformer_encoder(src_embedded_t)  # [B, 30, hidden_dim]

                # Predict 10 frames (but only use the first)
                queries = self.query_tokens[:, :pred_len, :].expand(batch_size, -1, -1)
                queries = self.pos_encoding(queries.transpose(0, 1)).transpose(0, 1)
                queries = self.dropout(queries)

                decoder_output = self.transformer_decoder(queries, memory_t)  # [B, 10, hidden_dim]

                # Project all 10 frames
                all_robot_preds = self.output_proj(decoder_output)  # [B, 10, 12]
                all_phase_preds = self.phase_head(decoder_output)    # [B, 10, 3]

                # Only take the first prediction
                first_robot = all_robot_preds[:, 0:1, :]  # [B, 1, 12]
                first_phase = all_phase_preds[:, 0:1, :]  # [B, 1, 3]

                robot_outputs.append(first_robot)
                phase_outputs.append(first_phase)

                # Prepare next window by sliding forward
                if t < pred_len - 1:
                    if self.training and y_gt is not None:
                        # Training: use ground truth
                        next_robot = y_gt[:, t:t+1, :]  # [B, 1, 12]
                    else:
                        # Inference: use prediction
                        next_robot = first_robot

                    # Split current input into joints and arms
                    current_joints = current_src[:, :, :27]  # [B, 30, 27]
                    current_arms = current_src[:, :, 27:]    # [B, 30, 12]

                    # Slide window: shift left by 1 frame
                    # Joints: keep last frame unchanged (human doesn't move forward in time for sliding window)
                    new_joints = current_joints  # [B, 30, 27] - keep same joints window

                    # Arms: remove oldest, add newest prediction
                    new_arms = torch.cat([current_arms[:, 1:, :], next_robot], dim=1)  # [B, 30, 12]

                    # Reassemble input
                    current_src = torch.cat([new_joints, new_arms], dim=-1)  # [B, 30, 39]

            # Concatenate all first predictions
            robot_output = torch.cat(robot_outputs, dim=1)  # [B, pred_len, 12]
            phase_logits = torch.cat(phase_outputs, dim=1)  # [B, pred_len, 3]

        if return_phase:
            return robot_output, phase_logits
        else:
            return robot_output

    def _project_output_to_hidden(self, robot_output):
        """Project robot output to hidden dimension"""
        return self.output_to_hidden(robot_output)


def get_model(skeleton_dim=39, output_dim=12, hidden_dim=256,
              num_layers=4, num_heads=8, dropout=0.1, max_pred_len=10):
    """Create and return the model"""
    return MotionTransformer(
        skeleton_dim=skeleton_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_pred_len=max_pred_len
    )
