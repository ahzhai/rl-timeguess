#!/usr/bin/env python3
"""
Policy/value network for geo-localization RL with a frozen CNN backbone.

We reuse the baseline CNN backbone from baseline.train_baseline_CNN, drop its head,
and add:
  - policy head over discrete actions
  - value head
  - coordinate regression head (for terminate predictions)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.train_baseline_CNN import CNNGeo


class RLGeoPolicy(nn.Module):
    """
    Actor-critic with attention over the embedding sequence.

    crop -> frozen backbone -> embedding (per step)
    embedding sequence (B, T, D) + mask -> cross-attention -> context (B, D)
    context -> policy logits | value | (lat, lon)
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        checkpoint_path: Optional[str] = None,
        num_actions: int = 7,
        hidden_dim: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")

        # Build baseline CNN and optionally load checkpoint weights.
        cnn = CNNGeo(backbone_name=backbone_name, pretrained=True)
        if checkpoint_path:
            try:
                state = torch.load(checkpoint_path, map_location="cpu")
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                cnn.load_state_dict(state, strict=False)
                print(f"Loaded baseline checkpoint from {checkpoint_path}")
            except Exception as exc:
                print(f"Warning: failed to load checkpoint {checkpoint_path}: {exc}")

        # Use the backbone as a frozen feature extractor.
        self.backbone = cnn.backbone.to(self.device)
        for p in self.backbone.parameters():
            p.requires_grad = False
        feat_dim = self.backbone.num_features

        # Optional small bottleneck MLP (kept simple by default).
        if hidden_dim > 0:
            self.encoder_head = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
            policy_input_dim = hidden_dim
        else:
            self.encoder_head = None
            policy_input_dim = feat_dim

        self.policy_head = nn.Linear(policy_input_dim, num_actions)
        self.value_head = nn.Linear(policy_input_dim, 1)
        self.coord_head = nn.Linear(policy_input_dim, 2)

        self.attn_query = nn.Parameter(torch.zeros(policy_input_dim))
        self.attn_key = nn.Linear(policy_input_dim, policy_input_dim)
        self.attn_value = nn.Linear(policy_input_dim, policy_input_dim)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Run crop through backbone (no gradients on backbone). Returns (B, feat_dim)."""
        with torch.no_grad():
            feat = self.backbone(obs)
        if self.encoder_head is not None:
            feat = self.encoder_head(feat)
        return feat

    def forward(
        self, embedding_sequence: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """embedding_sequence (B, T, D), mask (B, T) 1=valid 0=pad -> logits, value, coords."""
        B, T, D = embedding_sequence.shape
        K = self.attn_key(embedding_sequence)
        V = self.attn_value(embedding_sequence)
        scores = (K @ self.attn_query.unsqueeze(-1)).squeeze(-1) / (D ** 0.5)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        context = (weights.unsqueeze(1) @ V).squeeze(1)
        logits = self.policy_head(context)
        value = self.value_head(context)
        coords = self.coord_head(context)
        return logits, value.squeeze(-1), coords


__all__ = ["RLGeoPolicy"]

