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

from baseline.train_baseline_CNN import CNNGeo


class RLGeoPolicy(nn.Module):
    """
    Minimal actor-critic model:

      crop -> frozen CNN backbone -> feature
           -> policy logits over actions
           -> state value
           -> (lat, lon) regression for terminate
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

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Run crop through backbone (no gradients on backbone)."""
        with torch.no_grad():
            feat = self.backbone(obs)
        if self.encoder_head is not None:
            feat = self.encoder_head(feat)
        return feat

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (B, 3, H, W) tensor.

        Returns:
            logits: (B, num_actions)
            value:  (B, 1)
            coords: (B, 2) predicted [lat, lon] in degrees
        """
        feat = self.encode(obs)
        logits = self.policy_head(feat)
        value = self.value_head(feat)
        coords = self.coord_head(feat)
        return logits, value.squeeze(-1), coords


__all__ = ["RLGeoPolicy"]

