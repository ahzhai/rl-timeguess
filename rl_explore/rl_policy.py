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

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from baseline.train_baseline_CNN import CNNGeo


class RLGeoPolicy(nn.Module):
    """
    Actor-critic with Simple Embedding Aggregation (mean pooling).

    For each episode we maintain a history of backbone embeddings for all crops
    seen so far. These are mean-pooled to form a single context vector, which
    is passed to the policy, value, and coord heads.

      crop -> frozen CNN backbone -> embedding (once per step)
      list of embeddings -> mean pool -> context vector
      context vector -> policy logits | value | (lat, lon)
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
        """Run crop through backbone (no gradients on backbone). Returns (B, feat_dim)."""
        with torch.no_grad():
            feat = self.backbone(obs)
        if self.encoder_head is not None:
            feat = self.encoder_head(feat)
        return feat

    def aggregate_embeddings(
        self, list_of_embeddings: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Mean-pool a list of embeddings over the temporal dimension to form a
        single context vector per batch item.

        Args:
            list_of_embeddings: List of tensors, each (B, feat_dim) or (1, feat_dim).

        Returns:
            (B, feat_dim) tensor.
        """
        if not list_of_embeddings:
            raise ValueError("aggregate_embeddings requires at least one embedding.")
        stacked = torch.stack(list_of_embeddings, dim=0)
        return torch.mean(stacked, dim=0)

    def forward(
        self, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run actor, critic, and coord heads on an aggregated context vector.

        Args:
            context: (B, feat_dim) tensor — mean-pooled embeddings for the episode so far.

        Returns:
            logits: (B, num_actions)
            value:  (B,)
            coords: (B, 2) predicted [lat, lon] in degrees
        """
        logits = self.policy_head(context)
        value = self.value_head(context)
        coords = self.coord_head(context)
        return logits, value.squeeze(-1), coords


__all__ = ["RLGeoPolicy"]

