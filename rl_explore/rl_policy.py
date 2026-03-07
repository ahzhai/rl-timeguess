#!/usr/bin/env python3
"""
Policy/value network for geo-localization RL with a frozen CNN backbone.

We reuse the baseline CNN backbone from baseline.train_baseline_CNN, drop its head,
and add:
  - policy head over discrete actions
  - value head
  - coordinate regression head (for terminate predictions)

Optional: embedding aggregation over the last K observations (history_len > 0).
Current observation embedding is concatenated with aggregated past embeddings
and projected before the policy/value/coord heads.
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
           -> [optional: concat with aggregated history -> project]
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
        history_len: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device("cpu")
        self.history_len = max(0, history_len)

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

        self.embed_dim = policy_input_dim

        # When using history, concat [current_emb, agg_emb] and project back to policy_input_dim.
        if self.history_len > 0:
            self.combine_proj = nn.Sequential(
                nn.Linear(2 * policy_input_dim, policy_input_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.combine_proj = None

        self.policy_head = nn.Linear(policy_input_dim, num_actions)
        self.value_head = nn.Linear(policy_input_dim, 1)
        self.coord_head = nn.Linear(policy_input_dim, 2)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Run crop through backbone (no gradients on backbone). Returns (B, embed_dim)."""
        with torch.no_grad():
            feat = self.backbone(obs)
        if self.encoder_head is not None:
            feat = self.encoder_head(feat)
        return feat

    def forward(
        self,
        obs: torch.Tensor,
        agg_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (B, 3, H, W) tensor.
            agg_embedding: optional (B, embed_dim) aggregated history. If history_len > 0
                and None, zeros are used. Ignored when history_len == 0.

        Returns:
            logits: (B, num_actions)
            value:  (B, 1)
            coords: (B, 2) predicted [lat, lon] in degrees
        """
        feat = self.encode(obs)
        if self.history_len > 0:
            if agg_embedding is None:
                agg_embedding = torch.zeros(
                    feat.size(0), self.embed_dim, device=feat.device, dtype=feat.dtype
                )
            combined = torch.cat([feat, agg_embedding], dim=-1)
            feat = self.combine_proj(combined)
        logits = self.policy_head(feat)
        value = self.value_head(feat)
        coords = self.coord_head(feat)
        return logits, value.squeeze(-1), coords


__all__ = ["RLGeoPolicy"]

