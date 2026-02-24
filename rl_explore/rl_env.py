#!/usr/bin/env python3
"""
Simple image panning/zooming environment for RL.
Each episode = one panorama; agent moves a crop and eventually terminates with a coordinate guess.

We keep this self-contained (no gym dependency) and use tensors directly in the training loop.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms

from baseline.train_baseline_CNN import (
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    haversine_km,
)


@dataclass
class Sample:
    path: str
    lat: float
    lon: float


class GeoPanningEnv:
    """
    Minimal environment for sequential visual exploration.

    - Underlying state: full RGB image + (lat, lon).
    - Agent state: crop center + zoom level.
    - Observation: current crop, already transformed for the CNN backbone.
    - Actions (discrete ints):
        0: pan left
        1: pan right
        2: pan up
        3: pan down
        4: zoom in
        5: zoom out
        6: terminate and predict

    Step API is slightly extended: for terminate, you must pass pred_latlon.
    """

    ACTION_PAN_LEFT = 0
    ACTION_PAN_RIGHT = 1
    ACTION_PAN_UP = 2
    ACTION_PAN_DOWN = 3
    ACTION_ZOOM_IN = 4
    ACTION_ZOOM_OUT = 5
    ACTION_TERMINATE = 6

    def __init__(
        self,
        samples: List[Tuple[str, float, float]],
        max_steps: int = 10,
        step_penalty: float = 0.01,
        alpha: float = 1.0,
        pan_frac: float = 0.2,
        zoom_levels: Optional[Tuple[float, ...]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:
            samples: list of (path, lat, lon).
            max_steps: maximum steps before forced termination.
            step_penalty: per-move negative reward.
            alpha: reward scale for terminal distance penalty.
            pan_frac: fraction of image dimension to move per pan action.
            zoom_levels: relative crop size levels; smaller => more zoom.
            device: device to put tensors on.
        """
        self.samples = [Sample(*s) for s in samples]
        if not self.samples:
            raise ValueError("GeoPanningEnv requires at least one sample.")

        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.alpha = alpha
        self.pan_frac = pan_frac
        self.zoom_levels = zoom_levels or (1.0, 0.7, 0.5)
        self.device = device or torch.device("cpu")

        # Transform matches the baseline CNN.
        self.transform = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        # Internal state
        self._rng = random.Random(42)
        self._img: Optional[Image.Image] = None
        self._img_w: int = 0
        self._img_h: int = 0
        self._lat: float = 0.0
        self._lon: float = 0.0
        self._center_x: float = 0.5  # normalized [0, 1]
        self._center_y: float = 0.5
        self._zoom_idx: int = 0
        self._t: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> torch.Tensor:
        """Start a new episode and return the first crop tensor."""
        sample = self._rng.choice(self.samples)
        img_path = Path(sample.path)
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")
        self._img = Image.open(img_path).convert("RGB")
        self._img_w, self._img_h = self._img.size
        self._lat, self._lon = sample.lat, sample.lon

        # Reset camera roughly to center, medium zoom.
        self._center_x = 0.5
        self._center_y = 0.5
        self._zoom_idx = min(1, len(self.zoom_levels) - 1)
        self._t = 0

        return self._get_obs()

    def step(
        self,
        action: int,
        pred_latlon: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, bool, dict]:
        """
        Take one environment step.

        For ACTION_TERMINATE you must pass pred_latlon as a (2,) tensor [lat, lon] in degrees.
        Returns: (next_obs, reward, done, info)
        """
        if self._img is None:
            raise RuntimeError("Call reset() before step().")

        reward = 0.0
        done = False
        info = {}

        if action == self.ACTION_TERMINATE:
            if pred_latlon is None:
                raise ValueError("pred_latlon must be provided for terminate action.")
            pred_lat = float(pred_latlon[0].item())
            pred_lon = float(pred_latlon[1].item())
            dist_km = haversine_km(pred_lat, pred_lon, self._lat, self._lon)
            # Log-scaled distance penalty.
            reward = -self.alpha * math.log1p(dist_km)
            done = True
            info["dist_km"] = dist_km
            info["true_lat"] = self._lat
            info["true_lon"] = self._lon
            info["pred_lat"] = pred_lat
            info["pred_lon"] = pred_lon
        else:
            # Movement step.
            self._apply_move(action)
            reward = -self.step_penalty

        self._t += 1
        if not done and self._t >= self.max_steps:
            # Force termination with zero reward contribution here; the training
            # loop can choose to compute a prediction and add its reward.
            done = True
            info["forced_termination"] = True

        obs = self._get_obs()
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_move(self, action: int) -> None:
        """Update camera center / zoom within bounds."""
        # Convert pan fraction to pixel displacement based on current crop size.
        zoom_scale = self.zoom_levels[self._zoom_idx]
        crop_w = self._img_w * zoom_scale
        crop_h = self._img_h * zoom_scale
        dx = self.pan_frac * (crop_w / self._img_w)
        dy = self.pan_frac * (crop_h / self._img_h)

        if action == self.ACTION_PAN_LEFT:
            self._center_x -= dx
        elif action == self.ACTION_PAN_RIGHT:
            self._center_x += dx
        elif action == self.ACTION_PAN_UP:
            self._center_y -= dy
        elif action == self.ACTION_PAN_DOWN:
            self._center_y += dy
        elif action == self.ACTION_ZOOM_IN:
            self._zoom_idx = min(self._zoom_idx + 1, len(self.zoom_levels) - 1)
        elif action == self.ACTION_ZOOM_OUT:
            self._zoom_idx = max(self._zoom_idx - 1, 0)

        # Clamp center to [0, 1].
        self._center_x = float(max(0.0, min(1.0, self._center_x)))
        self._center_y = float(max(0.0, min(1.0, self._center_y)))

    def _get_obs(self) -> torch.Tensor:
        """Return current crop as a normalized tensor."""
        assert self._img is not None

        zoom_scale = self.zoom_levels[self._zoom_idx]
        crop_w = int(self._img_w * zoom_scale)
        crop_h = int(self._img_h * zoom_scale)

        cx = int(self._center_x * self._img_w)
        cy = int(self._center_y * self._img_h)

        left = max(0, cx - crop_w // 2)
        top = max(0, cy - crop_h // 2)
        right = min(self._img_w, left + crop_w)
        bottom = min(self._img_h, top + crop_h)

        # Adjust if at border so crop has the right size.
        left = max(0, right - crop_w)
        top = max(0, bottom - crop_h)

        crop = self._img.crop((left, top, right, bottom))
        tensor = self.transform(crop).to(self.device)
        return tensor


__all__ = ["GeoPanningEnv"]

