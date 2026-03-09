#!/usr/bin/env python3
"""
Minimal PPO-style training loop for geo-localization RL.

This is intentionally compact and self-contained:
  - loads the same metadata/images as the baseline CNN script
  - uses GeoPanningEnv for sequential panning/zooming
  - uses RLGeoPolicy with a frozen CNN backbone

Usage (example):
  python -m rl_explore.rl_train --metadata data/gsv-cities/Dataframes --data_root data/gsv-cities
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

from baseline.train_baseline_CNN import (
    RANDOM_SEED,
    load_metadata,
    median_geoguessr_score,
)
from rl_explore.rl_env import GeoPanningEnv
from rl_explore.rl_policy import RLGeoPolicy


@dataclass
class Transition:
    obs: torch.Tensor
    action: int
    logp: float
    value: float
    reward: float
    done: bool
    pred_coords: torch.Tensor
    agg_embedding: torch.Tensor  # aggregated past embeddings (for history); (1, embed_dim)
    true_coords: torch.Tensor  # (1, 2) ground-truth [lat, lon] for auxiliary loss


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _aggregate_history(buffer: list, device: torch.device, embed_dim: int) -> torch.Tensor:
    """Aggregate buffer of (embed_dim,) tensors via mean. Returns (1, embed_dim)."""
    if not buffer:
        return torch.zeros(1, embed_dim, device=device)
    stacked = torch.stack(buffer, dim=0)
    return stacked.mean(dim=0, keepdim=True)


def rollout(
    env: GeoPanningEnv,
    policy: RLGeoPolicy,
    steps_per_batch: int,
    gamma: float,
    device: torch.device,
) -> Tuple[List[Transition], List[float]]:
    """
    Collect a batch of transitions using the current policy.
    When policy.history_len > 0, maintains a buffer of last K embeddings and passes
    their mean as agg_embedding. Buffer is cleared on episode end.
    Returns transitions and list of episode returns.
    """
    policy.eval()
    transitions: List[Transition] = []
    episode_returns: List[float] = []
    embed_dim = policy.embed_dim
    history_len = policy.history_len
    buffer: List[torch.Tensor] = []

    obs = env.reset()
    ep_ret = 0.0
    steps_collected = 0

    while steps_collected < steps_per_batch:
        obs_batch = obs.unsqueeze(0)  # (1, C, H, W)
        agg = _aggregate_history(buffer, device, embed_dim) if history_len > 0 else None
        logits, value, coords = policy(obs_batch, agg_embedding=agg)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)

        action_int = int(action.item())
        pred_coords = coords[0]

        # For terminate, pass predicted coords into env.step so it can compute reward.
        if action_int == env.ACTION_TERMINATE:
            next_obs, reward, done, info = env.step(action_int, pred_coords)
        else:
            next_obs, reward, done, info = env.step(action_int)

        ep_ret += reward

        # Store agg used for this step (for PPO); use zeros when history disabled.
        stored_agg = (
            _aggregate_history(buffer, device, embed_dim).to(device)
            if history_len > 0
            else torch.zeros(1, embed_dim, device=device)
        )
        true_coords = torch.tensor(
            [[info["true_lat"], info["true_lon"]]],
            device=device,
            dtype=torch.float32,
        )
        transitions.append(
            Transition(
                obs=obs.to(device),
                action=action_int,
                logp=float(logp.item()),
                value=float(value.item()),
                reward=float(reward),
                done=done,
                pred_coords=pred_coords.detach().to(device),
                agg_embedding=stored_agg,
                true_coords=true_coords,
            )
        )
        steps_collected += 1

        # Append current embedding to buffer (for next step), keep last history_len.
        if history_len > 0:
            emb = policy.encode(obs_batch).squeeze(0).detach()
            buffer.append(emb)
            if len(buffer) > history_len:
                buffer.pop(0)

        if done:
            episode_returns.append(ep_ret)
            obs = env.reset()
            ep_ret = 0.0
            buffer = []
        else:
            obs = next_obs

    return transitions, episode_returns


def compute_gae(
    transitions: List[Transition],
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute advantages and returns with GAE.
    Outputs tensors for obs, agg, actions, old_logp, returns, advantages, true_coords.
    """
    rewards = [t.reward for t in transitions]
    values = [t.value for t in transitions]
    dones = [t.done for t in transitions]

    values_tensor = torch.tensor(values, dtype=torch.float32)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

    advantages = torch.zeros_like(rewards_tensor)
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(len(transitions))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values_tensor

    obs_batch = torch.stack([t.obs for t in transitions], dim=0)
    agg_batch = torch.cat([t.agg_embedding for t in transitions], dim=0)
    true_coords_batch = torch.cat([t.true_coords for t in transitions], dim=0)
    actions_batch = torch.tensor([t.action for t in transitions], dtype=torch.long)
    logp_batch = torch.tensor([t.logp for t in transitions], dtype=torch.float32)

    # Normalize advantages for stability.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return obs_batch, agg_batch, actions_batch, logp_batch, returns, advantages, true_coords_batch


def ppo_update(
    policy: RLGeoPolicy,
    optimizer: torch.optim.Optimizer,
    obs_batch: torch.Tensor,
    agg_batch: torch.Tensor,
    actions_batch: torch.Tensor,
    logp_old_batch: torch.Tensor,
    returns_batch: torch.Tensor,
    adv_batch: torch.Tensor,
    true_coords_batch: torch.Tensor,
    clip_ratio: float,
    vf_coef: float,
    ent_coef: float,
    aux_coef: float,
    epochs: int,
    minibatch_size: int,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Run several epochs of PPO updates over the collected batch. Includes auxiliary SmoothL1 loss on coords."""
    policy.train()
    n = obs_batch.size(0)
    indices = torch.arange(n)

    last_pi_loss, last_v_loss, last_ent, last_aux_loss = 0.0, 0.0, 0.0, 0.0

    for _ in range(epochs):
        perm = indices[torch.randperm(n)]
        for start in range(0, n, minibatch_size):
            end = min(start + minibatch_size, n)
            mb_idx = perm[start:end]

            obs_mb = obs_batch[mb_idx].to(device)
            agg_mb = agg_batch[mb_idx].to(device)
            act_mb = actions_batch[mb_idx].to(device)
            logp_old_mb = logp_old_batch[mb_idx].to(device)
            ret_mb = returns_batch[mb_idx].to(device)
            adv_mb = adv_batch[mb_idx].to(device)
            true_coords_mb = true_coords_batch[mb_idx].to(device)

            logits, value, coords = policy(obs_mb, agg_embedding=agg_mb)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act_mb)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - logp_old_mb)
            unclipped = ratio * adv_mb
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_mb
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = F.mse_loss(value, ret_mb)
            aux_loss = F.smooth_l1_loss(coords, true_coords_mb)

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy + aux_coef * aux_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            last_pi_loss = float(policy_loss.item())
            last_v_loss = float(value_loss.item())
            last_ent = float(entropy.item())
            last_aux_loss = float(aux_loss.item())

    return last_pi_loss, last_v_loss, last_ent, last_aux_loss


def evaluate_policy(
    samples: List[Tuple[str, float, float]],
    policy: RLGeoPolicy,
    device: torch.device,
    max_steps: int = 10,
) -> Tuple[float, float]:
    """Run greedy evaluation on a subset of samples and report median/mean distance."""
    policy.eval()
    env = GeoPanningEnv(samples, max_steps=max_steps, device=device)
    embed_dim = policy.embed_dim
    history_len = policy.history_len

    preds = []
    trues = []

    with torch.no_grad():
        for sample in samples:
            obs = env.reset()
            buffer: List[torch.Tensor] = []
            done = False
            steps = 0
            while not done and steps < max_steps:
                obs_b = obs.unsqueeze(0)
                agg = (
                    _aggregate_history(buffer, device, embed_dim)
                    if history_len > 0
                    else None
                )
                logits, _, coords = policy(obs_b, agg_embedding=agg)
                dist = torch.distributions.Categorical(logits=logits)
                # Greedy action for eval.
                action = torch.argmax(dist.logits, dim=-1)
                action_int = int(action.item())
                pred_coords = coords[0]

                if history_len > 0:
                    emb = policy.encode(obs_b).squeeze(0)
                    buffer.append(emb)
                    if len(buffer) > history_len:
                        buffer.pop(0)

                if action_int == env.ACTION_TERMINATE or steps == max_steps - 1:
                    # Force termination at last step.
                    _, _, _, info = env.step(env.ACTION_TERMINATE, pred_coords)
                    preds.append(pred_coords.cpu())
                    trues.append(torch.tensor([info["true_lat"], info["true_lon"]]))
                    done = True
                else:
                    obs, _, done, _ = env.step(action_int)
                steps += 1

    if not preds:
        return float("nan"), float("nan")
    pred_tensor = torch.stack(preds, dim=0)
    true_tensor = torch.stack(trues, dim=0)
    med_score, mean_score = median_geoguessr_score(pred_tensor, true_tensor)
    return med_score, mean_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/gsv-cities/Dataframes",
        help="Path to per-city CSVs (same as baseline).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/gsv-cities",
        help="Root directory for images (same as baseline).",
    )
    parser.add_argument("--samples_per_city", type=int, default=100)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument(
        "--baseline_checkpoint",
        type=str,
        default="",
        help="Optional path to pretrained baseline CNN weights.",
    )
    # RL hyperparameters (kept simple).
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps_per_batch", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--step_penalty", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--history_len",
        type=int,
        default=5,
        help="Number of past embeddings to aggregate (0 = no history).",
    )
    parser.add_argument(
        "--aux_coef",
        type=float,
        default=0.5,
        help="Weight for auxiliary SmoothL1 loss on predicted vs true (lat, lon).",
    )

    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print(f"Loading metadata from {args.metadata} ...")
    samples = load_metadata(args.metadata, args.data_root, sample_per_city=args.samples_per_city)
    if not samples:
        raise SystemExit("No samples found. Check --metadata and --data_root.")
    print(f"Total samples: {len(samples)}")

    # 80/10/10 split as in baseline.
    n = len(samples)
    random.Random(RANDOM_SEED).shuffle(samples)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_samples = samples[:n_train]
    val_samples = samples[n_train : n_train + n_val]
    test_samples = samples[n_train + n_val :]

    device = select_device()
    print(f"Using device: {device}")

    env = GeoPanningEnv(
        train_samples,
        max_steps=args.max_steps,
        step_penalty=args.step_penalty,
        alpha=args.alpha,
        device=device,
    )
    policy = RLGeoPolicy(
        backbone_name=args.backbone,
        checkpoint_path=args.baseline_checkpoint or None,
        num_actions=7,
        hidden_dim=0,
        history_len=args.history_len,
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr,
    )

    for epoch in range(1, args.epochs + 1):
        transitions, ep_returns = rollout(env, policy, args.steps_per_batch, args.gamma, device)
        obs_b, agg_b, act_b, logp_b, ret_b, adv_b, true_coords_b = compute_gae(
            transitions,
            gamma=args.gamma,
            lam=args.gae_lambda,
        )

        pi_loss, v_loss, ent, aux_loss = ppo_update(
            policy,
            optimizer,
            obs_b,
            agg_b,
            act_b,
            logp_b,
            ret_b,
            adv_b,
            true_coords_b,
            clip_ratio=args.clip_ratio,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            aux_coef=args.aux_coef,
            epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            device=device,
        )

        avg_ret = sum(ep_returns) / max(1, len(ep_returns))
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"episodes={len(ep_returns)} | "
            f"avg_return={avg_ret:.3f} | "
            f"pi_loss={pi_loss:.4f} | v_loss={v_loss:.4f} | ent={ent:.3f} | aux_loss={aux_loss:.4f}"
        )

        # Periodic evaluation on validation set (subset for speed).
        if val_samples:
            subset = val_samples[: min(512, len(val_samples))]
            med_score, mean_score = evaluate_policy(subset, policy, device, max_steps=args.max_steps)
            print(
                f"  Val median GeoGuessr score: {med_score:.2f} | "
                f"mean: {mean_score:.2f}"
            )

    # Final evaluation on test set (same metric as baseline for comparison).
    if test_samples:
        subset = test_samples[: min(1024, len(test_samples))]
        med_score, mean_score = evaluate_policy(subset, policy, device, max_steps=args.max_steps)
        print("\nFinal test performance (GeoGuessr score, 0–5000):")
        print(f"  Median GeoGuessr score: {med_score:.2f}")
        print(f"  Mean GeoGuessr score:   {mean_score:.2f}")


if __name__ == "__main__":
    main()

