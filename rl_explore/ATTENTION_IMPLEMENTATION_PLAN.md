# Implementation Plan: Attention-Based Embedding Aggregation

Replace mean-pooling with single-head cross-attention over the embedding sequence so the policy can weight which views matter. Correctness and simplicity are the priorities.

---

## 1. Design

- **Mechanism**: One learned query vector attends over the sequence of crop embeddings (cross-attention). Output is one context vector per batch item, then the same policy / value / coord heads as now.
- **Training**: Attention must receive gradients from PPO and the coord loss. Therefore store the **embedding sequence** (and mask) in each transition and **re-run attention** in the PPO forward; do not store only a precomputed context.
- **API**: Single entry point only. Replace `forward(context)` with `forward(embedding_sequence, mask)`. Remove mean-pooling and `aggregate_embeddings`.

---

## 2. Policy (`rl_policy.py`)

**New parameters**

- `attn_query`: `nn.Parameter` shape `(feat_dim,)`
- `attn_key`: `nn.Linear(feat_dim, feat_dim)`
- `attn_value`: `nn.Linear(feat_dim, feat_dim)`

**Attention (vectorized)**

- Inputs: `embedding_sequence` `(B, T, D)`, `mask` `(B, T)` with 1 = valid, 0 = padding.
- Steps:
  1. `K = attn_key(embedding_sequence)` → `(B, T, D)`
  2. `V = attn_value(embedding_sequence)` → `(B, T, D)`
  3. `scores = (K @ attn_query.unsqueeze(-1)).squeeze(-1) / sqrt(D)` → `(B, T)`  
     (So that a single query attends over the T positions.)
  4. `scores = scores.masked_fill(mask == 0, -1e9)`
  5. `weights = F.softmax(scores, dim=-1)` → `(B, T)`
  6. `context = (weights.unsqueeze(1) @ V).squeeze(1)` → `(B, D)`

**API**

- `forward(embedding_sequence, mask)` → run attention → `context` → same three heads → return `(logits, value, coords)`.
- Remove `aggregate_embeddings`.

**Initialization**

- Initialize `attn_query` to zeros so that before training, scores are zero and softmax is uniform (similar to mean pooling). Key/value use default `Linear` init.

**Edge case**

- T = 1: one position gets weight 1; no special case needed.

---

## 3. Transition (`rl_train.py`)

- **Remove** field `obs`.
- **Add**:
  - `embedding_sequence`: `(1, max_steps, feat_dim)` — padded with zeros for positions ≥ T.
  - `mask`: `(1, max_steps)` — 1 for positions `[0, T)`, 0 for `[T, max_steps)`.

All transitions use the same length `max_steps` so batching is a single `torch.cat` on dim 0.

---

## 4. Rollout (`rl_train.py`)

- **Signature**: Add argument `max_steps: int`.
- **Per step** (unchanged until aggregation): encode current crop, append to `current_episode_embeddings` (list of `(1, feat_dim)`).
- **Before policy call**:
  1. `T = len(current_episode_embeddings)`
  2. `stacked = torch.stack(current_episode_embeddings, dim=1)` → `(1, T, feat_dim)`
  3. Pad `stacked` to `(1, max_steps, feat_dim)` (zeros for indices `T..max_steps-1`). E.g. `padded = torch.zeros(1, max_steps, feat_dim, device=..., dtype=stacked.dtype); padded[:, :T] = stacked`
  4. `mask = torch.zeros(1, max_steps, ...); mask[:, :T] = 1`
  5. `logits, value, coords = policy(padded, mask)`
- **Store**: `embedding_sequence=padded.detach().to(device)`, `mask=mask.to(device)` (and other existing transition fields). Do not store pixels.

---

## 5. GAE and PPO (`rl_train.py`)

**compute_gae**

- From transitions: `embedding_sequences_batch = torch.cat([t.embedding_sequence for t in transitions], dim=0)` → `(N, max_steps, feat_dim)`.
- `masks_batch = torch.cat([t.mask for t in transitions], dim=0)` → `(N, max_steps)`.
- Return these instead of `obs_batch`; keep all other returns (actions, logp, returns, advantages, terminate_mask, true_coords_batch).

**ppo_update**

- Replace argument `obs_batch` with `embedding_sequences_batch` and `masks_batch`.
- Minibatch: `seq_mb = embedding_sequences_batch[mb_idx].to(device)`, `mask_mb = masks_batch[mb_idx].to(device)`.
- Call `policy(seq_mb, mask_mb)`. Rest of PPO (losses, backward, optimizer) unchanged.

---

## 6. Evaluation (`evaluate_policy`)

- Same as rollout: list of embeddings per episode; each step stack → pad to `(1, max_steps, feat_dim)` and build mask `(1, max_steps)`; call `policy(padded, mask)`; greedy action. `max_steps` is already an argument.

---

## 7. Main

- Pass `max_steps=args.max_steps` into `rollout`.
- Use `embedding_sequences_batch` and `masks_batch` from `compute_gae` and pass them into `ppo_update` instead of `obs_batch`.

---

## 8. Implementation order

1. **Policy**: Add attention params and computation; change `forward(embedding_sequence, mask)`; remove `aggregate_embeddings`; init query to zeros.
2. **Transition**: Replace `obs` with `embedding_sequence` and `mask`.
3. **Rollout**: Build sequence, pad, mask, call `policy(..., mask)`, store sequence and mask; add `max_steps` arg.
4. **compute_gae**: Build and return `embedding_sequences_batch`, `masks_batch`; update call sites.
5. **ppo_update**: New args, minibatch `seq_mb`/`mask_mb`, call `policy(seq_mb, mask_mb)`.
6. **evaluate_policy**: Use same pad + mask + `policy(..., mask)`.
7. **Main**: Wire `max_steps` and the two new batch tensors.
8. **Check**: Short run (e.g. 1 epoch, small batch) to confirm no shape errors and loss moves.

---

## 9. Optional later

- Return attention weights from `forward` for logging/visualization.
- Multi-head attention (multiple queries, then combine contexts).
