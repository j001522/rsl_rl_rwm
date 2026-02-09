# Phase 1: Latent Space Bottleneck - Implementation Changes

**Branch:** `feature/latent-space` (both repos)  
**Parent branches:** `random_priors` (rsl_rl_rwm), `uncertainty-modeling` (robotic_world_model)  
**Date started:** February 2026

---

## Overview

This document tracks all code changes for Phase 1 of the TD-MPC2/RWM hybrid architecture.
Phase 1 adds a **latent-space bottleneck** to the RWM system dynamics, allowing the world
model to learn compressed representations instead of predicting raw states directly.

### Key Architectural Decisions

1. **Backbone processes encoded states**: In latent mode, the RNN/MLP backbone receives
   `(encoded_latent, action)` pairs, not raw `(state, action)`.

2. **TD-MPC2-style encoder**: Uses NormedLinear (Linear -> LayerNorm -> Mish) + SimNorm
   on the output layer. Latent representations lie on product of simplices.

3. **No aleatoric uncertainty in latent mode**: `LatentDynamicsHead` does not predict std.
   Uncertainty is purely epistemic (ensemble disagreement in latent space).

4. **JEPA-style consistency targets**: Encoder targets are computed with `torch.no_grad()`
   and `.detach()`. The encoder trains through initial history encoding and reconstruction
   loss, NOT through target computation (prevents representation collapse).

5. **Randomized priors**: Prior perturbation added BEFORE SimNorm, so both learned
   prediction and diversity noise are jointly normalized. Same detach pattern as raw mode.

6. **Auxiliary branch stays in raw state space**: Always uses `state_dim` input dimensions,
   even in latent mode. Physical predictions (contact, termination) don't benefit from latent encoding.

7. **Full backward compatibility**: All new parameters default to reproducing original
   behavior (`latent_mode=False`).

---

## Repository: `rsl_rl_rwm`

Path: `/gpfs/work4/0/prjs0951/Giacomo/isaac-sim/overlay/rsl_rl_rwm/`

### New Files

#### 1. `rsl_rl/modules/architectures/latent_layers.py` -- COMPLETE

Latent-space building blocks adapted from TD-MPC2.

| Component | Description |
|-----------|-------------|
| `SimNorm(simnorm_dim=8)` | Splits input into groups, applies softmax per group. Constrains each group to a probability simplex. |
| `NormedLinear(in_features, out_features, act, dropout)` | `Linear -> (Dropout) -> LayerNorm -> Activation`. Default activation: Mish. |
| `latent_mlp(in_dim, hidden_dims, out_dim, output_act, dropout)` | MLP builder. Hidden layers use NormedLinear (Mish). Output layer: NormedLinear with custom act if provided, else plain Linear. |

#### 2. `rsl_rl/modules/architectures/encoder_decoder.py` -- COMPLETE

| Component | Description |
|-----------|-------------|
| `StateEncoder(state_dim, latent_dim, hidden_dims, simnorm_dim, dropout)` | Maps raw state -> latent. Uses `latent_mlp` with `SimNorm` output activation. Asserts `latent_dim % simnorm_dim == 0`. |
| `StateDecoder(latent_dim, state_dim, hidden_dims, dropout)` | Maps latent -> raw state. Uses `latent_mlp` with plain Linear output (unconstrained values). |

### Modified Files

#### 3. `rsl_rl/modules/architectures/mlp.py` -- COMPLETE

**Changes:**
- Added `from rsl_rl.modules.architectures.latent_layers import SimNorm, NormedLinear, latent_mlp` (line 5)
- Added `LatentDynamicsHead` class (lines 38-158)
- **BUG FIX NEEDED:** The original `MLPStateHead` class was accidentally deleted during editing.
  Must be restored between `MLPBase` and `LatentDynamicsHead`. (**STATUS: FIXED**)

**`LatentDynamicsHead` details:**
- Predicts next latent state with SimNorm output (no residual connection)
- `output_std = False` (no aleatoric uncertainty)
- Main network: `latent_mlp()` (NormedLinear hidden + plain Linear output before SimNorm)
- Prior network: plain `Linear + ReLU` layers with Xavier init, output `.detach()`-ed
- SimNorm applied after `main_output + prior_output * prior_scale`
- `forward(x, x_state_batch=None)` returns `(latent_mean, zeros)` for API compatibility

#### 4. `rsl_rl/modules/system_dynamics.py` -- COMPLETE (with bug fix)

**New constructor parameters (all with defaults for backward compat):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_mode` | `False` | Opt-in flag for latent space dynamics |
| `latent_dim` | `256` | Latent representation size |
| `simnorm_dim` | `8` | SimNorm group size |
| `encoder_hidden_dims` | `[256]` | Encoder hidden layer widths |
| `decoder_hidden_dims` | `[256]` | Decoder hidden layer widths |
| `latent_head_hidden_dims` | `[256]` | Latent dynamics head hidden widths |
| `encoder_dropout` | `0.0` | Encoder dropout rate |
| `consistency_coef` | `2.0` | Weight for consistency loss in latent space |
| `reconstruction_coef` | `1.0` | Weight for reconstruction loss |

**Key method changes:**

| Method | Change |
|--------|--------|
| `__init__()` | Stores latent config. `self.latent_dim = latent_dim if latent_mode else state_dim` |
| `_init_networks()` | Creates encoder/decoder when latent_mode. Uses `LatentDynamicsHead` for state heads. |
| `_create_base()` | Accepts `use_latent_dim` param. In latent mode with `use_latent_dim=True`, uses `latent_dim` for input; otherwise `state_dim`. State base: `use_latent_dim=True`, Auxiliary base: `use_latent_dim=False`. |
| `encode(raw_states)` | Returns `self.encoder(raw_states)` or pass-through |
| `decode(latent_states)` | Returns `self.decoder(latent_states)` or pass-through |
| `forward()` | Encodes inputs, runs backbone+heads in latent space, decodes output to raw state |
| `forward_latent()` | NEW: Takes/returns latent states directly (for autoregressive rollouts) |
| `compute_loss()` | Returns **9 values** (was 7): added `consistency_loss`, `reconstruction_loss` |
| `compute_state_loss()` | JEPA-style: encodes targets with `no_grad`, encodes initial history with grads. Consistency = MSE in latent space. Reconstruction = MSE in raw space after decode. |
| `compute_auxiliary_loss()` | Unchanged. Always operates on raw states. |

**BUG FIX:** `_create_base()` was using `self.latent_dim` for BOTH state and auxiliary bases.
Fixed by adding `use_latent_dim` parameter. Auxiliary base always uses `state_dim`. (**STATUS: FIXED**)

#### 5. `rsl_rl/modules/architectures/__init__.py` -- UPDATED

**Before:**
```python
from .mlp import MLPBase, MLPStateHead, MLPStateHeadWithPrior, MLPAuxiliaryHead
from .rnn import RNNBase
```

**After:**
```python
from .mlp import MLPBase, MLPStateHead, MLPStateHeadWithPrior, MLPAuxiliaryHead, LatentDynamicsHead
from .latent_layers import SimNorm, NormedLinear, latent_mlp
from .encoder_decoder import StateEncoder, StateDecoder
from .rnn import RNNBase
```

#### 6. `rsl_rl/algorithms/mbpo_ppo.py` -- UPDATED

**Changes to `update_system_dynamics()` (line 192):**
- Unpacks 9 values from `compute_loss()` (was 7): added `consistency_loss`, `reconstruction_loss`
- Adds `consistency` and `reconstruction` to `system_dynamics_loss_weights` in total loss
- Tracks and returns `mean_system_consistency_loss` and `mean_system_reconstruction_loss`
- Return signature: 9 values (was 7)

**Default loss weights updated:**
```python
system_dynamics_loss_weights={
    "state": 1.0, "sequence": 1.0, "bound": 1.0, "kl": 1.0,
    "consistency": 0.0, "reconstruction": 0.0,  # NEW (0.0 = no effect in raw mode)
    "extension": 1.0, "contact": 1.0, "termination": 1.0
}
```
Note: `consistency` and `reconstruction` losses are already folded into `state_loss` via
`consistency_coef` and `reconstruction_coef` in `compute_state_loss()`. The weights here
allow further external scaling but default to 0.0 since `state_loss` already includes them.
Set to 1.0 if you want to log/weight them separately as part of the total loss.

#### 7. `rsl_rl/runners/mbpo_on_policy_runner.py` -- UPDATED

**Changes to `learn()` (line 132):**
- Unpacks 9 values from `update_system_dynamics()` (was 7)

**Changes to `log()` (line 217):**
- Logs `System Dynamics/consistency_loss` and `System Dynamics/reconstruction_loss` to tensorboard

---

## Repository: `robotic_world_model`

Path: `/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/`

#### 8. `source/mbrl/mbrl/rl/rsl_rl/rl_cfg.py` -- UPDATED

**`RslRlSystemDynamicsCfg` -- new fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `latent_mode` | `bool` | `False` | Enable latent space dynamics |
| `latent_dim` | `int` | `256` | Latent representation dimension |
| `simnorm_dim` | `int` | `8` | SimNorm group size |
| `encoder_hidden_dims` | `list[int]` | `[256]` | Encoder hidden widths |
| `decoder_hidden_dims` | `list[int]` | `[256]` | Decoder hidden widths |
| `latent_head_hidden_dims` | `list[int]` | `[256]` | Latent dynamics head hidden widths |
| `encoder_dropout` | `float` | `0.0` | Encoder dropout |
| `consistency_coef` | `float` | `2.0` | Consistency loss coefficient |
| `reconstruction_coef` | `float` | `1.0` | Reconstruction loss coefficient |

**`RslRlMbrlPpoAlgorithmCfg` -- note:**
The `system_dynamics_loss_weights` field already exists as `dict[str, float] = MISSING`.
Users must include `"consistency"` and `"reconstruction"` keys when configuring in latent mode.
Default values in the algorithm constructor handle the case when they're missing (defaulting to 0.0).

---

## Loss Flow Diagram

```
compute_state_loss() per ensemble member:
  |
  |-- Raw mode (latent_mode=False):
  |     state_loss = MSE(predicted_state, target_state)  [with residual]
  |     consistency_loss = 0
  |     reconstruction_loss = 0
  |
  |-- Latent mode (latent_mode=True):
  |     latent_target = encoder(next_state).detach()     [no_grad, JEPA-style]
  |     latent_pred = head(backbone(encoder(current), action))
  |     consistency_loss = MSE(latent_pred, latent_target)
  |     reconstruction_loss = MSE(decoder(latent_pred), next_state)
  |     state_loss = consistency_coef * consistency_loss + reconstruction_coef * reconstruction_loss

compute_loss() aggregates across ensemble -> returns 9 values

update_system_dynamics() in mbpo_ppo.py:
  total_loss = w_state * state_loss + w_sequence * seq_loss + w_bound * bound_loss
             + w_kl * kl_loss + w_consistency * consistency_loss + w_reconstruction * recon_loss
             + w_extension * ext_loss + w_contact * contact_loss + w_termination * term_loss
```

---

## Known Issues / Decisions Deferred

1. **Double-counting concern**: In latent mode, `state_loss` already includes consistency
   and reconstruction losses (weighted by `consistency_coef` and `reconstruction_coef`).
   The external `system_dynamics_loss_weights["consistency"]` and `["reconstruction"]`
   provide additional scaling. Default 0.0 avoids double-counting. For separate logging,
   they're tracked independently regardless of the weight.

2. **Auxiliary base in latent mode**: Fixed. The auxiliary base always uses `state_dim`
   because it processes raw states for physical predictions.

3. **Encoder gradient flow**: Encoder receives gradients from:
   - Initial history encoding (through backbone -> head -> loss)
   - Reconstruction loss (through decoder)
   - NOT from target latent computation (detached for JEPA-style stability)

4. **`MLPStateHead` restoration**: The class was accidentally removed. Restored in this branch.

---

## Files Changed Summary

| File | Repo | Status |
|------|------|--------|
| `rsl_rl/modules/architectures/latent_layers.py` | rsl_rl_rwm | NEW |
| `rsl_rl/modules/architectures/encoder_decoder.py` | rsl_rl_rwm | NEW |
| `rsl_rl/modules/architectures/mlp.py` | rsl_rl_rwm | MODIFIED |
| `rsl_rl/modules/architectures/__init__.py` | rsl_rl_rwm | MODIFIED |
| `rsl_rl/modules/system_dynamics.py` | rsl_rl_rwm | MODIFIED |
| `rsl_rl/algorithms/mbpo_ppo.py` | rsl_rl_rwm | MODIFIED |
| `rsl_rl/runners/mbpo_on_policy_runner.py` | rsl_rl_rwm | MODIFIED |
| `source/mbrl/mbrl/rl/rsl_rl/rl_cfg.py` | robotic_world_model | MODIFIED |
