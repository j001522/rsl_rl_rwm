# Phase 2: EMA Target Encoder & Consistency Loss - Implementation Changes

**Branch:** `feature/consistency-loss` (both repos)  
**Parent branch:** `feature/latent-space`  
**Date started:** February 2026  
**Planned tag:** `v0.3.0-tdmpc2-consistency`

---

## Overview

This document tracks all code changes for Phase 2 of the TD-MPC2/RWM hybrid architecture.
Phase 2 replaces the stop-gradient consistency targets from Phase 1 with a proper
**EMA (Exponential Moving Average) target encoder** following the JEPA approach, and adds
an optional **bidirectional encoder consistency loss** (TD-MPC2 style).

### Why EMA Target Encoder?

In Phase 1, consistency targets were produced by the same online encoder under `torch.no_grad()`.
This is equivalent to stop-gradient, which is what TD-MPC2 uses. However, TD-MPC2 has
reward/value heads that provide additional learning signal to prevent representation collapse.

Since we don't yet have reward/value heads (Phase 3), the EMA target encoder provides
stronger anti-collapse guarantees by:

1. **Smoothly evolving targets**: The target encoder lags behind the online encoder,
   preventing the representation from changing too rapidly.
2. **Implicit regularization**: EMA acts as a form of temporal ensembling, stabilizing
   the consistency objective.
3. **JEPA alignment**: This matches the I-JEPA / V-JEPA / BYOL approach where a 
   momentum-updated target network provides stable prediction targets.

### Key Architectural Decisions

1. **EMA target encoder for consistency targets**: `θ_target ← m * θ_target + (1-m) * θ_online`
   where `m = target_encoder_momentum` (default 0.99).

2. **Target encoder has no gradients**: All parameters have `requires_grad=False`. 
   Updated only via EMA after each optimizer step.

3. **Target encoder always in eval mode**: Stays in eval mode even when the model is
   set to train mode (no dropout/batchnorm train behavior).

4. **Bidirectional encoder consistency (optional)**: When `encoder_consistency_coef > 0`,
   adds a TD-MPC2-style loss that trains the online encoder to produce outputs consistent
   with (stop-gradient) dynamics predictions. This provides an additional gradient path
   for the encoder.

5. **Reconstruction loss stays**: Retained as additional anti-collapse mechanism. The
   decoder provides a direct supervision signal in raw state space.

6. **Full backward compatibility**: New parameters default to reasonable values.
   `target_encoder_momentum=0.99` and `encoder_consistency_coef=0.0` reproduce
   similar behavior to Phase 1 (with the improvement of using EMA instead of 
   stop-gradient for targets).

7. **Save/load**: The target encoder state_dict is included in the model's `state_dict()`
   automatically as a submodule. On load, it is restored from the checkpoint. For fresh
   initialization, it is deep-copied from the online encoder.

---

## Repository: `rsl_rl_rwm`

Path: `/gpfs/work4/0/prjs0951/Giacomo/isaac-sim/overlay/rsl_rl_rwm/`

### Modified Files

#### 1. `rsl_rl/modules/system_dynamics.py`

**New import:**
- `import copy` (for `copy.deepcopy` of encoder)

**New constructor parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_encoder_momentum` | `0.99` | EMA momentum for target encoder. Higher = slower update. |
| `encoder_consistency_coef` | `0.0` | Weight for bidirectional encoder consistency loss. 0 = disabled. |

**New attribute initialization in `_init_networks()`:**
- `self.target_encoder = copy.deepcopy(self.encoder)` when `latent_mode=True`
- All target encoder parameters set to `requires_grad=False`
- `self.target_encoder = None` when `latent_mode=False`

**New method: `update_target_encoder(momentum=None)`**
- Decorated with `@torch.no_grad()`
- Performs EMA update: `θ_target ← m * θ_target + (1-m) * θ_online`
- Uses `self.target_encoder_momentum` if `momentum` is not provided
- No-op if `target_encoder is None` (raw mode)

**Changes to `compute_state_loss()`:**

| Aspect | Phase 1 (Before) | Phase 2 (After) |
|--------|------------------|-----------------|
| Consistency targets | `self.encoder(state_batch)` under `no_grad` | `self.target_encoder(state_batch)` under `no_grad` |
| Target detach | Explicit `.detach()` on each target | Not needed (target encoder has no grad) |
| Encoder consistency | Not present | Optional: `MSE(encoder(s_next), sg(latent_pred))` |
| Return values | 6 values | 7 values (added `encoder_consistency_loss`) |

**Changes to `compute_loss()`:**
- Tracks `encoder_consistency_losses` list alongside other losses
- Unpacks 7 values from `compute_state_loss()` (was 6)
- Returns 10 values (was 9): added `encoder_consistency_loss`

**Changes to `train()`:**
- Keeps `self.target_encoder.eval()` when model enters train mode

#### 2. `rsl_rl/algorithms/mbpo_ppo.py`

**Changes to `update_system_dynamics()`:**
- Tracks `mean_system_encoder_consistency_loss`
- Unpacks 10 values from `compute_loss()` (was 9)
- Adds `encoder_consistency` weight to total loss via `system_dynamics_loss_weights.get("encoder_consistency", 0.0)`
- Calls `self.system_dynamics.update_target_encoder()` after each `optimizer.step()`
- Returns 10 values (was 9)

#### 3. `rsl_rl/runners/mbpo_on_policy_runner.py`

**Changes to `learn()`:**
- Unpacks 10 values from `update_system_dynamics()` (was 9)

**Changes to `log()`:**
- Logs `System Dynamics/encoder_consistency_loss` to tensorboard

---

## Repository: `robotic_world_model`

Path: `/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/`

#### 4. `source/mbrl/mbrl/rl/rsl_rl/rl_cfg.py`

**`RslRlSystemDynamicsCfg` -- new fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_encoder_momentum` | `float` | `0.99` | EMA momentum for target encoder |
| `encoder_consistency_coef` | `float` | `0.0` | Bidirectional encoder consistency weight |

---

## Loss Flow Diagram (Phase 2)

```
compute_state_loss() per ensemble member:
  |
  |-- Raw mode (latent_mode=False):  [unchanged from Phase 1]
  |     state_loss = MSE(predicted_state, target_state)
  |     consistency_loss = 0, reconstruction_loss = 0, encoder_consistency_loss = 0
  |
  |-- Latent mode (latent_mode=True):
  |     latent_target = TARGET_ENCODER(next_state)    [EMA, no_grad -- CHANGED]
  |     latent_pred = head(backbone(encoder(current), action))
  |     
  |     consistency_loss = MSE(latent_pred, latent_target)
  |     reconstruction_loss = MSE(decoder(latent_pred), next_state)
  |     
  |     if encoder_consistency_coef > 0:              [NEW -- optional]
  |         encoder_consistency_loss = MSE(encoder(next_state), sg(latent_pred))
  |     else:
  |         encoder_consistency_loss = 0
  |     
  |     state_loss = consistency_coef * consistency_loss
  |                + reconstruction_coef * reconstruction_loss
  |                + encoder_consistency_coef * encoder_consistency_loss

compute_loss() aggregates across ensemble -> returns 10 values (was 9)

update_system_dynamics() in mbpo_ppo.py:
  total_loss = w_state * state_loss + w_sequence * seq_loss + ...
             + w_encoder_consistency * encoder_consistency_loss + ...
  optimizer.step()
  system_dynamics.update_target_encoder()    [NEW -- EMA update after each step]
```

---

## Gradient Flow

### Online Encoder receives gradients through:
1. **Initial history encoding** → backbone → head → consistency_loss (dynamics path)
2. **Reconstruction loss** path: latent_pred → decoder → MSE(decoded, raw_target)
3. **Bidirectional encoder consistency** (if enabled): encoder(s_next) → MSE(_, sg(latent_pred))

### Target Encoder:
- **No optimizer gradients** (requires_grad=False)
- Updated ONLY via EMA: `θ_target ← m * θ_target + (1-m) * θ_online`
- Called after each `optimizer.step()` in the training loop

### Dynamics (backbone + heads):
- Gradients from consistency_loss (match EMA target encoder output)
- Gradients from reconstruction_loss (through decoder)
- Gradients from encoder_consistency_loss ONLY through `latent_pred` being used as
  stop-gradient target (no gradient through this path for dynamics)

---

## Configuration Examples

### Minimal (default -- EMA with no bidirectional loss):
```python
system_dynamics_cfg = dict(
    latent_mode=True,
    latent_dim=256,
    target_encoder_momentum=0.99,      # NEW
    encoder_consistency_coef=0.0,       # NEW (disabled)
    consistency_coef=2.0,
    reconstruction_coef=1.0,
)
```

### With bidirectional encoder consistency:
```python
system_dynamics_cfg = dict(
    latent_mode=True,
    latent_dim=256,
    target_encoder_momentum=0.99,
    encoder_consistency_coef=1.0,       # Enabled
    consistency_coef=2.0,
    reconstruction_coef=1.0,
)
```

### Higher momentum (slower target update, more stable):
```python
system_dynamics_cfg = dict(
    latent_mode=True,
    latent_dim=256,
    target_encoder_momentum=0.995,      # Slower update
    encoder_consistency_coef=0.0,
    consistency_coef=2.0,
    reconstruction_coef=1.0,
)
```

---

## Files Changed Summary

| File | Repo | Status | Lines Changed |
|------|------|--------|---------------|
| `rsl_rl/modules/system_dynamics.py` | rsl_rl_rwm | MODIFIED | +import, +params, +target_encoder init, +update method, +compute_state_loss, +compute_loss, +train |
| `rsl_rl/algorithms/mbpo_ppo.py` | rsl_rl_rwm | MODIFIED | +unpack 10 values, +EMA update call, +encoder_consistency tracking |
| `rsl_rl/runners/mbpo_on_policy_runner.py` | rsl_rl_rwm | MODIFIED | +unpack 10 values, +tensorboard logging |
| `source/mbrl/mbrl/rl/rsl_rl/rl_cfg.py` | robotic_world_model | MODIFIED | +target_encoder_momentum, +encoder_consistency_coef |
| `PHASE2_CONSISTENCY_LOSS_CHANGES.md` | rsl_rl_rwm | NEW | This document |
