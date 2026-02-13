# Residual Decoder for Latent-Space World Model

## Overview

This change adds an optional **residual prediction mode** to the `StateDecoder`. Instead of directly mapping a latent vector to an absolute raw state, the decoder receives both the latent prediction and the current raw state, and predicts a **delta** that is added to the current state.

```
Direct mode (default):   s_{t+1} = Decoder(z_{t+1})
Residual mode:           s_{t+1} = s_t + Decoder(concat(z_{t+1}, s_t))
```

## Motivation

The raw-state dynamics heads (`MLPStateHead`, `MLPStateHeadWithPrior`) predict residuals: `next_state = current_state + delta`. The latent dynamics head (`LatentDynamicsHead`) predicts next latent states directly (with SimNorm), which is correct for the latent space. However, the decoder that maps latent back to raw state space was using direct prediction, creating an asymmetry.

Residual decoding offers:

- **Easier learning**: the network only needs to predict what changed, not reconstruct the full state from scratch.
- **Smaller output magnitudes**: residuals are typically small, leading to better-conditioned gradients.
- **Reduced compounding error**: in long-horizon autoregressive rollouts, small delta errors accumulate more slowly than absolute prediction errors.
- **Consistency**: aligns the decoder's prediction semantics with the raw-state heads.

## Configuration

Enable via the `residual_decoder` parameter on `SystemDynamicsEnsemble`:

```python
dynamics = SystemDynamicsEnsemble(
    state_dim=45,
    action_dim=12,
    # ... other args ...
    latent_mode=True,
    residual_decoder=True,   # NEW: enable residual decoder
)
```

Default is `False` (direct decoder, backward-compatible).

Only has an effect when `latent_mode=True`. In raw-state mode, the decoder is not used.

## Files Changed

### `rsl_rl/modules/architectures/encoder_decoder.py`

- `StateDecoder.__init__()`: Added `residual: bool = False` parameter. When True, the MLP input dimension becomes `latent_dim + state_dim` (concatenation of latent and current state) instead of just `latent_dim`.
- `StateDecoder.forward()`: Added `current_state` parameter. When `residual=True`, concatenates latent and current state, runs through MLP, and returns `current_state + delta`. When `residual=False`, ignores `current_state` (backward-compatible).

### `rsl_rl/modules/system_dynamics.py`

- `SystemDynamicsEnsemble.__init__()`: Added `residual_decoder: bool = False` parameter.
- `_init_networks()`: Passes `residual=self.residual_decoder` to `StateDecoder`.
- `decode()`: Updated signature to accept optional `current_raw_states`, forwarded to `self.decoder()`.
- `forward()`: Both decode call sites (ensemble mean and gather paths) now pass `x_state_batch[:, -1]` (the current raw state from input history) to `decode()`.
- `compute_state_loss()`: Reconstruction loss computation now passes `state_batch[:, self.history_horizon + i - 1]` (the raw state just before the transition target) to `self.decoder()`.

## Data Flow

### Training (`compute_state_loss`)

```
state_batch[:, t-1]  ─────────────────────────────────────────┐
                                                               │
state_batch[:, t]  ──► target_encoder ──► z_target             │  (current raw state)
                                                               │
z_t ──► dynamics_head ──► z_{t+1}_pred ──► decoder(z, s_t) ──► s_t + delta
                                                     │
                                                     ▼
                                            reconstruction_loss = MSE(pred, s_{t+1})
```

### Inference (`forward`)

```
x_state_batch[:, -1]  ────────────────────────────────┐
                                                       │  (current raw state)
x_state_batch ──► encoder ──► base ──► heads ──►       │
                                        │              │
                                  latent_means ──► decode(z, s_t) ──► raw state output
```

## Backward Compatibility

- `residual_decoder=False` (default): behavior is identical to before. The `current_state` argument to `StateDecoder.forward()` is ignored.
- Raw state mode (`latent_mode=False`): decoder is never instantiated, so `residual_decoder` has no effect.
- `forward_latent()`: unchanged, as it operates entirely in latent space and never decodes.
