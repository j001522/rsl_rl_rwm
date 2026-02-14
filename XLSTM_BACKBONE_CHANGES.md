# xLSTM Backbone Architecture — Changes Documentation

## Overview

This branch (`feature/xlstm-backbone`) adds **xLSTM** (Extended Long Short-Term Memory) as a new backbone architecture option for the `SystemDynamicsEnsemble`. The xLSTM architecture from [NX-AI/xlstm](https://github.com/NX-AI/xlstm) provides two cell types:

- **mLSTM (Matrix Memory)**: Uses a matrix-valued memory and covariance update rule with exponential gating. Fully parallelizable during training. No custom CUDA compilation needed.
- **sLSTM (Scalar Memory)**: Traditional scalar memory with exponential gating and multiple memory states. Includes a gated feedforward sub-block. Optionally uses a custom CUDA kernel (requires compute capability >= 8.0).

### Why xLSTM for Robotic World Models?

The xLSTM backbone is primarily designed for **latent-space dynamics** (256-dim latent states), where its matrix memory and multi-head attention-like mechanism can capture complex temporal dependencies more effectively than standard GRU/LSTM. For raw-state mode (57-dim), the standard GRU backbone is likely sufficient and more parameter-efficient.

Key advantages:
- **Matrix memory** (mLSTM): Can store and retrieve high-dimensional patterns in latent space
- **Parallelizable training**: mLSTM blocks use parallel scan for full-sequence processing
- **Exponential gating**: Better gradient flow for long sequences (history_horizon=32)
- **Modular block stacking**: Can mix mLSTM and sLSTM blocks at different positions

## Files Modified

### New Files

1. **`rsl_rl/modules/architectures/xlstm_base.py`** — `xLSTMBase` class
   - Wraps `xLSTMBlockStack` from the `xlstm` library
   - Follows the same interface as `RNNBase`: `forward(x_state_batch, x_action_batch) → [batch, embedding_dim]`
   - Input projection: `Linear(state_dim + action_dim, embedding_dim)`
   - Uses `step()` mode for single-timestep autoregressive processing (maintains recurrent state)
   - Uses `forward()` mode for multi-timestep sequences (parallel processing, then initializes step state)
   - `reset()`: Full state reset (sets internal state to `None`)
   - `reset_partial(env_ids)`: Zeroes out state for specific batch indices (supports vectorized envs)
   - Graceful import: `XLSTM_AVAILABLE` flag if library not installed

2. **`robotic_world_model/slurm/configs/pretrain_xlstm_latent.yaml`** — Pretraining config for xLSTM + latent mode

### Modified Files

3. **`rsl_rl/modules/architectures/__init__.py`** — Added export of `xLSTMBase` and `XLSTM_AVAILABLE`

4. **`rsl_rl/modules/system_dynamics.py`**
   - Added import of `xLSTMBase` and `XLSTM_AVAILABLE`
   - Added `elif self.architecture_config["type"] == "xlstm":` case in `_create_base()` (lines ~234-255)
   - Updated all 6 occurrences of `["rnn", "rssm"]` → `["rnn", "rssm", "xlstm"]` for autoregressive handling in:
     - `compute_state_loss()` — action slicing and autoregressive state feeding
     - `compute_auxiliary_loss()` — action slicing and state feeding
   - Output dimension set from `architecture_config.get("xlstm_embedding_dim", 256)`

5. **`rsl_rl/runners/mbpo_on_policy_runner.py`**
   - Updated `["rnn", "rssm"]` → `["rnn", "rssm", "xlstm"]` in imagination rollout (line ~190)

6. **`robotic_world_model/slurm/slurm_train.sh`**
   - Added `add_arch_override` helper function for architecture config keys
   - Added override mappings for architecture type and all xLSTM-specific parameters
   - Added special handling for `xlstm_slstm_at` (list type)
   - Added override mappings for head shape parameters (shared across backbones)

## Architecture Config Parameters

### xLSTM-Specific Keys (in `architecture_config` dict)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | str | — | Must be `"xlstm"` to use this backbone |
| `xlstm_embedding_dim` | int | 256 | Internal embedding dimension (= output dim of backbone) |
| `xlstm_num_blocks` | int | 4 | Number of xLSTM blocks in the stack |
| `xlstm_num_heads` | int | 4 | Number of heads for mLSTM/sLSTM cells |
| `xlstm_context_length` | int | 64 | Max sequence length (allocates causal mask buffer) |
| `xlstm_conv1d_kernel_size` | int | 4 | Causal 1D convolution kernel size |
| `xlstm_slstm_at` | list[int] | [] | Block indices that use sLSTM (rest use mLSTM) |
| `xlstm_slstm_backend` | str | "vanilla" | sLSTM backend: "vanilla" (any GPU) or "cuda" (CC >= 8.0) |
| `xlstm_proj_factor_mlstm` | float | 2.0 | mLSTM up-projection factor |
| `xlstm_proj_factor_slstm_ff` | float | 1.3 | sLSTM feedforward projection factor |
| `xlstm_dropout` | float | 0.0 | Dropout rate |

### Example Configuration (Python)

```python
architecture_config = {
    "type": "xlstm",
    "xlstm_embedding_dim": 256,
    "xlstm_num_blocks": 4,
    "xlstm_num_heads": 4,
    "xlstm_context_length": 64,
    "xlstm_conv1d_kernel_size": 4,
    "xlstm_slstm_at": [],           # All mLSTM (no CUDA requirement)
    "xlstm_slstm_backend": "vanilla",
    "xlstm_proj_factor_mlstm": 2.0,
    "xlstm_proj_factor_slstm_ff": 1.3,
    "xlstm_dropout": 0.0,
    # Head shapes (same as RNN/MLP)
    "state_mean_shape": [128],
    "state_logstd_shape": [128],
    "extension_shape": [128],
    "contact_shape": [128],
    "termination_shape": [128],
}
```

## State Management

### How It Differs from RNNBase

| Aspect | RNNBase (GRU/LSTM) | xLSTMBase |
|--------|-------------------|-----------|
| State storage | `self.memory.hidden_states` (tensor) | `self._state` (nested dict) |
| Full reset | Set to `None` | Set to `None` |
| Partial reset | `hidden_states[:, env_ids, :] = 0` | Zero out per-block state tensors |
| Full-seq mode | `rnn(x, hidden_states)` | `xlstm_stack(x)` (stateless, parallel) |
| Step mode | Implicit (RNN processes seq) | `xlstm_stack.step(x, state)` (explicit) |
| Output dim | `rnn_hidden_size` (configurable) | `embedding_dim` (configurable) |

### Internal State Structure

When using `step()` mode, the state is a nested dict:
```
{
    "block_0": {
        "mlstm_state": (c_state, n_state, m_state),  # for mLSTM blocks
        "conv_state": (conv_buffer,)
    },
    "block_1": {
        "slstm_state": tensor,  # for sLSTM blocks (num_states, B, hidden)
        "conv_state": (conv_buffer,)
    },
    ...
}
```

## Requirements

- **Python package**: `pip install xlstm` (NX-AI/xlstm library)
- **GPU**: Any CUDA GPU for mLSTM-only configs
- **GPU (sLSTM with CUDA backend)**: Compute capability >= 8.0 (A100, A6000, RTX 3090+)
- **Graceful degradation**: If `xlstm` is not installed, the code imports cleanly (`XLSTM_AVAILABLE=False`) and raises a clear error only when `type="xlstm"` is requested.

## Backward Compatibility

- All changes are additive — no existing behavior is modified
- Default backbone remains `"rnn"` (GRU) as defined in the existing Python config files
- The xLSTM backbone is only activated when `architecture_config["type"]` is explicitly set to `"xlstm"`
- The `xlstm` library is an optional dependency (lazy error on use, not on import)

## SLURM Usage

```bash
# Pretrain with xLSTM + latent mode
sbatch --job-name=pretrain-xlstm --partition=gpu_a100 --time=24:00:00 \
       slurm_train.sh configs/pretrain_xlstm_latent.yaml
```

## Design Decisions

1. **mLSTM-only by default**: The default `xlstm_slstm_at=[]` means all blocks are mLSTM. This avoids CUDA compilation requirements and provides the most relevant capability (matrix memory for high-dimensional latent states).

2. **Step mode for autoregressive training**: The training loop in `compute_state_loss` processes timesteps one at a time autoregressively. We use `step()` mode for this, maintaining state across calls. For the initial history window (multi-timestep), we use `forward()` for efficiency, then initialize step state.

3. **Input projection**: Since xLSTM requires `embedding_dim` to match input and output, we use a linear projection from `(state_dim + action_dim)` to `embedding_dim`. This differs from RNN which can have different input and hidden sizes natively.

4. **`context_length` parameter**: Set to 64 by default (sufficient for history_horizon=32 + forecast_horizon=8). Allocates a `(context_length, context_length)` causal mask buffer in each mLSTM cell — memory scales quadratically, so don't set excessively large.
