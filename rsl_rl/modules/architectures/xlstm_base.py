"""xLSTM backbone for SystemDynamicsEnsemble.

Uses NX-AI/xlstm library (pip install xlstm) to provide an xLSTM-based
sequence model backbone, following the same interface as RNNBase.

The xLSTM architecture combines:
- mLSTM (matrix memory): parallelizable via matrix memory and covariance update rule,
  suited for high-dimensional latent representations
- sLSTM (scalar memory): traditional scalar memory with exponential gating,
  suited for state tracking

Key differences from RNNBase:
- xLSTM preserves embedding dimension (input_dim == output_dim)
- Uses step() mode for autoregressive processing (state explicitly managed)
- Supports partial state reset for vectorized environments
- mLSTM blocks are the default (no CUDA compilation needed)
"""

import torch
import torch.nn as nn

try:
    from xlstm import (
        xLSTMBlockStack,
        xLSTMBlockStackConfig,
        mLSTMBlockConfig,
        mLSTMLayerConfig,
        sLSTMBlockConfig,
        sLSTMLayerConfig,
        FeedForwardConfig,
    )
    XLSTM_AVAILABLE = True
except ImportError:
    XLSTM_AVAILABLE = False


class xLSTMBase(nn.Module):
    """xLSTM backbone following the same interface as RNNBase.
    
    Processes (state, action) pairs through an xLSTM block stack and returns
    the output at the last timestep (matching RNNBase behavior).
    
    Architecture:
        input_proj: Linear(state_dim + action_dim, embedding_dim)
        xlstm_stack: xLSTMBlockStack(embedding_dim -> embedding_dim)
    
    The output dimension equals embedding_dim (set via architecture_config["xlstm_embedding_dim"]).
    
    Args:
        input_dim: Combined dimension of (state + action).
        device: Device string (e.g. "cuda:0").
        architecture_config: Dict with xLSTM-specific configuration:
            - xlstm_embedding_dim (int): Internal embedding dimension. Default: 256.
            - xlstm_num_blocks (int): Number of xLSTM blocks. Default: 4.
            - xlstm_num_heads (int): Number of heads for mLSTM/sLSTM. Default: 4.
            - xlstm_context_length (int): Max sequence length for causal mask. Default: 64.
            - xlstm_conv1d_kernel_size (int): Causal conv1d kernel size. Default: 4.
            - xlstm_slstm_at (list[int]): Block indices that use sLSTM instead of mLSTM.
                Default: [] (all mLSTM, no CUDA compilation needed).
            - xlstm_slstm_backend (str): "vanilla" or "cuda". Default: "vanilla".
            - xlstm_proj_factor_mlstm (float): mLSTM up-projection factor. Default: 2.0.
            - xlstm_proj_factor_slstm_ff (float): sLSTM feedforward projection factor. Default: 1.3.
            - xlstm_dropout (float): Dropout rate. Default: 0.0.
    """
    
    def __init__(
        self,
        input_dim: int,
        device: str,
        architecture_config: dict = None,
    ):
        super().__init__()
        
        if not XLSTM_AVAILABLE:
            raise ImportError(
                "xlstm library not found. Install it with: pip install xlstm\n"
                "See https://github.com/NX-AI/xlstm for details."
            )
        
        self.input_dim = input_dim
        self.device = device
        
        # Parse config with defaults
        cfg = architecture_config or {}
        self.embedding_dim = cfg.get("xlstm_embedding_dim", 256)
        num_blocks = cfg.get("xlstm_num_blocks", 4)
        num_heads = cfg.get("xlstm_num_heads", 4)
        context_length = cfg.get("xlstm_context_length", 64)
        conv1d_kernel_size = cfg.get("xlstm_conv1d_kernel_size", 4)
        slstm_at = cfg.get("xlstm_slstm_at", [])
        slstm_backend = cfg.get("xlstm_slstm_backend", "vanilla")
        proj_factor_mlstm = cfg.get("xlstm_proj_factor_mlstm", 2.0)
        proj_factor_slstm_ff = cfg.get("xlstm_proj_factor_slstm_ff", 1.3)
        dropout = cfg.get("xlstm_dropout", 0.0)
        
        # Input projection: (state_dim + action_dim) -> embedding_dim
        self.input_proj = nn.Linear(input_dim, self.embedding_dim, device=device)
        
        # Build xLSTM config
        # mLSTM block template (used for blocks not in slstm_at)
        mlstm_block = mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=conv1d_kernel_size,
                qkv_proj_blocksize=4,
                num_heads=num_heads,
                proj_factor=proj_factor_mlstm,
            )
        )
        
        # sLSTM block template (used for blocks in slstm_at)
        slstm_block = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend=slstm_backend,
                num_heads=num_heads,
                conv1d_kernel_size=conv1d_kernel_size,
            ),
            feedforward=FeedForwardConfig(
                proj_factor=proj_factor_slstm_ff,
                act_fn="gelu",
            ),
        )
        
        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=mlstm_block,
            slstm_block=slstm_block,
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=self.embedding_dim,
            add_post_blocks_norm=True,
            bias=False,
            dropout=dropout,
            slstm_at=slstm_at if slstm_at else [],
        )
        
        self.xlstm_stack = xLSTMBlockStack(xlstm_config).to(device)
        
        # Internal state for step-by-step processing
        # State structure: dict of block states, managed by xLSTMBlockStack.step()
        self._state = None
    
    def forward(self, x_state_batch: torch.Tensor, x_action_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass through the xLSTM backbone.
        
        Processes the full (state, action) sequence and returns the output
        at the last timestep, matching RNNBase behavior.
        
        When sequence length is 1 (single timestep, as in autoregressive training),
        uses step() mode to maintain recurrent state across calls.
        When sequence length > 1 (initial history), uses full-sequence forward()
        for efficient parallel processing, then switches to step mode.
        
        Args:
            x_state_batch: State tensor [batch, seq_len, state_dim].
            x_action_batch: Action tensor [batch, seq_len, action_dim].
            
        Returns:
            Output tensor [batch, embedding_dim] (last timestep output).
        """
        # Concatenate state and action
        x = torch.cat([x_state_batch, x_action_batch], dim=-1)  # [B, S, input_dim]
        
        # Project to embedding dimension
        x = self.input_proj(x)  # [B, S, embedding_dim]
        
        seq_len = x.shape[1]
        
        if seq_len == 1:
            # Single timestep: use step() mode to maintain recurrent state
            out, self._state = self.xlstm_stack.step(x, self._state)
            return out[:, 0]  # [B, embedding_dim]
        else:
            # Full sequence: use parallel forward for efficiency
            # This is used for the initial history window
            # After this, subsequent calls with seq_len=1 will use step() mode
            #
            # We process the full sequence with forward(), then initialize
            # step() state from the last position for future autoregressive calls.
            out = self.xlstm_stack(x)  # [B, S, embedding_dim]
            
            # Initialize step state by running through step() mode
            # This is needed so that subsequent step() calls have correct state
            self._state = None
            for t in range(seq_len):
                _, self._state = self.xlstm_stack.step(x[:, t:t+1, :], self._state)
            
            return out[:, -1]  # [B, embedding_dim]
    
    def reset(self):
        """Reset all hidden states (full reset for all environments)."""
        self._state = None
    
    def reset_partial(self, env_ids):
        """Reset hidden states for specific environment indices.
        
        Zeroes out the recurrent state for the given batch indices while
        preserving state for all other indices.
        
        Args:
            env_ids: Tensor or list of environment indices to reset.
        """
        if self._state is None:
            return
        
        for block_key in self._state:
            block_state = self._state[block_key]
            
            # Reset mLSTM state: tuple of (c_state, n_state, m_state)
            # c_state: [B, NH, DH, DH], n_state: [B, NH, DH, 1], m_state: [B, NH, 1, 1]
            if "mlstm_state" in block_state:
                for tensor in block_state["mlstm_state"]:
                    tensor[env_ids] = 0
            
            # Reset sLSTM state: single tensor [num_states, B, hidden_size]
            if "slstm_state" in block_state:
                block_state["slstm_state"][:, env_ids] = 0
            
            # Reset conv state: tuple of tensors
            if "conv_state" in block_state:
                for tensor in block_state["conv_state"]:
                    tensor[env_ids] = 0
