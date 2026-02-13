"""
Encoder and decoder modules for the latent-space world model.

The encoder maps raw state observations to a latent representation using
NormedLinear layers with SimNorm output (TD-MPC2 style).

The decoder maps latent representations back to raw state space for
reconstruction loss computation and imagination environment compatibility.
"""

import torch
import torch.nn as nn

from rsl_rl.modules.architectures.latent_layers import SimNorm, NormedLinear, latent_mlp


class StateEncoder(nn.Module):
    """Encodes raw state observations into a latent representation.
    
    Architecture: NormedLinear hidden layers (Linear -> LayerNorm -> Mish)
    with SimNorm on the output layer, following TD-MPC2.
    
    The SimNorm output ensures the latent space has structured geometry:
    each group of `simnorm_dim` elements forms a probability simplex.
    
    Args:
        state_dim: Raw state observation dimension.
        latent_dim: Latent representation dimension. Must be divisible by simnorm_dim.
        hidden_dims: List of hidden layer widths (default: [256]).
        simnorm_dim: SimNorm group size (default: 8).
        dropout: Dropout rate for hidden layers (default: 0.0).
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | None = None,
        simnorm_dim: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256]
        
        assert latent_dim % simnorm_dim == 0, (
            f"latent_dim ({latent_dim}) must be divisible by simnorm_dim ({simnorm_dim})"
        )
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.simnorm_dim = simnorm_dim
        
        self.net = latent_mlp(
            in_dim=state_dim,
            hidden_dims=hidden_dims,
            out_dim=latent_dim,
            output_act=SimNorm(simnorm_dim),
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw state to latent representation.
        
        Args:
            x: Raw state tensor of shape [..., state_dim].
            
        Returns:
            Latent representation of shape [..., latent_dim].
        """
        return self.net(x)


class StateDecoder(nn.Module):
    """Decodes latent representation back to raw state space.
    
    Plain MLP (no SimNorm on output) since the target is unconstrained
    raw state values. Uses NormedLinear hidden layers for consistency
    with the encoder.
    
    Args:
        latent_dim: Latent representation dimension.
        state_dim: Raw state observation dimension.
        hidden_dims: List of hidden layer widths (default: [256]).
        dropout: Dropout rate for hidden layers (default: 0.0).
        residual: If True, enables residual prediction mode. The decoder receives
            both the latent prediction and the current raw state (concatenated),
            and predicts a delta that is added to the current state:
            s_{t+1} = s_t + Decoder(concat(z_{t+1}, s_t)). Default: False.
    """
    
    def __init__(
        self,
        latent_dim: int,
        state_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        residual: bool = False,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256]
        
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        self.residual = residual
        
        # In residual mode, input is concat(latent, current_state)
        in_dim = latent_dim + state_dim if residual else latent_dim
        
        self.net = latent_mlp(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=state_dim,
            output_act=None,  # plain Linear output for unconstrained state values
            dropout=dropout,
        )
    
    def forward(self, z: torch.Tensor, current_state: torch.Tensor | None = None) -> torch.Tensor:
        """Decode latent representation to raw state.
        
        Args:
            z: Latent tensor of shape [..., latent_dim].
            current_state: Current raw state tensor of shape [..., state_dim].
                Required when residual=True. Ignored when residual=False.
            
        Returns:
            Reconstructed state of shape [..., state_dim].
            In residual mode: current_state + delta.
            In direct mode: absolute state prediction.
        """
        if self.residual:
            assert current_state is not None, (
                "current_state is required when residual=True"
            )
            x = torch.cat([z, current_state], dim=-1)
            delta = self.net(x)
            return current_state + delta
        return self.net(z)
