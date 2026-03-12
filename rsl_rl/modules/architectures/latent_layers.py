"""
Latent-space layers adapted from TD-MPC2 for the RWM hybrid architecture.

Contains:
- SimNorm: Simplicial normalization (softmax over groups)
- NormedLinear: Linear + LayerNorm + activation (Mish by default)
- latent_mlp: MLP builder using NormedLinear layers

Reference: "TD-MPC2: Scalable, Robust World Models for Continuous Control" (Hansen et al., 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithmic transform: sign(x) * ln(1 + |x|).
    
    Compresses large magnitudes while preserving sign. Smooth near zero.
    Used to normalize reward and value targets so their MSE loss stays
    on a comparable scale to latent-space consistency losses.
    
    Reference: TD-MPC2 (Hansen et al., 2024)
    """
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1).
    
    Recovers the original scale from symlog-compressed values.
    """
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


class SimNorm(nn.Module):
    """Simplicial normalization.
    
    Splits the input into groups of `dim` elements and applies softmax
    within each group. This constrains each group to lie on a probability
    simplex (non-negative, sums to 1).
    
    For a latent vector of size L with simnorm_dim=V, produces L/V groups,
    each a V-dimensional probability distribution.
    
    Reference: https://arxiv.org/abs/2204.00616
    
    Args:
        simnorm_dim: Size of each simplex group (default 8).
    """
    
    def __init__(self, simnorm_dim: int = 8):
        super().__init__()
        self.dim = simnorm_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)
    
    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """Linear layer with LayerNorm and activation.
    
    Order: Linear -> (Dropout) -> LayerNorm -> Activation
    
    Default activation is Mish. Can be overridden (e.g., to SimNorm for
    encoder/dynamics output layers).
    
    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        act: Activation module (default: nn.Mish).
        dropout: Dropout rate (default: 0.0).
    """
    
    def __init__(self, in_features: int, out_features: int, act=None, dropout: float = 0.0):
        super().__init__(in_features, out_features, bias=True)
        self.ln = nn.LayerNorm(out_features)
        self.act = act if act is not None else nn.Mish(inplace=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.act(self.ln(x))
    
    def __repr__(self):
        return (
            f"NormedLinear(in={self.in_features}, out={self.out_features}, "
            f"act={self.act.__class__.__name__})"
        )


def latent_mlp(
    in_dim: int,
    hidden_dims: list[int],
    out_dim: int,
    output_act: nn.Module | None = None,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Build an MLP using NormedLinear hidden layers.
    
    Hidden layers use NormedLinear (Linear + LayerNorm + Mish).
    Output layer uses NormedLinear with custom activation if output_act is
    provided, otherwise uses plain nn.Linear (no norm, no activation).
    
    Args:
        in_dim: Input dimension.
        hidden_dims: List of hidden layer widths.
        out_dim: Output dimension.
        output_act: Activation for the output layer (e.g., SimNorm). 
                    None means plain Linear output.
        dropout: Dropout rate for hidden layers.
    
    Returns:
        nn.Sequential of layers.
    """
    dims = [in_dim] + hidden_dims + [out_dim]
    layers = []
    
    # Hidden layers: NormedLinear with Mish
    for i in range(len(dims) - 2):
        layers.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout))
    
    # Output layer
    if output_act is not None:
        layers.append(NormedLinear(dims[-2], dims[-1], act=output_act))
    else:
        layers.append(nn.Linear(dims[-2], dims[-1]))
    
    return nn.Sequential(*layers)
