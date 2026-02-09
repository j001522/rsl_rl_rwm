import torch
import torch.nn as nn
import copy

from rsl_rl.modules.architectures.latent_layers import SimNorm, NormedLinear, latent_mlp


class MLPBase(nn.Module):
    def __init__(
        self,
        input_dim: int,
        device: str,
        architecture_config: dict = None,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        
        base_shape = architecture_config["base_shape"]
        layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in base_shape:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.layers = nn.Sequential(*layers).to(self.device)
        self.layers.train()
        
    def forward(self, x_state_batch, x_action_batch):
        x = torch.cat([x_state_batch, x_action_batch], dim=-1).flatten(1, 2)
        x = self.layers(x)
        return x
    
    def reset(self):
        pass


class MLPStateHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        device: str,
        architecture_config: dict = None,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.device = device
        self.state_mean_shape = architecture_config["state_mean_shape"]
        self.state_logstd_shape = architecture_config["state_logstd_shape"]

        state_mean_layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in self.state_mean_shape:
            state_mean_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            state_mean_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        state_mean_layers.append(nn.Linear(self.state_mean_shape[-1], state_dim))
        self.state_mean_layers = nn.Sequential(*state_mean_layers).to(self.device)
        self.state_mean_layers.train()

        if self.state_logstd_shape is not None:
            self.output_std = True
            state_logstd_layers = []
            curr_in_dim = self.input_dim
            for hidden_dim in self.state_logstd_shape:
                state_logstd_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                state_logstd_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            state_logstd_layers.append(nn.Linear(self.state_logstd_shape[-1], state_dim))
            self.state_logstd_layers = nn.Sequential(*state_logstd_layers).to(self.device)
            self.state_logstd_layers.train()
        else:
            self.output_std = False

        if self.output_std:
            self.state_min_logstd = nn.Parameter(torch.ones(1, state_dim, device=self.device) * -5.0)
            self.state_log_delta_logstd = nn.Parameter(torch.ones(1, state_dim, device=self.device) * 0.0)

    def forward(self, x, x_state_batch):
        if x.dim() == 3:
            sequence_len = x.shape[1]
            x = x.flatten(0, 1)
            x_state_batch = x_state_batch.flatten(0, 1).unsqueeze(1)
        else:
            sequence_len = 0
        state_mean = self.state_mean_layers(x) + x_state_batch[:, -1]
        state_logstd = self.state_logstd_layers(x) if self.output_std else -torch.inf * torch.ones(x.shape[0], self.state_dim, device=self.device)
        if self.output_std:
            self.state_max_logstd = self.state_min_logstd + torch.exp(self.state_log_delta_logstd)
            state_logstd = self.state_max_logstd - nn.functional.softplus(self.state_max_logstd - state_logstd)
            state_logstd = self.state_min_logstd + nn.functional.softplus(state_logstd - self.state_min_logstd)
        if sequence_len > 0:
            state_mean = state_mean.view(-1, sequence_len, self.state_dim)
            state_logstd = state_logstd.view(-1, sequence_len, self.state_dim)
        return state_mean, torch.exp(state_logstd)

    def reset(self):
        pass


class LatentDynamicsHead(nn.Module):
    """Dynamics head that predicts next latent state with SimNorm output.
    
    Unlike MLPStateHead which predicts raw state deltas with a residual
    connection (next = current + delta), this head directly predicts the
    next latent state and applies SimNorm to keep it on the simplex.
    
    Supports randomized priors (Osband et al.) for ensemble diversity,
    following the same pattern as DynamicsHeadWithPrior in TD-MPC2:
    the prior perturbation is added BEFORE SimNorm, so both learned
    prediction and diversity noise are jointly normalized.
    
    This head does NOT predict uncertainty (std). Uncertainty in latent
    mode comes purely from ensemble disagreement (epistemic uncertainty).
    
    Args:
        input_dim: Input dimension (from base network output).
        latent_dim: Latent state dimension to predict.
        device: Device to place tensors on.
        architecture_config: Dict with 'latent_head_hidden_dims' key.
        simnorm_dim: SimNorm group size (default: 8).
        prior_scale: Scale for randomized prior output (0 = disabled).
        prior_hidden_div: Divisor for prior hidden dimensions (default: 4).
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        device: str,
        architecture_config: dict = None,
        simnorm_dim: int = 8,
        prior_scale: float = 0.0,
        prior_hidden_div: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        self.prior_scale = prior_scale
        self.prior_hidden_div = prior_hidden_div
        self.output_std = False  # Latent heads don't predict std
        
        hidden_dims = architecture_config.get("latent_head_hidden_dims", [256])
        
        # Main trainable MLP (no output activation -- SimNorm applied after prior)
        self.main_mlp = latent_mlp(
            in_dim=input_dim,
            hidden_dims=hidden_dims,
            out_dim=latent_dim,
            output_act=None,
        ).to(self.device)
        
        # Random prior network (frozen output, detached gradients)
        if self.prior_scale > 0:
            prior_hidden_dims = [max(h // prior_hidden_div, 8) for h in hidden_dims]
            prior_layers = []
            curr_in_dim = input_dim
            for hidden_dim in prior_hidden_dims:
                prior_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                prior_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            prior_layers.append(nn.Linear(prior_hidden_dims[-1], latent_dim))
            self.prior_mlp = nn.Sequential(*prior_layers).to(self.device)
            
            # Xavier initialization for prior
            for m in self.prior_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
        else:
            self.prior_mlp = None
        
        # SimNorm applied after main + prior
        self.simnorm = SimNorm(simnorm_dim).to(self.device)
    
    def forward(self, x, x_state_batch=None):
        """Predict next latent state.
        
        Args:
            x: Base network output, shape [batch, base_dim] or [batch, seq, base_dim].
            x_state_batch: Ignored (kept for API compatibility with MLPStateHead).
                          Latent dynamics has no residual connection.
        
        Returns:
            (latent_mean, latent_std): Tuple where latent_mean has shape
                [batch, latent_dim] and latent_std is zeros (no aleatoric
                uncertainty in latent space).
        """
        # Handle sequence dimension
        if x.dim() == 3:
            sequence_len = x.shape[1]
            x = x.flatten(0, 1)
        else:
            sequence_len = 0
        
        # Main prediction
        out = self.main_mlp(x)
        
        # Add prior perturbation
        if self.prior_mlp is not None:
            with torch.no_grad():
                prior_out = self.prior_mlp(x)
            prior_out = prior_out.detach()
            out = out + prior_out * self.prior_scale
        
        # Apply SimNorm (after prior, so both are jointly normalized)
        latent_mean = self.simnorm(out)
        
        # No aleatoric uncertainty in latent space
        latent_std = torch.zeros_like(latent_mean)
        
        # Reshape for sequence output
        if sequence_len > 0:
            latent_mean = latent_mean.view(-1, sequence_len, self.latent_dim)
            latent_std = latent_std.view(-1, sequence_len, self.latent_dim)
        
        return latent_mean, latent_std
    
    def reset(self):
        pass



class MLPStateHeadWithPrior(nn.Module):
    """
    MLP State Head with randomized prior for ensemble diversity.
    
    Based on "Randomized Prior Functions for Deep Reinforcement Learning" (Osband et al.).
    The prior network is a frozen copy of the mean network with Xavier initialization.
    Its output is detached (no gradient flow) and added to the trainable mean output.
    
    This encourages diverse predictions across ensemble members, preventing collapse
    to identical predictions in sparse reward or low-data regimes.
    
    Key design decisions:
    - Prior only applies to MEANS, not variances (std). This preserves the learned
      uncertainty estimates while diversifying point predictions.
    - Prior is added BEFORE the residual state connection (next = delta + current),
      so it perturbs the predicted delta, not the absolute state.
    - Prior network uses smaller hidden dims (hidden_dim // prior_hidden_div) to
      reduce compute while maintaining diversity.
    
    Args:
        input_dim: Input dimension (from base network output).
        state_dim: State dimension to predict.
        device: Device to place tensors on.
        architecture_config: Dict with 'state_mean_shape' and 'state_logstd_shape'.
        prior_scale: Scale factor for prior output. 0 = no prior (same as MLPStateHead).
        prior_hidden_div: Divisor for prior hidden dimensions (default 4).
    """
    
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        device: str,
        architecture_config: dict = None,
        prior_scale: float = 1.0,
        prior_hidden_div: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.device = device
        self.prior_scale = prior_scale
        self.prior_hidden_div = prior_hidden_div
        
        self.state_mean_shape = architecture_config["state_mean_shape"]
        self.state_logstd_shape = architecture_config["state_logstd_shape"]

        # --- 1. Main Trainable Mean Network ---
        state_mean_layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in self.state_mean_shape:
            state_mean_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            state_mean_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        state_mean_layers.append(nn.Linear(self.state_mean_shape[-1], state_dim))
        self.state_mean_layers = nn.Sequential(*state_mean_layers).to(self.device)
        self.state_mean_layers.train()

        # --- 2. Random Prior Network for Mean (frozen, detached output) ---
        if self.prior_scale > 0:
            # Use smaller hidden dims for prior (reduces compute, maintains diversity)
            prior_hidden_dims = [max(h // prior_hidden_div, 8) for h in self.state_mean_shape]
            prior_layers = []
            curr_in_dim = self.input_dim
            for hidden_dim in prior_hidden_dims:
                prior_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                prior_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            prior_layers.append(nn.Linear(prior_hidden_dims[-1], state_dim))
            self.prior_mean_layers = nn.Sequential(*prior_layers).to(self.device)
            
            # Initialize with Xavier/Glorot (as per Osband et al.)
            for m in self.prior_mean_layers.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
            
            # Note: We don't set requires_grad=False to avoid issues with torch.compile.
            # Instead, we detach the output in forward() to block gradient flow.
        else:
            self.prior_mean_layers = None

        # --- 3. LogStd Network (unchanged, no prior) ---
        if self.state_logstd_shape is not None:
            self.output_std = True
            state_logstd_layers = []
            curr_in_dim = self.input_dim
            for hidden_dim in self.state_logstd_shape:
                state_logstd_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                state_logstd_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            state_logstd_layers.append(nn.Linear(self.state_logstd_shape[-1], state_dim))
            self.state_logstd_layers = nn.Sequential(*state_logstd_layers).to(self.device)
            self.state_logstd_layers.train()
        else:
            self.output_std = False

        if self.output_std:
            self.state_min_logstd = nn.Parameter(torch.ones(1, state_dim, device=self.device) * -5.0)
            self.state_log_delta_logstd = nn.Parameter(torch.ones(1, state_dim, device=self.device) * 0.0)

    def forward(self, x, x_state_batch):
        # Handle sequence dimension
        if x.dim() == 3:
            sequence_len = x.shape[1]
            x = x.flatten(0, 1)
            x_state_batch = x_state_batch.flatten(0, 1).unsqueeze(1)
        else:
            sequence_len = 0

        # --- Forward Pass: Mean with Prior ---
        # 1. Compute trainable mean output (delta)
        trainable_delta = self.state_mean_layers(x)

        # 2. Add prior perturbation (if active)
        if self.prior_mean_layers is not None:
            with torch.no_grad():
                prior_out = self.prior_mean_layers(x)
            # Detach to ensure no gradient flow to prior params
            prior_out = prior_out.detach()
            # Add scaled prior to trainable delta
            trainable_delta = trainable_delta + prior_out * self.prior_scale

        # 3. Add residual connection: next_state = current_state + delta
        state_mean = trainable_delta + x_state_batch[:, -1]

        # --- Forward Pass: Std (unchanged, no prior) ---
        if self.output_std:
            state_logstd = self.state_logstd_layers(x)
            self.state_max_logstd = self.state_min_logstd + torch.exp(self.state_log_delta_logstd)
            state_logstd = self.state_max_logstd - nn.functional.softplus(self.state_max_logstd - state_logstd)
            state_logstd = self.state_min_logstd + nn.functional.softplus(state_logstd - self.state_min_logstd)
        else:
            state_logstd = -torch.inf * torch.ones(x.shape[0], self.state_dim, device=self.device)
        
        # Reshape for sequence output
        if sequence_len > 0:
            state_mean = state_mean.view(-1, sequence_len, self.state_dim)
            state_logstd = state_logstd.view(-1, sequence_len, self.state_dim)
            
        return state_mean, torch.exp(state_logstd)

    def reset(self):
        pass


class MLPAuxiliaryHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        extension_dim: int,
        contact_dim: int,
        termination_dim: int,
        device: str,
        architecture_config: dict = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.extension_dim = extension_dim
        self.contact_dim = contact_dim
        self.termination_dim = termination_dim
        self.device = device

        if extension_dim > 0:
            extension_shape = architecture_config["extension_shape"]
            extension_layers = []
            curr_in_dim = self.input_dim
            for hidden_dim in extension_shape:
                extension_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                extension_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            extension_layers.append(nn.Linear(extension_shape[-1], extension_dim))
            self.extension_layers = nn.Sequential(*extension_layers).to(self.device)
            self.extension_layers.train()

        if contact_dim > 0:
            contact_shape = architecture_config["contact_shape"]
            contact_layers = []
            curr_in_dim = self.input_dim
            for hidden_dim in contact_shape:
                contact_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                contact_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            contact_layers.append(nn.Linear(contact_shape[-1], contact_dim))
            self.contact_layers = nn.Sequential(*contact_layers).to(self.device)
            self.contact_layers.train()
        
        if termination_dim > 0:
            termination_shape = architecture_config["termination_shape"]
            termination_layers = []
            curr_in_dim = self.input_dim
            for hidden_dim in termination_shape:
                termination_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                termination_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            termination_layers.append(nn.Linear(termination_shape[-1], termination_dim))
            self.termination_layers = nn.Sequential(*termination_layers).to(self.device)
            self.termination_layers.train()

    def forward(self, x, x_state_batch):
        if x.dim() == 3:
            sequence_len = x.shape[1]
            x = x.flatten(0, 1)
        else:
            sequence_len = 0
        
        extension_pred = self.extension_layers(x) if self.extension_dim > 0 else None
        contact_logits = self.contact_layers(x) if self.contact_dim > 0 else None
        termination_logits = self.termination_layers(x) if self.termination_dim > 0 else None
        
        if sequence_len > 0:
            extension_pred = extension_pred.view(-1, sequence_len, self.extension_dim) if self.extension_dim > 0 else None
            contact_logits = contact_logits.view(-1, sequence_len, self.contact_dim) if self.contact_dim > 0 else None
            termination_logits = termination_logits.view(-1, sequence_len, self.termination_dim) if self.termination_dim > 0 else None
        
        return extension_pred, contact_logits, termination_logits

    def reset(self):
        pass
