# RSL_RL_RWM: Comprehensive Technical Reference

**Version**: 3.1.0  
**Repository**: https://github.com/leggedrobotics/rsl_rl_rwm  
**Your Fork**: https://github.com/j001522/rsl_rl_rwm  
**Analysis Date**: January 24, 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Core World Model Architecture](#core-world-model-architecture)
4. [Algorithm Implementation](#algorithm-implementation)
5. [Training Runner](#training-runner)
6. [Data Storage Components](#data-storage-components)
7. [Critical Interfaces](#critical-interfaces)
8. [Configuration System](#configuration-system)
9. [Extension Points](#extension-points)

---

## Overview

RSL_RL_RWM is a PyTorch-based library implementing **Robotic World Model (RWM)** and **Uncertainty-Aware RWM (U-RWM)** for model-based reinforcement learning. It extends the base RSL_RL library with:

- **Ensemble-based world models** with uncertainty quantification
- **MBPO-PPO algorithm** (Model-Based Policy Optimization with PPO)
- **Imagination rollouts** for sample-efficient learning
- **GPU-accelerated training** with vectorized environments

**Key Papers**:
- [Robotic World Model (2025)](https://arxiv.org/abs/2501.10100)
- [Uncertainty-Aware RWM (2025)](https://arxiv.org/abs/2504.16680)

---

## Repository Structure

```
rsl_rl_rwm/
├── rsl_rl/
│   ├── modules/              # Neural network modules
│   │   ├── system_dynamics.py          # ⭐ WORLD MODEL CORE
│   │   ├── architectures/
│   │   │   ├── mlp.py                  # MLP base and heads
│   │   │   └── rnn.py                  # RNN base and memory
│   │   ├── actor_critic.py             # Policy network (MLP)
│   │   ├── actor_critic_recurrent.py   # Policy network (RNN)
│   │   └── plotter.py                  # Visualization utilities
│   │
│   ├── algorithms/           # Training algorithms
│   │   ├── ppo.py                      # Base PPO
│   │   ├── mbpo_ppo.py                 # ⭐ MBPO-PPO (model-based)
│   │   └── distillation.py             # Policy distillation
│   │
│   ├── runners/              # Training loops
│   │   ├── on_policy_runner.py         # Base runner
│   │   └── mbpo_on_policy_runner.py    # ⭐ Model-based runner
│   │
│   ├── storage/              # Data management
│   │   ├── replay_buffer.py            # ⭐ For world model training
│   │   └── rollout_storage.py          # For PPO training
│   │
│   ├── networks/             # Network utilities
│   │   ├── mlp.py
│   │   ├── memory.py
│   │   └── normalization.py
│   │
│   ├── env/                  # Environment wrappers
│   │   └── vec_env.py
│   │
│   └── utils/                # Utilities
│       ├── utils.py
│       └── wandb_utils.py
│
├── config/
│   └── example_config.yaml   # Configuration template
│
├── pyproject.toml            # Package metadata
└── README.md
```

---

## Core World Model Architecture

### File: `rsl_rl/modules/system_dynamics.py`

The **SystemDynamicsEnsemble** is the heart of RWM.

#### Class Hierarchy

```
SystemDynamicsEnsemble (nn.Module)
├── State Prediction Branch
│   ├── state_base: RNNBase | MLPBase (shared encoder)
│   └── state_heads: ModuleList[MLPStateHead] (N ensemble members)
│
└── Auxiliary Prediction Branch
    ├── auxiliary_base: RNNBase | MLPBase (separate encoder)
    └── auxiliary_heads: ModuleList[MLPAuxiliaryHead] (N ensemble members)
```

#### Key Attributes

```python
class SystemDynamicsEnsemble:
    # Dimensions
    state_dim: int              # Robot state dimension (e.g., 45)
    action_dim: int             # Action dimension (e.g., 12)
    extension_dim: int          # Optional extension outputs
    contact_dim: int            # Contact prediction dimension (e.g., 8)
    termination_dim: int        # Termination prediction dimension (e.g., 1)
    
    # Architecture
    ensemble_size: int          # Number of ensemble members (1-5)
    history_horizon: int        # History length for prediction (e.g., 10)
    architecture_config: dict   # {"type": "rnn"|"mlp", ...}
    
    # Training
    freeze_auxiliary: bool      # Whether to freeze auxiliary branch
```

#### Architecture Types

**1. MLP-based (Non-recurrent)**

- **Input**: `[batch, history_horizon, state_dim + action_dim]` → flattened
- **Base**: `MLPBase` with configurable hidden layers
- **Use case**: When full history is available

**2. RNN-based (Recurrent)**

- **Input**: `[batch, 1, state_dim + action_dim]` (sequential)
- **Base**: `RNNBase` with GRU or LSTM
- **Memory**: Hidden states maintained across time steps
- **Use case**: Long-horizon predictions, autoregressive rollouts

#### Network Components

**MLPBase** (`rsl_rl/modules/architectures/mlp.py:5-32`)

```python
class MLPBase(nn.Module):
    """Encodes state-action history into features"""
    
    def __init__(self, input_dim, device, architecture_config):
        # architecture_config["base_shape"] = [256, 256, 256]
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            ...
        )
    
    def forward(self, x_state_batch, x_action_batch):
        # Concat and flatten history
        x = torch.cat([x_state_batch, x_action_batch], dim=-1)
        x = x.flatten(1, 2)  # [batch, history*dim]
        return self.layers(x)  # [batch, hidden_dim]
```

**RNNBase** (`rsl_rl/modules/architectures/rnn.py:5-27`)

```python
class RNNBase(nn.Module):
    """Recurrent encoder for sequential processing"""
    
    def __init__(self, input_dim, device, architecture_config):
        # architecture_config["rnn_type"] = "gru"|"lstm"
        # architecture_config["rnn_hidden_size"] = 256
        # architecture_config["rnn_num_layers"] = 2
        self.memory = Memory(input_dim, device, ...)
    
    def forward(self, x_state_batch, x_action_batch):
        x = torch.cat([x_state_batch, x_action_batch], dim=-1)
        x = self.memory(x)  # Auto-updates hidden states
        return x[:, -1]  # Return last timestep
```

**MLPStateHead** (`rsl_rl/modules/architectures/mlp.py:35-97`)

```python
class MLPStateHead(nn.Module):
    """Predicts next state mean and std (per ensemble member)"""
    
    def forward(self, x, x_state_batch):
        # Predict delta state (residual connection)
        state_mean = self.state_mean_layers(x) + x_state_batch[:, -1]
        
        # Predict aleatoric uncertainty (bounded)
        state_logstd = self.state_logstd_layers(x)
        state_logstd = self.max_logstd - softplus(self.max_logstd - state_logstd)
        state_logstd = self.min_logstd + softplus(state_logstd - self.min_logstd)
        
        return state_mean, torch.exp(state_logstd)
```

**MLPAuxiliaryHead** (`rsl_rl/modules/architectures/mlp.py:100-172`)

```python
class MLPAuxiliaryHead(nn.Module):
    """Predicts auxiliary outputs (contacts, terminations, extensions)"""
    
    def forward(self, x, x_state_batch):
        extension_pred = self.extension_layers(x) if self.extension_dim > 0 else None
        contact_logits = self.contact_layers(x) if self.contact_dim > 0 else None
        termination_logits = self.termination_layers(x) if self.termination_dim > 0 else None
        
        return extension_pred, contact_logits, termination_logits
```

#### Critical Methods

**1. Forward (Prediction)**

Location: `rsl_rl/modules/system_dynamics.py:85-127`

```python
def forward(self, x_state_batch, x_action_batch, model_ids=None):
    """
    Predict next state and uncertainties
    
    Args:
        x_state_batch: [batch, history_horizon, state_dim]
        x_action_batch: [batch, history_horizon, action_dim]
        model_ids: [1, batch, 1] - which ensemble member to use (None = mean)
    
    Returns:
        output_state_means: [batch, state_dim]
        aleatoric_uncertainty: [batch] - data noise (mean std across dims)
        epistemic_uncertainty: [batch] - model disagreement (std across ensemble)
        output_extensions: [batch, extension_dim] or None
        output_contacts: [batch, contact_dim] or None
        output_terminations: [batch, termination_dim] or None
    """
```

**Flow**:
1. Pass history through `state_base` → features
2. Each `state_head` predicts `(mean, std)` → stack to `[ensemble_size, batch, state_dim]`
3. Same for auxiliary predictions
4. **Aggregate**:
   - If `model_ids=None`: Average across ensemble
   - Else: Select specific member with `torch.gather`
5. **Compute uncertainties**:
   - Aleatoric: `state_stds.mean(dim=0).sum(dim=1)` (average prediction uncertainty)
   - Epistemic: `state_means.std(dim=0).sum(dim=1)` (ensemble disagreement)

**2. Compute Loss (Training)**

Location: `rsl_rl/modules/system_dynamics.py:129-177`

```python
def compute_loss(self, state_batch, action_batch, extension_batch, 
                 contact_batch, termination_batch, bootstrap=False):
    """
    Compute training losses for all ensemble members
    
    Args:
        state_batch: [batch, seq_len, state_dim] - ground truth states
        action_batch: [batch, seq_len, action_dim]
        *_batch: Ground truth auxiliary outputs
        bootstrap: Whether to use bootstrap sampling for diversity
    
    Returns: 7 scalar losses
        state_loss, sequence_loss, bound_loss, kl_loss,
        extension_loss, contact_loss, termination_loss
    """
```

**Bootstrap Sampling**: Each ensemble member trained on different random subset of data (with replacement) to encourage diversity.

**Loss Components**:

1. **State Loss**: MSE or Gaussian NLL between predicted and actual next state
2. **Sequence Loss**: For recurrent models, loss on intermediate sequence predictions
3. **Bound Loss**: Regularization to prevent std bounds from collapsing
4. **KL Loss**: For variational models (RSSM)
5. **Extension Loss**: MSE on extension predictions
6. **Contact Loss**: BCE on contact predictions (binary)
7. **Termination Loss**: BCE on termination predictions (binary)

**3. Autoregressive Prediction**

Location: `rsl_rl/algorithms/mbpo_ppo.py:265-301`

```python
def system_dynamics_autoregressive_prediction(self, state_traj, action_traj, ...):
    """
    Roll out world model autoregressively
    
    For i in range(history_horizon, trajectory_length):
        state_pred[i] = world_model(state_pred[i-H:i], action[i-H:i])
    
    Used for:
    - Evaluation (trajectory error)
    - Visualization (real vs predicted)
    """
```

**4. Reset (State Management)**

Location: `rsl_rl/modules/system_dynamics.py:325-332`

```python
def reset(self):
    """Reset RNN hidden states (crucial for recurrent models)"""
    self.state_base.reset()
    for head in self.state_heads:
        head.reset()
    if self.auxiliary_base is not None:
        self.auxiliary_base.reset()
        for head in self.auxiliary_heads:
            head.reset()
```

**Called**: At start of each minibatch during training, before imagination rollouts.

---

## Algorithm Implementation

### File: `rsl_rl/algorithms/mbpo_ppo.py`

The **MBPOPPO** class extends base PPO with world model integration.

#### Class Structure

```python
class MBPOPPO(PPO):
    """Model-Based Policy Optimization with PPO"""
    
    # Inherited from PPO
    policy: ActorCritic              # Policy network
    optimizer: Adam                   # Policy optimizer
    storage: RolloutStorage          # Real experience buffer
    
    # Model-based additions
    system_dynamics: SystemDynamicsEnsemble  # ⭐ World model
    system_dynamics_optimizer: Adam          # World model optimizer
    system_replay_buffer: ReplayBuffer       # ⭐ For world model training
    imagination_storage: RolloutStorage      # ⭐ Imagined experience buffer
    
    state_normalizer: EmpiricalNormalization  # Normalize states
    action_normalizer: EmpiricalNormalization # Normalize actions
```

#### Key Methods

**1. Fill History Buffer**

Location: `rsl_rl/algorithms/mbpo_ppo.py:163-180`

```python
def fill_history_buffer(self, obs):
    """
    Extract system state/action from observations and store in replay buffer
    
    Called: After every environment step
    """
    system_state = obs["system_state"]        # [num_envs, state_dim]
    system_action = obs["system_action"]      # [num_envs, action_dim]
    system_extension = obs.get("system_extension")
    system_contact = obs.get("system_contact")
    system_termination = obs.get("system_termination")
    
    # Normalize before storing
    system_state = self.state_normalizer(system_state)
    system_action = self.action_normalizer(system_action)
    
    # Store as sequences
    self.system_replay_buffer.insert([
        system_state.unsqueeze(1),   # Add time dimension
        system_action.unsqueeze(1),
        ...
    ])
```

**2. Update System Dynamics**

Location: `rsl_rl/algorithms/mbpo_ppo.py:192-244`

```python
def update_system_dynamics(self):
    """
    Train world model on collected experience
    
    Returns: 7 mean losses across all minibatches
    """
    # Sample minibatches
    system_generator = self.system_replay_buffer.mini_batch_generator(
        sequence_length=self.history_horizon + self.forecast_horizon,
        num_mini_batches=self.system_dynamics_num_mini_batches,
        mini_batch_size=self.system_dynamics_mini_batch_size,
    )
    
    for state_batch, action_batch, ... in system_generator:
        self.system_dynamics.reset()  # Reset RNN states
        
        # Compute losses
        losses = self.system_dynamics.compute_loss(
            state_batch, action_batch, ..., bootstrap=True
        )
        
        # Weighted loss
        total_loss = sum(weight * loss for weight, loss in zip(weights, losses))
        
        # Optimize
        self.system_dynamics_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.system_dynamics.parameters(), max_grad_norm)
        self.system_dynamics_optimizer.step()
    
    return mean_losses
```

**3. Prepare Imagination**

Location: `rsl_rl/algorithms/mbpo_ppo.py:304-307`

```python
def prepare_imagination(self):
    """
    Sample initial states/actions from replay buffer for imagination rollouts
    
    Returns:
        imagination_state_history: [num_imagination_envs, history_horizon, state_dim]
        imagination_action_history: [num_imagination_envs, history_horizon, action_dim]
    """
    generator = self.system_replay_buffer.mini_batch_generator(
        sequence_length=self.history_horizon,
        num_mini_batches=1,
        mini_batch_size=self.imagination_storage.num_envs
    )
    state_history, action_history = next(generator)[:2]
    return state_history, action_history
```

**4. Combined Batch Generator**

Location: `rsl_rl/algorithms/mbpo_ppo.py:309-329`

```python
def mini_batch_generator_combined(self, real_storage, imagination_storage):
    """
    Combine real and imagined experience for policy updates
    
    Yields: Combined batches with double the data
    """
    for real_batch, imag_batch in zip(real_gen, imag_gen):
        combined = [torch.cat([r, i], dim=0) for r, i in zip(real_batch, imag_batch)]
        yield tuple(combined)
```

**5. Update (Policy Training)**

Location: `rsl_rl/algorithms/mbpo_ppo.py:331-580`

Same as PPO, but:
- If `imagination=True`: Uses `mini_batch_generator_combined`
- Trains on both real and imagined experience

---

## Training Runner

### File: `rsl_rl/runners/mbpo_on_policy_runner.py`

The **MBPOOnPolicyRunner** orchestrates the full training loop.

#### Training Loop Structure

Location: `rsl_rl/runners/mbpo_on_policy_runner.py:76-176`

```python
def learn(self, num_learning_iterations):
    """
    Main training loop
    
    For each iteration:
        1. Collect real rollouts (num_steps_per_env steps)
        2. Fill replay buffer
        3. Train world model (update_system_dynamics)
        4. [After warmup] Generate imagination rollouts
        5. [After warmup] Update policy on real + imagined data
        6. Log and save
    """
```

**Detailed Flow**:

```
Iteration i:
│
├─ 1. Real Rollout (num_steps_per_env = 24 steps)
│   ├─ For step in range(24):
│   │   ├─ actions = policy.act(obs)
│   │   ├─ obs, rewards, dones = env.step(actions)
│   │   ├─ [If i >= warmup] storage.add_transitions(...)
│   │   └─ fill_history_buffer(obs)  # Always add to replay buffer
│   └─ compute_returns(obs)  # Compute advantages
│
├─ 2. Train World Model
│   └─ losses = update_system_dynamics()
│
├─ 3. [If i >= warmup] Imagination Rollout
│   ├─ state_hist, action_hist = prepare_imagination()
│   ├─ For step in range(num_imagination_steps):
│   │   ├─ imag_obs = env.get_imagination_observation(state_hist, action_hist)
│   │   ├─ imag_actions = policy.act(imag_obs)
│   │   ├─ imag_obs, imag_rewards, ... = env.imagination_step(...)
│   │   └─ imagination_storage.add_transitions(...)
│   └─ compute_returns(imag_obs, imagination=True)
│
├─ 4. [If i >= warmup] Update Policy
│   └─ loss_dict = update(imagination=True)  # Train on real + imagined
│
└─ 5. Log and Save
```

#### Imagination Rollout

Location: `rsl_rl/runners/mbpo_on_policy_runner.py:178-215`

```python
def imagine(self):
    """
    Generate imagined rollouts using world model
    
    Returns: Observations, rewards, uncertainties for logging
    """
    # Initialize from replay buffer
    state_history, action_history = self.alg.prepare_imagination()
    self.env.unwrapped.prepare_imagination()
    
    for i in range(self.num_imagination_steps):
        # Resample commands periodically
        if i % command_resample_interval == 0:
            self.env.unwrapped.sample_imagination_command()
        
        # For RNN: keep only last state/action
        if architecture_type in ["rnn", "rssm"] and i > 0:
            state_history = state_history[:, -1:]
            action_history = action_history[:, -1:]
        
        # Get observation from current state
        imag_obs = self.env.unwrapped.get_imagination_observation(
            state_history, action_history
        )
        
        # Sample action from policy
        imag_actions = self.alg.act(imag_obs)
        
        # Step world model ⭐
        imag_obs, imag_rewards, imag_dones, extras, state_history, action_history, uncertainty = \
            self.env.unwrapped.imagination_step(imag_actions, state_history, action_history)
        
        # Store transition
        self.alg.process_env_step(imag_obs, imag_rewards, imag_dones, extras, imagination=True)
    
    self.alg.compute_returns(imag_obs, imagination=True)
    return ...
```

**Warmup Period**: World model is trained from iteration 0, but policy only uses imagined data after `system_dynamics_warmup_iterations` (e.g., 50 iterations).

---

## Data Storage Components

### 1. ReplayBuffer (World Model Training)

**File**: `rsl_rl/storage/replay_buffer.py`

```python
class ReplayBuffer:
    """Circular buffer for world model training data"""
    
    def __init__(self, dim, buffer_size, device):
        # dim = [state_dim, action_dim, extension_dim, contact_dim, termination_dim]
        self.buffer_size = buffer_size  # e.g., 10000
        self.replay_buf = [
            torch.zeros(num_envs, buffer_size, d, device=device) 
            for d in dim
        ]
```

**Key Methods**:

1. **insert** (line 30-54): Add new data in circular manner
2. **mini_batch_generator** (line 56-79): Sample sequences for training
3. **_generate_valid_indices** (line 81-86): Avoid sampling across episode boundaries

**Sampling Strategy**: 
- Samples continuous sequences of length `history_horizon + forecast_horizon`
- Avoids sequences that span episode resets (using termination flags)
- Supports bootstrap sampling for ensemble diversity

### 2. RolloutStorage (Policy Training)

**File**: `rsl_rl/storage/rollout_storage.py`

```python
class RolloutStorage:
    """On-policy buffer for PPO training"""
    
    def __init__(self, training_type, num_envs, num_transitions_per_env, obs, ...):
        self.observations = TensorDict(...)  # [num_transitions, num_envs, ...]
        self.actions = torch.zeros(...)
        self.rewards = torch.zeros(...)
        self.dones = torch.zeros(...)
        
        # For RL
        self.values = torch.zeros(...)
        self.returns = torch.zeros(...)
        self.advantages = torch.zeros(...)
```

**Key Methods**:

1. **add_transitions** (line 78-99): Store transition at current step
2. **compute_returns** (line 102-123): Compute GAE advantages
3. **mini_batch_generator** (line 125-161): Sample for PPO updates

**Two Instances**:
- `self.storage`: Real experience
- `self.imagination_storage`: Imagined experience

---

## Critical Interfaces

### 1. World Model ↔ Algorithm Interface

**Contract**: Any world model must implement these methods.

```python
class CustomWorldModel(nn.Module):
    # REQUIRED ATTRIBUTES
    state_dim: int
    action_dim: int
    extension_dim: int
    contact_dim: int
    termination_dim: int
    history_horizon: int
    ensemble_size: int
    architecture_config: dict
    device: str
    
    # REQUIRED METHODS
    
    def forward(self, x_state_batch, x_action_batch, model_ids=None):
        """
        Predict next state and uncertainties
        
        Args:
            x_state_batch: [batch, history_horizon, state_dim]
            x_action_batch: [batch, history_horizon, action_dim]
            model_ids: [1, batch, 1] or None
        
        Returns: 6 tensors
            state_means: [batch, state_dim]
            aleatoric_uncertainty: [batch]
            epistemic_uncertainty: [batch]
            extensions: [batch, extension_dim] or None
            contacts: [batch, contact_dim] or None
            terminations: [batch, termination_dim] or None
        """
        pass
    
    def compute_loss(self, state_batch, action_batch, extension_batch, 
                     contact_batch, termination_batch, bootstrap=False):
        """
        Compute training losses
        
        Args:
            *_batch: [batch, seq_len, *_dim]
            bootstrap: bool
        
        Returns: 7 scalar losses
            state_loss, sequence_loss, bound_loss, kl_loss,
            extension_loss, contact_loss, termination_loss
        """
        pass
    
    def reset(self):
        """Reset internal states (e.g., RNN hidden states)"""
        pass
```

**Called By**:
- `MBPOPPO.update_system_dynamics()`: Training
- `MBPOPPO.system_dynamics_autoregressive_prediction()`: Evaluation
- Environment's `imagination_step()`: Inference

### 2. World Model ↔ Environment Interface

**Environment Must Implement** (in `robotic_world_model` repo):

```python
class ManagerBasedMBRLEnv:
    
    def prepare_imagination(self):
        """Initialize imagination-specific state (e.g., sample commands)"""
        pass
    
    def sample_imagination_command(self):
        """Resample command during imagination rollout"""
        pass
    
    def get_imagination_observation(self, state_history, action_history):
        """
        Convert normalized state/action history to observation dict
        
        Args:
            state_history: [num_envs, history_horizon, state_dim] (normalized)
            action_history: [num_envs, history_horizon, action_dim] (normalized)
        
        Returns:
            obs: TensorDict with keys ["policy", "critic", ...]
        """
        pass
    
    def imagination_step(self, actions, state_history, action_history):
        """
        Execute one imagination step using world model
        
        Args:
            actions: [num_envs, action_dim]
            state_history: [num_envs, history_horizon, state_dim]
            action_history: [num_envs, history_horizon, action_dim]
        
        Returns:
            obs: TensorDict
            rewards: [num_envs, 1]
            dones: [num_envs, 1]
            extras: dict
            state_history: [num_envs, history_horizon, state_dim] (updated)
            action_history: [num_envs, history_horizon, action_dim] (updated)
            epistemic_uncertainty: [num_envs]
        
        Flow:
            1. Query world model: state_mean, aleatoric, epistemic, ... = 
                   system_dynamics.forward(state_history, action_history)
            2. Parse predictions into named states
            3. Compute rewards from predicted states
            4. Add uncertainty penalty to rewards (optional)
            5. Update history buffers (sliding window)
        """
        pass
```

**Location**: `robotic_world_model/source/mbrl/mbrl/envs/manager_based_mbrl_env.py`

### 3. Algorithm ↔ Runner Interface

**Runner creates Algorithm**:

```python
# In OnPolicyRunner.__init__
algorithm_class = eval(cfg["algorithm"]["class_name"])  # "MBPOPPO"
self.alg = algorithm_class(
    policy=policy,
    system_dynamics=system_dynamics,
    state_normalizer=state_normalizer,
    action_normalizer=action_normalizer,
    **algorithm_cfg
)
```

**Runner calls Algorithm methods**:

```python
# Collect data
self.alg.act(obs)
self.alg.process_env_step(obs, rewards, dones, extras)
self.alg.fill_history_buffer(obs)
self.alg.compute_returns(obs)

# Train
self.alg.update_system_dynamics()
self.alg.prepare_imagination()
self.alg.update(imagination=True)

# Evaluate
self.alg.evaluate_system_dynamics()
```

### 4. Observation Structure

**Expected Keys** (from environment):

```python
obs = TensorDict({
    # Policy network input
    "policy": torch.Tensor([num_envs, policy_obs_dim]),
    
    # Critic network input (may include privileged info)
    "critic": torch.Tensor([num_envs, critic_obs_dim]),
    
    # World model inputs (CRITICAL for RWM)
    "system_state": torch.Tensor([num_envs, state_dim]),      # Current state
    "system_action": torch.Tensor([num_envs, action_dim]),    # Last action
    
    # Optional world model targets
    "system_extension": torch.Tensor([num_envs, extension_dim]) or None,
    "system_contact": torch.Tensor([num_envs, contact_dim]) or None,
    "system_termination": torch.Tensor([num_envs, 1]) or None,
})
```

**Normalization Flow**:

```
Raw State → EmpiricalNormalization → Normalized State → ReplayBuffer
                                                       ↓
                                                  World Model Training
                                                       ↓
                                        Normalized Prediction → Denormalize → Physical State
```

---

## Configuration System

### Example Configuration Structure

**File**: `config/example_config.yaml`

```yaml
runner:
  class_name: MBPOOnPolicyRunner  # or OnPolicyRunner for model-free
  num_steps_per_env: 24
  max_iterations: 1500
  
  # Imagination settings (MBPO-specific)
  num_imagination_envs: 128       # How many parallel imagination rollouts
  num_imagination_steps: 24       # Length of imagination rollouts
  imagination_cfg:
    command_resample_interval: 6  # Resample commands every N steps
  
  # World model settings
  system_dynamics_warmup_iterations: 50   # Train WM before using imagination
  system_dynamics_num_visualizations: 3   # Number of trajectories to visualize
  system_dynamics_state_idx_dict: {...}  # For visualization
  
  policy:
    class_name: ActorCritic
    actor_hidden_dims: [256, 256, 256]
    critic_hidden_dims: [256, 256, 256]
  
  algorithm:
    class_name: MBPOPPO
    
    # PPO parameters
    learning_rate: 0.001
    num_learning_epochs: 5
    num_mini_batches: 4
    gamma: 0.99
    lam: 0.95
    
    # World model training
    system_dynamics_learning_rate: 1e-3
    system_dynamics_weight_decay: 0.0
    system_dynamics_forecast_horizon: 1
    system_dynamics_loss_weights:
      state: 1.0
      sequence: 1.0
      bound: 1.0
      kl: 0.0
      extension: 1.0
      contact: 1.0
      termination: 1.0
    system_dynamics_num_mini_batches: 10
    system_dynamics_mini_batch_size: 1000
    system_dynamics_replay_buffer_size: 10000
    
    # Evaluation
    system_dynamics_num_eval_trajectories: 10
    system_dynamics_len_eval_trajectory: 400
    system_dynamics_eval_traj_noise_scale: [0.1, 0.2, 0.4, 0.5, 0.8]
```

### World Model Architecture Config

**Passed to SystemDynamicsEnsemble**:

```python
architecture_config = {
    # Base type
    "type": "rnn",  # or "mlp"
    
    # For MLP
    "base_shape": [256, 256, 256],
    
    # For RNN
    "rnn_type": "gru",  # or "lstm"
    "rnn_num_layers": 2,
    "rnn_hidden_size": 256,
    
    # State head
    "state_mean_shape": [128],
    "state_logstd_shape": [128],
    
    # Auxiliary head
    "extension_shape": [128],
    "contact_shape": [128],
    "termination_shape": [128],
}
```

---

## Extension Points

### 1. Adding a New World Model

**Minimal Implementation**:

```python
# File: rsl_rl/modules/my_world_model.py

import torch
import torch.nn as nn

class TransformerWorldModel(nn.Module):
    """Custom transformer-based world model"""
    
    def __init__(self, state_dim, action_dim, extension_dim, contact_dim, 
                 termination_dim, device, ensemble_size=1, history_horizon=10,
                 architecture_config=None, **kwargs):
        super().__init__()
        
        # REQUIRED: Store all dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.extension_dim = extension_dim
        self.contact_dim = contact_dim
        self.termination_dim = termination_dim
        self.device = device
        self.ensemble_size = ensemble_size
        self.history_horizon = history_horizon
        self.architecture_config = architecture_config or {}
        
        # Build your architecture
        d_model = architecture_config.get("d_model", 256)
        nhead = architecture_config.get("nhead", 8)
        num_layers = architecture_config.get("num_layers", 6)
        
        self.embedding = nn.Linear(state_dim + action_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Ensemble of output heads
        self.state_heads = nn.ModuleList([
            nn.Linear(d_model, state_dim * 2)  # mean + logstd
            for _ in range(ensemble_size)
        ])
    
    def forward(self, x_state_batch, x_action_batch, model_ids=None):
        """REQUIRED: Implement prediction interface"""
        batch_size = x_state_batch.shape[0]
        
        # Embed history
        x = torch.cat([x_state_batch, x_action_batch], dim=-1)  # [B, H, S+A]
        x = self.embedding(x)  # [B, H, D]
        x = self.transformer(x)  # [B, H, D]
        x = x[:, -1]  # Take last timestep [B, D]
        
        # Ensemble predictions
        state_means_list = []
        state_stds_list = []
        for head in self.state_heads:
            out = head(x)  # [B, 2*state_dim]
            mean, logstd = out.chunk(2, dim=-1)
            mean = mean + x_state_batch[:, -1]  # Residual connection
            std = torch.exp(logstd.clamp(-10, 2))
            state_means_list.append(mean.unsqueeze(0))
            state_stds_list.append(std.unsqueeze(0))
        
        state_means = torch.cat(state_means_list, dim=0)  # [E, B, S]
        state_stds = torch.cat(state_stds_list, dim=0)    # [E, B, S]
        
        # Aggregate
        if model_ids is None:
            output_state_means = state_means.mean(dim=0)
        else:
            output_state_means = torch.gather(
                state_means, 0, model_ids.repeat(1, 1, self.state_dim)
            ).squeeze(0)
        
        # Uncertainties
        aleatoric = state_stds.mean(dim=0).sum(dim=1)
        epistemic = state_means.std(dim=0).sum(dim=1) if self.ensemble_size > 1 \
                    else torch.zeros(batch_size, device=self.device)
        
        # Dummy auxiliary outputs (or implement your own)
        extensions = None
        contacts = None
        terminations = None
        
        return output_state_means, aleatoric, epistemic, extensions, contacts, terminations
    
    def compute_loss(self, state_batch, action_batch, extension_batch, 
                     contact_batch, termination_batch, bootstrap=False):
        """REQUIRED: Implement training interface"""
        losses = []
        
        for i, head in enumerate(self.state_heads):
            # Bootstrap sampling
            if bootstrap:
                ids = torch.randint(0, state_batch.shape[0], 
                                   (state_batch.shape[0],), device=self.device)
            else:
                ids = torch.arange(0, state_batch.shape[0], device=self.device)
            
            # Single-step prediction
            state_input = state_batch[ids, :self.history_horizon]
            action_input = action_batch[ids, :self.history_horizon]
            state_target = state_batch[ids, self.history_horizon]
            
            # Forward
            x = torch.cat([state_input, action_input], dim=-1)
            x = self.embedding(x)
            x = self.transformer(x)[:, -1]
            out = head(x)
            mean, logstd = out.chunk(2, dim=-1)
            mean = mean + state_input[:, -1]
            
            # Gaussian NLL loss
            std = torch.exp(logstd.clamp(-10, 2))
            loss = nn.GaussianNLLLoss()(mean, state_target, std ** 2)
            losses.append(loss)
        
        state_loss = torch.stack(losses).mean()
        
        # Return 7 losses (pad with zeros if not used)
        return (state_loss, 
                torch.tensor(0.0, device=self.device),  # sequence_loss
                torch.tensor(0.0, device=self.device),  # bound_loss
                torch.tensor(0.0, device=self.device),  # kl_loss
                torch.tensor(0.0, device=self.device),  # extension_loss
                torch.tensor(0.0, device=self.device),  # contact_loss
                torch.tensor(0.0, device=self.device))  # termination_loss
    
    def reset(self):
        """REQUIRED: Reset internal states"""
        pass  # Transformers are stateless
```

**Register**:

```python
# rsl_rl/modules/__init__.py
from .my_world_model import TransformerWorldModel

__all__ = [..., "TransformerWorldModel"]
```

**Use**:

```python
# In task config
from rsl_rl.modules import TransformerWorldModel

system_dynamics = TransformerWorldModel(
    state_dim=45,
    action_dim=12,
    extension_dim=0,
    contact_dim=8,
    termination_dim=1,
    device="cuda",
    ensemble_size=5,
    history_horizon=10,
    architecture_config={
        "d_model": 256,
        "nhead": 8,
        "num_layers": 6,
    }
)
```

### 2. Modifying Loss Functions

**Location**: `rsl_rl/modules/system_dynamics.py:270-323`

**Current Losses**:

```python
def compute_regression_loss(self, state_mean_pred, state_std_pred, state_target, loss_type="mse"):
    if loss_type == "mse":
        state_pred = torch.randn_like(state_mean_pred) * state_std_pred + state_mean_pred
        state_loss = torch.sum(torch.square(state_pred - state_target), dim=1).mean()
    elif loss_type == "gaussian_nll":
        state_loss = nn.GaussianNLLLoss()(state_mean_pred, state_target, state_std_pred ** 2)
```

**Add Custom Loss**:

```python
def compute_regression_loss(self, state_mean_pred, state_std_pred, state_target, loss_type="mse"):
    if loss_type == "huber":
        # Robust to outliers
        delta = 1.0
        diff = state_mean_pred - state_target
        abs_diff = diff.abs()
        quadratic = torch.clamp(abs_diff, max=delta)
        linear = abs_diff - quadratic
        state_loss = (0.5 * quadratic ** 2 + delta * linear).sum(dim=1).mean()
        sequence_loss = torch.tensor(0.0, device=self.device)
        return state_loss, sequence_loss
    # ... existing code
```

### 3. Custom Uncertainty Penalties

**Location**: Environment's `imagination_step()` method

**Current**:

```python
# In robotic_world_model/source/mbrl/mbrl/envs/manager_based_mbrl_env.py
rewards -= self.uncertainty_penalty_weight * epistemic_uncertainty
```

**Alternatives**:

```python
# Adaptive penalty based on training progress
penalty_weight = self.base_weight * (1 - self.current_iter / self.max_iter)
rewards -= penalty_weight * epistemic_uncertainty

# Combined aleatoric + epistemic
total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
rewards -= self.uncertainty_weight * total_uncertainty

# Threshold-based
high_uncertainty_mask = epistemic_uncertainty > self.threshold
rewards[high_uncertainty_mask] *= 0.5  # Heavily penalize uncertain regions
```

### 4. Multi-Step Predictions

**Current**: World model predicts 1 step ahead (forecast_horizon=1)

**Modify**: `rsl_rl/modules/system_dynamics.py:179-231` (compute_state_loss)

```python
# Current: Single loop over forecast_horizon
for i in range(forecast_horizon):
    state_target = state_batch[:, self.history_horizon + i]
    ...

# Multi-step: Predict entire sequence at once
def forward_multistep(self, x_state_batch, x_action_batch, num_steps):
    predictions = []
    for i in range(num_steps):
        state_mean, ... = self.forward(x_state_batch, x_action_batch)
        predictions.append(state_mean)
        # Update history
        x_state_batch = torch.cat([x_state_batch[:, 1:], state_mean.unsqueeze(1)], dim=1)
    return torch.stack(predictions, dim=1)  # [batch, num_steps, state_dim]
```

---

## Summary of Critical Files

### World Model Core (Top Priority)

| File | Lines | Purpose |
|------|-------|---------|
| `rsl_rl/modules/system_dynamics.py` | 340 | **Main world model class** |
| `rsl_rl/modules/architectures/mlp.py` | 173 | MLP base and prediction heads |
| `rsl_rl/modules/architectures/rnn.py` | 44 | RNN base and memory |

### Algorithm & Training

| File | Lines | Purpose |
|------|-------|---------|
| `rsl_rl/algorithms/mbpo_ppo.py` | 581 | **MBPO-PPO algorithm** |
| `rsl_rl/algorithms/ppo.py` | 470 | Base PPO (for reference) |
| `rsl_rl/runners/mbpo_on_policy_runner.py` | 300+ | **Training loop orchestration** |

### Data Management

| File | Lines | Purpose |
|------|-------|---------|
| `rsl_rl/storage/replay_buffer.py` | 111 | **World model training buffer** |
| `rsl_rl/storage/rollout_storage.py` | 200+ | PPO training buffer |

### Policy Networks (Secondary)

| File | Lines | Purpose |
|------|-------|---------|
| `rsl_rl/modules/actor_critic.py` | 200+ | MLP policy |
| `rsl_rl/modules/actor_critic_recurrent.py` | 200+ | RNN policy |

### Dependencies

**External** (in `robotic_world_model` repo):
- Environment imagination interface
- Task configurations
- State/observation parsing

---

## Quick Reference: Key Dimensions

```python
# Environment
num_envs = 4096              # Parallel environments
num_steps_per_env = 24       # Steps per iteration

# World Model
state_dim = 45               # Robot state (pos, vel, orientation, etc.)
action_dim = 12              # Joint commands
history_horizon = 10         # Number of past states/actions
forecast_horizon = 1         # Prediction horizon
ensemble_size = 5            # Number of models

# Training
replay_buffer_size = 10000   # World model buffer
mini_batch_size = 1000       # World model minibatch
num_mini_batches = 10        # Per world model update
warmup_iterations = 50       # Before using imagination

# Imagination
num_imagination_envs = 128   # Parallel imagined rollouts
num_imagination_steps = 24   # Imagination horizon
```

---

## Version History

- **3.1.0** (Current): Uncertainty-aware RWM support
- **3.0.0**: Initial RWM implementation
- **Base**: Forked from `leggedrobotics/rsl_rl`

---

**End of Reference Document**
