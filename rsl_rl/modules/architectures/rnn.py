import torch
import torch.nn as nn


class RNNBase(nn.Module):
    def __init__(
        self,
        input_dim: int,
        device: str,
        architecture_config: dict = None,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        
        rnn_type = architecture_config["rnn_type"]
        rnn_num_layers = architecture_config["rnn_num_layers"]
        rnn_hidden_size = architecture_config["rnn_hidden_size"]
        self.memory = Memory(input_dim, device, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        
    def forward(self, x_state_batch, x_action_batch):
        x = torch.cat([x_state_batch, x_action_batch], dim=-1)
        x = self.memory(x)
        return x
    
    def reset(self):
        self.memory.reset()
    
    def reset_partial(self, env_ids):
        """Reset hidden states for specific environment indices."""
        self.memory.reset_partial(env_ids)


class Memory(nn.Module):
    def __init__(self, input_dim: int, device: str, type: str, num_layers: int, hidden_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=self.input_dim, hidden_size=hidden_size, num_layers=num_layers, device=self.device, batch_first=True)
        self.hidden_states = None

    def forward(self, x):
        x, self.hidden_states = self.rnn(x, self.hidden_states)
        return x[:, -1]
    
    def reset(self):
        self.hidden_states = None
    
    def reset_partial(self, env_ids):
        """Reset hidden states for specific environment indices."""
        if self.hidden_states is not None:
            # hidden_states shape: (num_layers, batch, hidden_size)
            self.hidden_states[:, env_ids, :] = 0.0