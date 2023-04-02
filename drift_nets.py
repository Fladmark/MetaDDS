import torch
import torch.nn as nn
from torch.autograd import grad
import numpy as np


class LinearConsInit(nn.Module):
    """Linear layer with constant init.
    """

    def __init__(self, input_size,output_size, alpha=1):
        super().__init__()
        self.alpha = alpha
        self.output_size = output_size
        self.input_size = input_size
        self.linear = nn.Linear(self.input_size, output_size, bias=True)
        self.linear.weight.data = torch.eye(output_size, self.input_size) * self.alpha
        self.linear.bias.data.zero_()

    def forward(self, x):
        return self.linear(x)


class LinearZero(nn.Module):
    """Linear layer with zero init.
    """

    def __init__(self, input_size,output_size, alpha=1):
        super().__init__()
        self.alpha = alpha
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size, bias=True)
        self.linear.weight.data = torch.zeros(output_size, input_size, dtype=torch.float32)
        self.linear.bias.data = torch.zeros(output_size, dtype=torch.float32)

    def forward(self, x):
        return self.linear(x)



class PISGRADNet(nn.Module):
    def __init__(self, architecture_specs, dim, name=None):
        super().__init__()

        self.alpha = architecture_specs.alpha
        self.stop_grad = architecture_specs.stop_grad
        self.architecture_specs = architecture_specs
        self.n_layers = len(self.architecture_specs.fully_connected_units)

        self.channels = self.architecture_specs.fully_connected_units[0]
        self.timestep_phase = nn.Parameter(torch.zeros(1, self.channels))

        self.timestep_coeff = torch.linspace(
            start=0.1, end=100, steps=self.channels)[None]

        self.size_of_pis_timestep_embedding = 128
        self.size_of_extended_input = 64 + 10

        self.time_coder_state = nn.Sequential(
            nn.Linear(self.size_of_pis_timestep_embedding, self.channels),
            self.architecture_specs.activation(), # sus
            nn.Linear(self.channels, self.channels),
        )

        layers = [nn.Linear(self.size_of_pis_timestep_embedding, self.channels)]
        for _ in range(self.n_layers):
            layers.append(self.architecture_specs.activation()) ## sus
            layers.append(nn.Linear(self.channels, self.channels))
        layers.append(self.architecture_specs.activation()) # sus
        layers.append(LinearConsInit(self.channels, dim, 0))
        self.time_coder_grad = nn.Sequential(*layers)

        state_time_layers = []
        state_time_layers.append(nn.Linear(self.channels+10, self.channels))
        state_time_layers.append(self.architecture_specs.activation())
        state_time_layers.append(nn.Linear(self.channels, self.channels))
        state_time_layers.append(self.architecture_specs.activation())
        state_time_layers.append(LinearZero(self.channels, dim))
        self.state_time_net = nn.Sequential(*state_time_layers)

        self.state_dim = dim
        self.dim = dim + 1
        self.nn_clip = 1.0e4
        self.lgv_clip = 1.0e2

    def get_pis_timestep_embedding(self, timesteps):
        sin_embed_cond = torch.sin(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        cos_embed_cond = torch.cos(
            (self.timestep_coeff * timesteps) + self.timestep_phase
        )
        return torch.cat([sin_embed_cond, cos_embed_cond], dim=-1)


    def forward(self, input_array, time_array, target=None, training=True, ode=False):
        time_array_emb = self.get_pis_timestep_embedding(time_array)

        grad_bool = self.stop_grad and not ode
        input_array.requires_grad_(True)
        target_val = target(input_array).sum()
        grad_val = grad(target_val, input_array, create_graph=not grad_bool)[0]
        grad_val = grad_val.clamp(-self.lgv_clip, self.lgv_clip)
        if grad_bool:
            grad_val = grad_val.detach()


        t_net_1 = self.time_coder_state(time_array_emb)
        t_net_2 = self.time_coder_grad(time_array_emb)

        input_array = input_array.type(torch.float32)
        extended_input = torch.cat((input_array, t_net_1), dim=-1)


        out_state = self.state_time_net(extended_input)

        out_state = out_state.clamp(-self.nn_clip, self.nn_clip)
        out_state_p_grad = out_state + t_net_2 * grad_val

        return out_state_p_grad