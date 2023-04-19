import torch
import toy_targets
from jax_to_torch import get_div_fn

device = "cpu"  # self.device

y0 = torch.normal(0, 1, size=(200, 4))
zeros = torch.zeros((200, 1), device=device)
shape = torch.cat((y0, zeros, zeros), dim=1).shape

get_div_fn()