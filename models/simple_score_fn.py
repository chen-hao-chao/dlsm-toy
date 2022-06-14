import torch
import torch.nn as nn

class simple_noise_conditioned_score_fn(nn.Module):
  """noise-conditioned score model"""

  def __init__(self, config):
    super().__init__()

    self.config = config
    self.nf = nf = config.model.nf
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2, nf * 16),
      nn.ReLU(),
      nn.Linear(nf * 16, nf * 8),
      nn.ReLU(),
      nn.Linear(nf * 8, nf * 4)
    )
    self.cond_layers = nn.Sequential(
      nn.Linear(1, nf * 4),
      nn.ReLU(),
      nn.Linear(nf * 4, nf * 2),
      nn.ReLU(),
      nn.Linear(nf * 2, nf * 1)
    )
    self.linear = nn.Linear(nf * 5, 2)

  def forward(self, x, cond):
    cond = torch.unsqueeze(cond, 1)
    cond = self.cond_layers(cond)
    x = self.layers(x)
    x_cond_cat = torch.cat((x, cond), 1)
    out = self.linear(x_cond_cat)
    return out

class simple_score_fn(nn.Module):
  """simple score model"""

  def __init__(self, config):
    super().__init__()

    self.config = config
    self.nf = nf = config.model.nf
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2, nf * 16),
      nn.ReLU(),
      nn.Linear(nf * 16, nf * 8),
      nn.ReLU(),
      nn.Linear(nf * 8, nf * 4)
    )
    self.linear = nn.Linear(nf * 4, 2)

  def forward(self, x):
    x = self.layers(x)
    out = self.linear(x)
    return out
