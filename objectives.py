import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np


def relative_kl_objective(augmented_trajectory, g, stl=False, trim=2, dim=2):
    """Vanilla relative KL control objective.

    Args:
        augmented_trajectory: X_{1:T} samples with ||u_t||^2/gamma as dim d+1
        g: terminal cost function typically - ln dπ/dp_1
        stl: boolean marking stl estimator usage
        trim: size of the augmented state space

    Returns:
        kl control loss
    """

    augmented_trajectory =  torch.nan_to_num(augmented_trajectory[0])
    #print(augmented_trajectory.shape)

    energy_cost_dt = augmented_trajectory[:, -1, -1]
    x_final_time = augmented_trajectory[:, -1, :dim]
    #print(energy_cost_dt)
    #print(x_final_time)

    # import pdb; pdb.set_trace()

    stl = augmented_trajectory[:, -1, dim] if stl else 0

    terminal_cost = g(x_final_time)

    return (energy_cost_dt + terminal_cost + stl).mean()

class ou_terminal_loss:
    def __init__(self, lnpi, sigma=1.0, tfinal=1.0, brown=False):
        self.lnpi = lnpi
        self.sigma = sigma
        self.tfinal = tfinal
        self.brown = brown
    """Terminal loss under OU reference prior at equilibrium.

    Can also be used for Brownian if you let sigma be the diff coef.

    Args:
        x_terminal: final time step samples from SDE
        lnpi: log target dist numerator
        sigma: stationary dist for OU dXt = -a* Xt * dt + sqrt(2a)*sigma*dW_t or
               diffusion coeficient for pinned brownian prior
        tfinal: terminal time value
        brown: flag for brownian reference process

    Returns:
        -(lnπ(X_T) - ln N(X_T; 0, sigma))
    """

    def __call__(self, x_terminal):
        _, d = x_terminal.shape
        ln_target = self.lnpi(x_terminal)

        if self.brown:
            sigma = np.sqrt(self.tfinal) * self.sigma
        else:
            sigma = self.sigma

        sigma_diag = sigma * torch.ones(d)
        equi_normal = MultivariateNormal(torch.zeros(d), torch.diag(sigma_diag).unsqueeze(0))
        #equi_normal = MultivariateNormal(torch.zeros(d), sigma * torch.ones(d))  # equilibrium distribution

        log_ou_equilibrium = equi_normal.log_prob(x_terminal)
        lrnd = -(ln_target - log_ou_equilibrium)

        # lrnd_mean = torch.mean(lrnd)
        # lrnd_std = torch.std(lrnd)
        # lrnd_norm = (lrnd - lrnd_mean) / lrnd_std
        # return lrnd_norm
        return lrnd