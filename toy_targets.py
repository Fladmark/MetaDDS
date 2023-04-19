import math

import torch
import numpy as np
from scipy.stats import norm
from torch.distributions import MultivariateNormal

def funnel(d=10, sig=3, clip_y=11):
    """Funnel distribution for testing. Returns energy and sample functions."""

    def neg_energy(x):
        def unbatched(x):
            v = x[0]
            log_density_v = norm.logpdf(v.item(),
                                        loc=0.,
                                        scale=3.)
            variance_other = torch.exp(v)
            other_dim = d - 1
            cov_other = torch.eye(other_dim) * variance_other
            mean_other = torch.zeros(other_dim)
            log_density_other = MultivariateNormal(mean_other, cov_other).log_prob(x[1:])
            return log_density_v + log_density_other

        output = torch.stack([unbatched(x_i) for x_i in x])
        return output

    def sample_data(n_samples):
        y = (sig * torch.tensor(np.random.randn(n_samples, 1))).clamp(-clip_y, clip_y)
        x = torch.tensor(np.random.randn(n_samples, d - 1)) * torch.exp(y / 2)
        #x = torch.tensor(np.random.randn(n_samples, d - 1)) * torch.exp(-y / 2)
        return torch.cat((y, x), axis=1)

    return neg_energy, sample_data

### OPTIMISATION TARGETS ###

def carrillo(x):
    def unbatched(x):
        v = x[0]
        log_density_v = norm.logpdf(v.item(),
                                    loc=0.,
                                    scale=3.)
        V_x = v**2 - 10*math.cos(2*math.pi*v) + 10
        V_x = torch.exp(-V_x/4)
        variance_other = V_x
        other_dim = 3
        cov_other = torch.eye(other_dim) * variance_other
        mean_other = torch.zeros(other_dim)
        log_density_other = MultivariateNormal(mean_other, cov_other).log_prob(x[1:])
        return log_density_v + log_density_other


    return torch.stack([unbatched(x_i) for x_i in x])

def carrillo2(x):
    def unbatched(x):
        v = x[0].item()
        sigma = 1
        V_x = v**2 - 10*math.cos(2*math.pi*v) + 10
        V_x = np.exp(-V_x/sigma)
        log_density = norm.logpdf(V_x)

        cov_other = torch.eye(1) * 1
        mean_other = torch.zeros(1)
        log_density_other = MultivariateNormal(mean_other, cov_other).log_prob(x)

        return log_density + log_density_other


    return torch.stack([unbatched(x_i) for x_i in x])

def mlpw(x):
    def unbatched(x):
        v = x[0]
        if torch.isnan(v):
            v = torch.nan_to_num(v)
        v = v #+ 2.
        log_density_v = norm.logpdf(v.item(),
                                    loc=0.,
                                    scale=3.)

        V_x = -(-math.sin(3*v) - v**2 + 0.7*v)
        #print(V_x)
        V_x = torch.exp(-V_x/ 1)
        variance_other = V_x
        #print(variance_other)
        other_dim = 3
        cov_other = torch.eye(other_dim) * variance_other
        mean_other = torch.zeros(other_dim)
        #print(f"cov other: {cov_other}")
        log_density_other = MultivariateNormal(mean_other, cov_other).log_prob(x[1:])
        return log_density_v + log_density_other


    return torch.stack([unbatched(x_i) for x_i in x])

def layeb01(x):
    def unbatched(x):
        v = x[0]
        print(v)
        # if torch.isnan(v):
        #     print("Nan")
        #     v = torch.nan_to_num(v)
        log_density_v = norm.logpdf(v.item(),
                                    loc=0.,
                                    scale=3.)

        V_x = torch.sqrt(torch.abs(torch.exp((v-1)**2))-1)
        #print(V_x)
        V_x = torch.exp(-V_x/ 5)
        if V_x == 0:
            variance_other = 0.001
        else:
            variance_other = V_x
        print(variance_other)
        other_dim = 3
        cov_other = torch.eye(other_dim) * variance_other
        mean_other = torch.zeros(other_dim)
        #print(f"cov other: {cov_other}")
        log_density_other = MultivariateNormal(mean_other, cov_other).log_prob(x[1:])
        return log_density_v + log_density_other


    return torch.stack([unbatched(x_i) for x_i in x])