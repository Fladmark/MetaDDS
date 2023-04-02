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
        x = torch.tensor(np.random.randn(n_samples, d - 1)) * torch.exp(-y / 2)
        return torch.cat((y, x), axis=1)

    return neg_energy, sample_data