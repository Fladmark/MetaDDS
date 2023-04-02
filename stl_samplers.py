"""Module containing sampler objects.
"""

from discretisation_schemes import uniform_step_scheme
from solvers import sdeint_ito_em_scan
from solvers import sdeint_ito_em_scan_ou
from functools import partial
import torch
import torch.nn as nn
from config import get_architecture_specs


class AugmentedBrownianFollmerSDESTL(nn.Module):
    def __init__(
        self, sigma, dim, drift_network_class, tfinal=1, dt=0.05, target=None,
        step_fac=100, step_scheme=None,
        alpha=1, detach_dif_path=False, detach_stl_drift=False,
        detach_dritf_path=False, tpu=True, diff_net=None,
        name="BM_STL_follmer_sampler", **_
    ):
        super().__init__()
        self.gamma = (sigma) ** 2
        self.dtype = torch.float32 if tpu else torch.float64
        self.dim = dim
        self.drift_network = drift_network_class
        self.step_scheme = step_scheme
        self.tfinal = tfinal
        self.dt = dt
        self.target = target
        self.detach_drift_path = detach_dritf_path
        self.detach_dif_path = detach_dif_path
        self.detach_drift_stoch = detach_stl_drift
        self.detached_drift = drift_network_class
        self.detached_drift.ts = step_scheme(0, self.tfinal, self.dt, dtype=self.dtype, **dict())

    def forward(self, batch_size, is_training=True, dt=None, ode=False, exact=False):
        key = torch.randint(0, 2**32-1, (2,), dtype=torch.int64)
        dt = self.dt if dt is None or is_training else dt
        return self.sample_aug_trajectory(batch_size, key, dt=dt, is_training=is_training, ode=ode, exact=exact)

    def init_sample(self, n, key):
        return torch.zeros((n, self.dim), dtype=self.dtype)

    def f_aug(self, y, t, args):
        t_ = t * torch.ones((y.shape[0], 1), dtype=self.dtype)
        y_no_aug = y[..., :self.dim]
        u_t = self.drift_network(y_no_aug, t_, self.target)
        gamma_t = self.g_aug(y, t, args)[..., :self.dim] ** 2
        u_t_normsq = ((u_t) ** 2 / gamma_t).sum(dim=-1)[..., None] / 2.0
        n, _ = y_no_aug.shape
        zeros = torch.zeros((n, 1), dtype=self.dtype)
        return torch.cat((u_t, zeros, u_t_normsq), dim=-1)

    def g_aug(self, y, t, args):
        t_ = t * torch.ones((y.shape[0], 1), dtype=self.dtype)
        y_no_aug = y[..., :self.dim]
        n, _ = y_no_aug.shape
        gamma_ = torch.sqrt(self.gamma) * torch.ones_like(y_no_aug)
        zeros = torch.zeros((n, 1), dtype=self.dtype)
        if self.detach_drift_stoch:
            u_t = self.detached_drift(y_no_aug, t_, self.target)
        else:
            u_t = self.drift_network(y_no_aug, t_, self.target)
        out = torch.cat((gamma_, u_t / gamma_, zeros), dim=-1)
        return out

    def sample_aug_trajectory(self, batch_size, key, dt=0.05, rng=None, **_):
        y0 = self.init_sample(batch_size, key)
        zeros = torch.zeros((batch_size, 1), dtype=self.dtype)
        y0_aug = torch.cat((y0, zeros, zeros), dim=1)

        def g_prod(y, t, args, noise):
            g_aug = self.g_aug(y, t, args)
            gdw = g_aug[:, :self.dim] * noise[:, :self.dim]
            udw = torch.einsum("ij,ij->i", g_aug[:, self.dim:-1], noise[:, :self.dim])
            zeros = 0.0 * g_aug[:, -1] * noise[:, -1]
            return torch.cat((gdw, udw[..., None], zeros[..., None]), dim=-1)

        param_trajectory, ts = sdeint_ito_em_scan(
            self.dim, self.f_aug, self.g_aug, y0_aug, key, dt=dt,
            g_prod=g_prod, end=self.tfinal, step_scheme=self.step_scheme,
            dtype=self.dtype
        )

        return param_trajectory, ts

class AugmentedOUDFollmerSDESTL(AugmentedBrownianFollmerSDESTL):
    """Basic stationary OU prior based sampler (stl augmented).
    """
    def __init__(
        self, sigma, dim, drift_network, tfinal=1, dt=0.05, target=None,
        step_fac=100, step_scheme=uniform_step_scheme,
        alpha=1, detach_dif_path=False, tpu=True, detach_stl_drift=False,
        detach_dritf_path=False,
        diff_net=None, exp_bool=False, name="Eact_OU_STL_follmer_sampler", **_
    ):
        super().__init__(
            sigma, dim, drift_network,
            step_scheme=step_scheme,
            target=target, detach_dritf_path=detach_dritf_path,
            detach_stl_drift=detach_stl_drift, tpu=tpu,
            detach_dif_path=detach_dif_path, tfinal=tfinal, dt=dt,
            diff_net=diff_net, name=name)
        self.alpha = alpha
        self.sigma = sigma
        self.exp_bool = exp_bool

    def init_sample(self, n, key):
        return torch.normal(0, self.sigma, size=(n, self.dim))

    def f_aug(self, y, t, args):
        """See base class."""
        t_ = t * torch.ones((y.shape[0], 1))

        y_no_aug = y[..., :self.dim]

        ode = True if args and "ode" in args else False
        detach = True if args and "detach" in args else False

        u_t = (self.detached_drift(
            y_no_aug, t_,
            self.target, ode=ode
        ) if detach else self.drift_network(
            y_no_aug, t_,
            self.target, ode=ode
        ))

        gamma_t_sq = self.g_aug(y, t, args)[..., :self.dim]**2

        u_t_normsq = ((u_t)**2 / gamma_t_sq).sum(axis=-1)[..., None] / 2.0

        n, _ = y_no_aug.shape
        zeros = torch.zeros((n, 1))

        state = torch.cat((u_t, zeros, u_t_normsq), axis=-1)
        return state

    def g_aug(self, y, t, args):
        """See base class."""
        t_ = t * torch.ones((y.shape[0], 1))
        y_no_aug = y[..., :self.dim]

        n, _ = y_no_aug.shape

        gamma_t = self.sigma * torch.ones_like(y_no_aug)

        zeros = torch.zeros((n, 1))

        detach = True if args and "detach" in args else False
        if self.detach_drift_stoch or detach:
            u_t = self.detached_drift(y_no_aug, t_, self.target)
        else:
            u_t = self.drift_network(y_no_aug, t_, self.target)

        delta_t = (u_t)  / gamma_t

        out = torch.cat((gamma_t, delta_t, zeros), axis=-1)

        return out


    def sample_aug_trajectory(
            self, batch_size, key, dt=0.05, rng=None, ode=False, exact=False, **_):

        device = "cpu" # change

        y0 = self.init_sample(batch_size, key).to(device)
        zeros = torch.zeros((batch_size, 1), device=device)

        if ode:
            y0_aug = torch.cat((y0, zeros, zeros), dim=1)
        else:
            y0_aug = torch.cat((y0, zeros, zeros, zeros), dim=1)

        # notice no g_prod as that is handled internally by this specialized
        # ou based sampler.
        ddpm_param = not self.exp_bool
        #integrator = partial(odeint_em_scan_ou, exact=exact) if ode else sdeint_ito_em_scan_ou
        integrator = sdeint_ito_em_scan_ou

        param_trajectory, ts = integrator(
            self.dim, self.alpha, self.f_aug, self.g_aug, y0_aug, key, dt=dt,
            end=self.tfinal, step_scheme=self.step_scheme, ddpm_param=ddpm_param,
            dtype=self.dtype
        )

        return param_trajectory, ts