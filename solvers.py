import torch
import numpy as np
from discretisation_schemes import uniform_step_scheme
from jax_to_torch import torch_scan

def sdeint_ito_em_scan_ou(
        dim, alpha, f, g, y0, rng, args=(), dt=1e-06,
        step_scheme=uniform_step_scheme, start=0, end=1, dtype=torch.float32,
        scheme_args=None, ddpm_param=True):

    scheme_args = scheme_args if scheme_args is not None else {}
    ts = step_scheme(start, end, dt, dtype=dtype, **scheme_args)
    detach = True if args and "detach" in args else False

    y_pas = y0
    t_pas = ts[0]

    def euler_step(ytpas, t_):
        (y_pas, t_pas, rng) = ytpas

        delta_t = t_ - t_pas

        if ddpm_param:
            beta_k = torch.clamp(alpha * torch.sqrt(delta_t), 0, 1)
            alpha_k = torch.sqrt(1.0 - beta_k ** 2)
        else:
            alpha_k = torch.clamp(torch.exp(-alpha * delta_t), 0, 0.99999)
            beta_k = torch.sqrt(1.0 - alpha_k ** 2)

        # print(rng)
        # this_rng, rng = rng[0], rng[1]#torch.split(rng, 2)
        # this_rng_generator = torch.Generator()  # Create an instance of torch.Generator
        # this_rng_generator.manual_seed(this_rng.item())

        noise = torch.normal(0, 1, size=y_pas.shape, dtype=dtype)#, generator=this_rng_generator)

        y_pas_naug = y_pas[:, :dim].detach() if detach else y_pas[:, :dim]
        g_aug = g(y_pas, t_pas, args)
        f_aug = f(y_pas, t_pas, args)

        y_naug = y_pas_naug * alpha_k + f_aug[:, :dim] * beta_k ** 2 + (
                g_aug[:, :dim] * noise[:, :dim]) * beta_k
        u_dw = y_pas[:, dim:dim + 1].squeeze(1) + torch.einsum(
            "ij,ij->i", g_aug[:, dim:-1], noise[:, :dim]) * beta_k

        u_sq = y_pas[:, -1] + f_aug[:, -1] * beta_k ** 2

        if detach:
            f_aug_det = f(y_pas, t_pas, ["detach"])
            g_aug_det = g(y_pas, t_pas, ["detach"])
            dot = (f_aug[:, :dim] * f_aug_det[:, :dim]).sum(axis=-1)

            v_dw = torch.einsum(
                "ij,ij->i", g_aug_det[:, dim:-1], noise[:, :dim]) * beta_k

            log_is_weight = y_pas[:, -2] + f_aug_det[:, -1] * beta_k ** 2
            log_is_weight += v_dw + dot * beta_k ** 2
        else:
            log_is_weight = u_sq

        y = torch.cat((y_naug,
                       u_dw[..., None],
                       log_is_weight[..., None],
                       u_sq[..., None]), axis=-1)

        out = (y, t_, rng)
        print("EU_STEP")
        return out, y

    _, ys = torch_scan(euler_step, (y_pas, t_pas, rng), ts[1:])

    return torch.swapaxes(torch.cat((y0[None], ys), axis=0), 0, 1), ts







# NOT USED AND NOT CONVERTED

import jax
import haiku as hk

def sdeint_ito_em_scan(
        dim, f, g, y0, rng, args=(), dt=1e-06, g_prod=None,
        step_scheme=uniform_step_scheme, start=0, end=1, dtype=np.float32,
        scheme_args=None):
    """Vectorised (scan based) implementation of EM discretisation.

    Args:
      f: drift coeficient - vector field
      g: diffusion coeficient
      y0: samples from initial dist
      rng: rng key
      args: optional arguments for f and g
      dt: discertisation step
      g_prod: multiplication routine for diff coef / noise, defaults to hadamard
      step_scheme: how to spread out the integration steps defaults to uniform
      start: start time defaults to 0
      end: end time defaults to 1
      dtype: float32 or 64 (for tpu)
      scheme_args: args for step scheme

    Returns:
      Trajectory augmented with the path objective (Time x batch_size x (dim + 1))
    """

    scheme_args = scheme_args if scheme_args is not None else {}
    if g_prod is None:
        def g_prod(y, t, args, noise):
            out = g(y, t, args) * noise
            return out

    ts = step_scheme(start, end, dt, dtype=dtype, **scheme_args)

    y_pas = y0
    t_pas = ts[0]

    def euler_step(ytpas, t_):
        (y_pas, t_pas, rng) = ytpas

        delta_t = t_ - t_pas

        this_rng, rng = jax.random.split(rng)
        noise = jax.random.normal(this_rng, y_pas.shape, dtype=dtype)

        f_full = f(y_pas, t_pas, args)
        g_full = g_prod(
            y_pas, t_pas, args, noise
        )

        y = y_pas + f_full * delta_t + g_full * np.sqrt(delta_t)

        # t_pas = t_
        # y_pas = y
        out = (y, t_, rng)
        return out, y

    _, ys = hk.scan(euler_step, (y_pas, t_pas, rng), ts[1:])

    return np.swapaxes(np.concatenate((y0[None], ys), axis=0), 0, 1), ts