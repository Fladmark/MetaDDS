import numpy as np
import torch

# can be converted to torch
def uniform_step_scheme(start, end, dt, dtype=np.float32, **_):
  """Standard uniform scaling.

  Args:
    start: start time defaults to 0
    end: end time defaults to 1
    dt: 1/number of steps to divide grid into
    dtype: for tpu support
    **_: placeholder to handle different scheme args

  Returns:
    time grid
  """
  n_steps = int((end- start) / dt)

  ts = np.linspace(start, end, n_steps, dtype=dtype)

  return ts


def cos_sq_fn_step_scheme(
    start, end, dt, s=0.008, dtype=torch.float32, **_):
  """Exponential decay step scheme from Nichol and Dhariwal 2021.

  Args:
    start: start time defaults to 0
    end: end time defaults to 1
    dt: number of steps to divide grid into
    s: shift to ensure non 0
    dtype: for tpu support
    **_: placeholder to handle different scheme args

  Returns:
    time grid
  """
  n_steps = int((end - start) / dt)

  pre_phase = torch.linspace(start, end, n_steps, dtype=dtype) / end
  phase = ((pre_phase + s) / (1 + s)) * torch.tensor(np.pi * 0.5, dtype=dtype)
  # Note this multiples small numbers however implemented it more stably
  # (removed sqrt from solver only sqrd here) and it made no difference to
  # results
  dts = torch.cos(phase)**4

  dts /= dts.sum()
  dts *= end  # We normalise s.t. \sum_k \beta_k = T (where beta_k = b_m*cos^4)

  dts_out = torch.cat((torch.tensor([start], dtype=dtype), torch.cumsum(dts, dim=0)))
  return dts_out