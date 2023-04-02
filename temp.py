class PISGRADNet(hk.Module):
  """PIS Grad network. Other than detaching should mimic the PIS Grad network.

  We detach the ULA gradients treating them as just features leading to much
  more stable training than PIS Grad.

  Attributes:
    config: ConfigDict specifying model architecture
  """

  def __init__(self,
               architecture_specs: configdict.ConfigDict,
               dim: int,
               name: Optional[str] = None):
    super(). __init__(name=name)

    self.alpha = architecture_specs.alpha
    self.stop_grad = architecture_specs.stop_grad
    self.architecture_specs = architecture_specs
    self.n_layers = len(self.architecture_specs.fully_connected_units)

    # For most PIS_GRAD experiments channels = 64
    self.channels = self.architecture_specs.fully_connected_units[0]
    self.timestep_phase = hk.get_parameter(
        "timestep_phase", shape=[1, self.channels], init=np.zeros)

    # Exact time_step coefs used in PIS GRAD
    self.timestep_coeff = np.linspace(
        start=0.1, stop=100, num=self.channels)[None]

    # This implements the time embedding for the non grad part of the network
    self.time_coder_state = hk.Sequential([
        hk.Linear(self.channels),
        self.architecture_specs.activation,
        hk.Linear(self.channels),
    ])

    # This carries out the time embedding for the NN(t) * log grad target
    self.time_coder_grad = hk.Sequential([hk.Linear(self.channels)] + [
        hk.Sequential(
            [self.architecture_specs.activation,
             hk.Linear(self.channels)]) for _ in range(self.n_layers)
    ] + [self.architecture_specs.activation,
         LinearConsInit(dim, 0)])

    # Time embedding and state concatenated network NN(x, emb(t))
    # This differs to PIS_grad where they do NN(Wx + emb(t))
    self.state_time_net = hk.Sequential([
        hk.Sequential([hk.Linear(x), self.architecture_specs.activation])
        for x in self.architecture_specs.fully_connected_units
    ] + [LinearZero(dim)])

    self.state_dim = dim
    self.dim = dim + 1
    self._grad_ln = hk.LayerNorm(-1, True, True)
    self.nn_clip = 1.0e4
    self.lgv_clip = 1.0e2

  def get_pis_timestep_embedding(self, timesteps: np.array):
    """PIS based timestep embedding.

    Args:
      timesteps: timesteps to embed

    Returns:
      embedded timesteps
    """

    sin_embed_cond = np.sin(
        (self.timestep_coeff * timesteps) + self.timestep_phase
    )
    cos_embed_cond = np.cos(
        (self.timestep_coeff * timesteps) + self.timestep_phase
    )
    return np.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

  def __call__(self,
               input_array: np.ndarray,
               time_array: np.ndarray,
               target: Optional[Callable[[np.ndarray], np.ndarray]] = None,
               training: Optional[bool] = True,
               ode: Optional[bool] = False) -> np.ndarray:
    """Evaluates (carries out a forward pass) the model at train/inference time.

    Args:
        input_array:  state to the network (N_points, N_dim)
        time_array:  time  to the network (N_points, 1)
        target: ln pi target for ULA based features
        training: if true evaluates the network in training phase else inference
        ode: Flag to turn of stopgrad for probflow estimator

    Returns:
        returns an ndarray of logits (N_points, n_dim)
    """

    time_array_emb = self.get_pis_timestep_embedding(time_array)

    grad_bool = self.stop_grad and not ode
    # Using score information as a feature
    grad = hk.grad(lambda _x: target(_x).sum())(input_array)
#     print("grad bool", grad_bool)
    grad = jax.lax.stop_gradient(grad) if grad_bool else grad
    grad = np.clip(grad, -self.lgv_clip, self.lgv_clip)

    t_net_1 = self.time_coder_state(time_array_emb)
    t_net_2 = self.time_coder_grad(time_array_emb)

    extended_input = np.concatenate((input_array, t_net_1), axis=-1)
    out_state = self.state_time_net(extended_input)

    out_state = np.clip(
        out_state, -self.nn_clip, self.nn_clip
    )

    out_state_p_grad = out_state + t_net_2 * grad
    return out_state_p_grad