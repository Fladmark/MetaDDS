import torch.optim

from toy_targets import funnel
from stl_samplers import AugmentedOUDFollmerSDESTL
from config import get_architecture_specs, get_ref_process
from objectives import *

architecture_specs = get_architecture_specs()
ref_process = get_ref_process()

model_def = AugmentedOUDFollmerSDESTL(
    sigma=ref_process.sigma,
    dim=ref_process.data_dim,
    drift_network=ref_process.drift_network,
    tfinal=ref_process.tfinal,
    dt=ref_process.dt,
    step_scheme=ref_process.step_scheme,
    alpha=ref_process.alpha,
    target=ref_process.target,
    tpu=ref_process.tpu,
    detach_stl_drift=ref_process.detach_stl_drift,
    diff_net=ref_process.diff_net,
    detach_dritf_path=ref_process.detach_dritf_path,
    detach_dif_path=ref_process.detach_dif_path,
    m=ref_process.m,
    log=ref_process.log,
    exp_bool=ref_process.exp_bool,
    exact=ref_process.exact
)

# for name, param in ref_process.drift_network.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
#         print(param.requires_grad)

import time
for i in range(1, 101):
    s_t = time.time()

    aug_trajectory = model_def(batch_size=300, is_training=True, ode=False)

    # for j in aug_trajectory:
    #     print(j)

    g = ou_terminal_loss(lnpi=ref_process.target,
                         sigma=ref_process.sigma,
                         tfinal=ref_process.tfinal,
                         brown=False)

    loss = relative_kl_objective(augmented_trajectory=aug_trajectory,
                                 g=g,
                                 stl=False,
                                 trim=2,
                                 dim=ref_process.data_dim)

    print(loss)
    adam = torch.optim.Adam(ref_process.drift_network.parameters(), lr=0.001)
    adam.zero_grad()
    loss.backward()
    adam.step()

    e_t = time.time()

    print(f"Epoch {i}, time of execution: {e_t - s_t}")



