from toy_targets import funnel
from stl_samplers import AugmentedOUDFollmerSDESTL
from config import get_architecture_specs, get_ref_process

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

result = model_def(batch_size=300, is_training=True, ode=False)
for i in result:
    print(i.shape)