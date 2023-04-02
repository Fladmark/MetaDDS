from discretisation_schemes import cos_sq_fn_step_scheme
import torch
from ml_collections import config_dict as configdict
from drift_nets import PISGRADNet
from toy_targets import funnel


architecture_specs = configdict.ConfigDict()
architecture_specs.alpha = 0.6875
architecture_specs.stop_grad = True
architecture_specs.fully_connected_units = [64, 64]
architecture_specs.activation = torch.nn.GELU

def get_architecture_specs():
    return architecture_specs

target, sample = funnel()
ref_process = configdict.ConfigDict()
ref_process.sigma = 1.075
ref_process.data_dim = 10
ref_process.drift_network = PISGRADNet(architecture_specs, 10)
ref_process.tfinal =6.4
ref_process.dt = 0.05
ref_process.step_scheme = cos_sq_fn_step_scheme
ref_process.alpha = 0.6875
ref_process.target = target
ref_process.tpu = True
ref_process.detach_stl_drift = False
ref_process.diff_net = None
ref_process.detach_dritf_path = False
ref_process.detach_dif_path = False
ref_process.m = 1
ref_process.log = False
ref_process.exp_bool = False
ref_process.exact = False

def get_ref_process():
    return ref_process