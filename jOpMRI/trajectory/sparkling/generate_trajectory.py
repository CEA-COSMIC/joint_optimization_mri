from omegaconf import OmegaConf
from sparkling.parameters.initializations.common import INITIALIZATION
from sparkling.utils.argparse import fix_n_check_params
from sparkling import Run


def generate_trajectory_from_density(density, traj_config, num_samples_per_shot=None, num_shots=None, initialization=None, verbose=10):
    INIT = OmegaConf.to_object(OmegaConf.load(traj_config))
    del INIT['traj_init']
    del INIT['algo_params']['shape_grad']
    INIT['kinetic_constraint_init'] = INITIALIZATION['kinetic_constraint_init']
    INIT['algo_params']['max_grad_iter'] = INITIALIZATION['algo_params']['max_grad_iter']
    INIT = fix_n_check_params(INIT)
    INIT['dist_params']['density'] = density
    if num_shots is not None:
        INIT['traj_params']['num_shots'] = num_shots
    if num_samples_per_shot is not None:
        INIT['traj_params']['num_samples_per_shot'] = num_samples_per_shot
        if num_samples_per_shot <= 100:
            INIT['algo_params']['start_decim'] = 16
        if num_samples_per_shot <= 50:
            INIT['algo_params']['start_decim'] = 8
    if initialization is not None:
        INIT['traj_params']['initialization'] = initialization
    runObj = Run(**INIT, verbose=verbose)
    runObj.initialize_shots()
    while runObj.current['decim'] >= 1:
        runObj.start_optimization()
        runObj.update_params_decim(do_decimation=True)
    shots = runObj.recon_params.get_original_kspace_point(
        runObj.current['shots'])
    gradients, k0, slew_rate = runObj.scan_consts.get_gradient_file(
        shots, True)
    return runObj.current['shots'], INIT, gradients, k0
