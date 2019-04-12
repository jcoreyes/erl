from railrl.launchers.ray.remote_launch import launch_remote_experiment
from railrl.launchers.ray.local_launch import launch_local_experiment

def launch_experiment(mode, local_launch_variant, use_gpu=False, *args,
                      **kwargs):
    local_launch_variant['use_gpu'] = use_gpu
    if mode == 'local':
        launch_local_experiment(**local_launch_variant)
    else:
        launch_remote_experiment(mode, local_launch_variant, use_gpu=use_gpu,
                                 *args, **kwargs)

