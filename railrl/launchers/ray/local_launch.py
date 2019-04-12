import logging

import ray
import ray.tune as tune
from ray.tune.logger import JsonLogger

from railrl.core.ray_experiment import SequentialRayExperiment
from railrl.core.ray_csv_logger import SequentialCSVLogger
import railrl.launchers.ray_config as config

def launch_local_experiment(init_algo_functions_and_log_fnames,
                            exp_variant, use_gpu=False,
                            exp_prefix='test', seeds=1, checkpoint_freq=50,
                            max_failures=10, resume=False, local_ray=True,
                            from_remote=False, resources_per_trial=None,
                            logging_level=logging.DEBUG):
    if from_remote:
        redis_address = ray.services.get_node_ip_address() + ':6379'
        ray.init(redis_address=redis_address, logging_level=logging_level)
    else:
        ray.init(local_mode=local_ray)
    for idx, (init_func, log_fname) in enumerate(init_algo_functions_and_log_fnames):
        init_algo_functions_and_log_fnames[idx] = (
            tune.function(init_func),
            log_fname
        )
    exp = tune.Experiment(
        name=exp_prefix,
        run=SequentialRayExperiment,
        upload_dir=config.LOG_BUCKET,
        num_samples=seeds,
        stop={"global_done": True},
        config={
            'algo_variant': exp_variant,
            'init_algo_functions_and_log_fnames': init_algo_functions_and_log_fnames,
            'use_gpu': use_gpu,
        },
        resources_per_trial=resources_per_trial,
        checkpoint_freq=checkpoint_freq,
        loggers=[JsonLogger, SequentialCSVLogger],
    )
    tune.run(
        exp,
        resume=resume,
        max_failures=max_failures,
        queue_trials=True,
    )


"""
This main should only be invoked by the ray on the remote instance. See
remote_launch.py. The experiment info is pickled and uploaded to the remote
instance. Then, ray invokes this file to start a local experiment from the pkl.
"""
if __name__ == "__main__":
    with open(config.EXPERIMENT_INFO_PKL_FILEPATH, "rb") as f:
        local_launch_variant = cloudpickle.load(f)
    launch_local_experiment(**local_launch_variant)
