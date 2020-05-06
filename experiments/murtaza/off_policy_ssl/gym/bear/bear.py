import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
import os

def experiment(variant):
    from railrl.core import logger
    logdir = logger.get_snapshot_dir()
    os.system('python -m BEAR.main --demo_data=research/railrl/data/local/demos/hc_action_noise_15.npy --off_policy_data=research/railrl/data/local/demos/hc_off_policy_15_demos_100.npy --eval_freq=1000 --algo_name=BEAR'
+ ' --env_name=HalfCheetah-v2 --log_dir='+logdir+' --lagrange_thresh=10.0 --distance_type=MMD'
+ ' --mode=auto --num_samples_match=5 --lamda=0.0 --version=0 --mmd_sigma=10.0 --kernel_type=laplacian --use_ensemble_variance="False" ')

if __name__ == "__main__":
    variant = dict(

    )

    search_space = {

    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_name = 'test1'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_name=exp_name,
                mode=mode,
                unpack_variant=False,
                variant=variant,
                num_exps_per_instance=1,
                use_gpu=False,
                gcp_kwargs=dict(
                    preemptible=False,
                ),
            )
