import random
import numpy as np
from hyperopt import hp
from railrl.misc.hypopt import optimize_and_save
from railrl.launchers.launcher_util import (
    create_base_log_dir,
    create_run_experiment_multiple_seeds,
)
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.hyperparameter import DeterministicHyperparameterSweeper


def experiment(variant):
    return 0


if __name__ == '__main__':
    n_seeds = 1
    mode = "here"
    exp_prefix = "dev"
    version = "Dev"
    run_mode = "none"

    # n_seeds = 10
    # mode = "ec2"
    # exp_prefix = "dev"
    # version = "Dev"

    # run_mode = 'grid'
    use_gpu = True
    if mode != "here":
        use_gpu = False

    variant = dict(
        version=version,
    )
    if run_mode == 'grid':
        search_space = {
            'algo_params.discount': [1, 0.9],
        }
        sweeper = DeterministicHyperparameterSweeper(search_space,
                                                     default_parameters=variant)
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            for i in range(n_seeds):
                seed = random.randint(0, 10000)
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                    sync_s3_log=True,
                    sync_s3_pkl=True,
                    periodic_sync_interval=600,
                )
    if run_mode == 'hyperopt':
        search_space = {
            'float_param': hp.uniform(
                'float_param',
                0.,
                5,
            ),
            'float_param2': hp.loguniform(
                'float_param2',
                np.log(0.01),
                np.log(1000),
            ),
            'seed': hp.randint('seed', 10000),
        }

        base_log_dir = create_base_log_dir(exp_prefix=exp_prefix)

        optimize_and_save(
            base_log_dir,
            create_run_experiment_multiple_seeds(
                n_seeds,
                experiment,
                exp_prefix=exp_prefix,
            ),
            search_space=search_space,
            extra_function_kwargs=variant,
            maximize=True,
            verbose=True,
            load_trials=True,
            num_rounds=500,
            num_evals_per_round=1,
        )
    else:
        for _ in range(n_seeds):
            seed = random.randint(0, 10000)
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                exp_id=0,
                use_gpu=use_gpu,
                sync_s3_log=True,
                sync_s3_pkl=True,
                periodic_sync_interval=600,
            )
