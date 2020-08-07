"""
Example of different ways to sweep hyperparameters.
"""
import random
import numpy as np
from hyperopt import hp
from rlkit.misc.hypopt import optimize_and_save
from rlkit.launchers.launcher_util import (
    create_log_dir,
    create_run_experiment_multiple_seeds,
)
from rlkit.launchers.launcher_util import run_experiment
import rlkit.misc.hyperparameter as hyp


def experiment(variant):
    return 0


if __name__ == '__main__':
    num_configurations = 1  # for random mode
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev"
    version = "Dev"
    run_mode = "none"

    # n_seeds = 10
    # mode = "ec2"
    # exp_prefix = "dev"
    # version = "Dev"

    run_mode = 'grid'
    use_gpu = True
    if mode != "local":
        use_gpu = False

    variant = dict(
        version=version,
    )
    if run_mode == 'grid':
        search_space = {
            'algo_params.discount': [1, 0.9],
        }
        sweeper = hyp.DeterministicHyperparameterSweeper(
            search_space, default_parameters=variant,
        )
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
                    use_gpu=use_gpu,
                )
    elif run_mode == 'custom_grid':
        for exp_id, (
                hp1,
                hp2,
        ) in enumerate([
            (True, 5),
            (False, 0),
            (True, 0),
            (False, 5),
        ]):
            variant['hp1'] = hp1
            variant['hp2'] = hp2
            for _ in range(n_seeds):
                seed = random.randint(0, 10000)
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                )
    elif run_mode == 'hyperopt':
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

        base_log_dir = create_log_dir(exp_prefix=exp_prefix)

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
    elif run_mode == 'random':
        hyperparameters = [
            hyp.LinearFloatParam('foo', 0, 1),
            hyp.LogFloatParam('bar', 1e-5, 1e2),
        ]
        sweeper = hyp.RandomHyperparameterSweeper(
            hyperparameters,
            default_kwargs=variant,
        )
        for _ in range(num_configurations):
            for exp_id in range(n_seeds):
                seed = random.randint(0, 10000)
                variant = sweeper.generate_random_hyperparameters()
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                    use_gpu=use_gpu,
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
            )
