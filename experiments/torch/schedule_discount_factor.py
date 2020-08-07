"""
See if slowly raising discount factor can stablize DDPG.
"""
import random

from rlkit.envs.mujoco.twod_point import TwoDPoint
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import (
    run_experiment,
    set_seed,
)


def experiment(variant):
    from rlkit.torch.ddpg import DDPG
    from rlkit.launchers.launcher_util import (
        set_seed,
    )
    seed = variant['seed']
    algo_params = variant['algo_params']
    env_params = variant['env_params']
    es_class = variant['es_class']
    es_params = variant['es_params']

    set_seed(seed)
    env = TwoDPoint(**env_params)
    es = es_class(
        env_spec=env.spec,
        **es_params
    )
    algorithm = DDPG(
        env,
        es,
        **algo_params
    )
    algorithm.train()


if __name__ == '__main__':
    n_seeds = 1
    mode = "here"
    exp_prefix = "dev-pytorch"

    # noinspection PyTypeChecker
    variant = dict(
        memory_dim=20,
        env_params=dict(
        ),
        algo_params=dict(
            subtraj_length=16,
        ),
        es_class=OUStrategy,
        es_params=dict(
            max_sigma=1,
            min_sigma=None,
        ),
    )
    exp_id = -1
    for _ in range(n_seeds):
        seed = random.randint(0, 99999)
        exp_id += 1
        set_seed(seed)
        variant['seed'] = seed
        variant['exp_id'] = exp_id

        run_experiment(
            experiment,
            exp_prefix=exp_prefix,
            seed=seed,
            mode=mode,
            variant=variant,
            exp_id=exp_id,
        )
