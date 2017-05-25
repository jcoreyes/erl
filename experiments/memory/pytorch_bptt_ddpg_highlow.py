"""
Try the PyTorch version of BPTT DDPG on HighLow env.
"""
import random
from railrl.envs.memory.continuous_memory_augmented import (
    ContinuousMemoryAugmented
)
from railrl.envs.memory.high_low import HighLow
from railrl.launchers.launcher_util import (
    run_experiment,
    set_seed,
)
from railrl.exploration_strategies.ou_strategy import OUStrategy


def experiment(variant):
    from railrl.algos.bptt_ddpg_pytorch import BDP
    from railrl.launchers.launcher_util import (
        set_seed,
    )
    from railrl.exploration_strategies.product_strategy import ProductStrategy
    seed = variant['seed']
    algo_params = variant['algo_params']
    es_params = variant['es_params']
    memory_dim = variant['memory_dim']
    env_params = variant['env_params']

    env_es_class = es_params['env_es_class']
    env_es_params = es_params['env_es_params']
    memory_es_class = es_params['memory_es_class']
    memory_es_params = es_params['memory_es_params']

    set_seed(seed)
    raw_env = HighLow(**env_params)
    env = ContinuousMemoryAugmented(
        raw_env,
        num_memory_states=memory_dim,
    )
    env_strategy = env_es_class(
        env_spec=raw_env.spec,
        **env_es_params
    )
    write_strategy = memory_es_class(
        env_spec=env.memory_spec,
        **memory_es_params
    )
    es = ProductStrategy([env_strategy, write_strategy])
    algorithm = BDP(
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
            num_steps=4,
        ),
        algo_params=dict(
            subtraj_length=4,
        ),
        es_params=dict(
            env_es_class=OUStrategy,
            env_es_params=dict(
                max_sigma=1,
                min_sigma=None,
            ),
            memory_es_class=OUStrategy,
            memory_es_params=dict(
                max_sigma=1,
                min_sigma=None,
            ),
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
