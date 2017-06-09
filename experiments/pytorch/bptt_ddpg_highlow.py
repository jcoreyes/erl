"""
Try the PyTorch version of BPTT DDPG on HighLow env.
"""
import random
from railrl.envs.memory.continuous_memory_augmented import (
    ContinuousMemoryAugmented
)
from railrl.envs.memory.high_low import HighLow
from railrl.exploration_strategies.noop import NoopStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import (
    run_experiment,
    set_seed,
)


def experiment(variant):
    from railrl.torch.bptt_ddpg import BpttDdpg
    from railrl.launchers.launcher_util import (
        set_seed,
    )
    from railrl.exploration_strategies.product_strategy import ProductStrategy
    seed = variant['seed']
    algo_params = variant['algo_params']
    es_params = variant['es_params']
    memory_dim = variant['memory_dim']
    env_params = variant['env_params']
    memory_aug_params = variant['memory_aug_params']

    env_es_class = es_params['env_es_class']
    env_es_params = es_params['env_es_params']
    memory_es_class = es_params['memory_es_class']
    memory_es_params = es_params['memory_es_params']

    set_seed(seed)
    raw_env = HighLow(**env_params)
    env = ContinuousMemoryAugmented(
        raw_env,
        num_memory_states=memory_dim,
        **memory_aug_params
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
    algorithm = BpttDdpg(
        env,
        es,
        **algo_params
    )
    algorithm.train()


if __name__ == '__main__':
    n_seeds = 1
    mode = "here"
    exp_prefix = "dev-pytorch"

    n_seeds = 10
    mode = "ec2"
    exp_prefix = "6-8-hl-tf-bptt-check-limits"

    use_gpu = True
    if mode == "ec2":
        use_gpu = False

    H = 128
    subtraj_length = 8
    version = "H = {0}, subtraj length = {1}".format(H, subtraj_length)
    # noinspection PyTypeChecker
    variant = dict(
        memory_dim=20,
        env_params=dict(
            num_steps=H,
        ),
        memory_aug_params=dict(
            max_magnitude=1,
        ),
        algo_params=dict(
            subtraj_length=subtraj_length,
            num_epochs=40,
            num_steps_per_epoch=1000,
            discount=1.,
            use_gpu=use_gpu,
            policy_optimize_bellman=False,
        ),
        es_params=dict(
            env_es_class=OUStrategy,
            env_es_params=dict(
                max_sigma=1,
                min_sigma=None,
            ),
            # memory_es_class=NoopStrategy,
            memory_es_class=OUStrategy,
            memory_es_params=dict(
                max_sigma=1,
                min_sigma=None,
            ),
        ),
        version=version,
    )
    exp_id = 0
    for _ in range(n_seeds):
        seed = random.randint(0, 10000)
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
            use_gpu=use_gpu,
        )
