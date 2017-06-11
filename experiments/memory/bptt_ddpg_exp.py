"""
Try the PyTorch version of BPTT DDPG on HighLow env.
"""
import random
from railrl.envs.memory.continuous_memory_augmented import (
    ContinuousMemoryAugmented
)
from railrl.envs.memory.high_low import HighLow
from railrl.envs.water_maze import WaterMazeEasy, WaterMaze, WaterMazeMemory
from railrl.exploration_strategies.noop import NoopStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import (
    run_experiment,
    set_seed,
)
from railrl.misc.hyperparameter import DeterministicHyperparameterSweeper
from railrl.policies.torch import MemoryPolicy
from railrl.qfunctions.torch import MemoryQFunction
from rllab.misc.instrument import VariantGenerator


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
    env_class = variant['env_class']
    env_params = variant['env_params']
    memory_aug_params = variant['memory_aug_params']

    env_es_class = es_params['env_es_class']
    env_es_params = es_params['env_es_params']
    memory_es_class = es_params['memory_es_class']
    memory_es_params = es_params['memory_es_params']

    set_seed(seed)
    raw_env = env_class(**env_params)
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
    qf = MemoryQFunction(
        int(raw_env.observation_space.flat_dim),
        int(raw_env.action_space.flat_dim),
        memory_dim,
        400,
        300,
    )
    policy = MemoryPolicy(
        int(raw_env.observation_space.flat_dim),
        int(raw_env.action_space.flat_dim),
        memory_dim,
        400,
        300,
    )
    algorithm = BpttDdpg(
        env,
        es,
        qf=qf,
        policy=policy,
        **algo_params
    )
    algorithm.train()


if __name__ == '__main__':
    n_seeds = 1
    mode = "here"
    exp_prefix = "dev-pytorch"

    run_mode = 'none'
    # n_seeds = 5
    # mode = "ec2"
    # exp_prefix = "6-10-bddpg-water-maze-easy-h20"
    # run_mode = 'grid'
    # mode = "local_docker"

    use_gpu = True
    if mode == "ec2":
        use_gpu = False

    # H = 16
    # subtraj_length = 8
    H = 20
    subtraj_length = 20
    version = "H = {0}, subtraj length = {1}".format(H, subtraj_length)
    # noinspection PyTypeChecker
    variant = dict(
        # memory_dim=2,
        memory_dim=20,
        env_class=WaterMazeEasy,
        # env_class=WaterMaze,
        # env_class=WaterMazeMemory,
        # env_class=HighLow,
        env_params=dict(
            horizon=H,
            use_small_maze=True,
            l2_action_penalty_weight=0,
        ),
        memory_aug_params=dict(
            max_magnitude=1,
        ),
        algo_params=dict(
            subtraj_length=subtraj_length,
            batch_size=subtraj_length*32,
            # batch_size=32*32,
            num_epochs=100,
            # num_steps_per_epoch=100,
            num_steps_per_epoch=1000,
            discount=1.,
            use_gpu=use_gpu,
            policy_optimize_bellman=True,
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
        version=version,
    )
    if run_mode == 'grid':
        search_space = {
            'algo_params.qf_learning_rate': [1e-3, 1e-5],
            'algo_params.action_policy_learning_rate': [1e-3, 1e-5],
            'algo_params.write_policy_learning_rate': [1e-3, 1e-5],
        }
        sweeper = DeterministicHyperparameterSweeper(search_space,
                                                     default_parameters=variant)
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            for i in range(n_seeds):
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    seed=i,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
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
