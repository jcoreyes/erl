import random

from railrl.envs.memory.continuous_memory_augmented import \
    ContinuousMemoryAugmented
from railrl.envs.memory.high_low import HighLow
from railrl.envs.water_maze import WaterMazeEasy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.exploration_strategies.product_strategy import ProductStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import RecurrentPolicy, MemoryPolicy
from railrl.qfunctions.torch import RecurrentMemoryQFunction
from railrl.torch.bptt_ddpg_rq import BpttDdpgRecurrentQ


def example(variant):
    env_class = variant['env_class']
    memory_dim = variant['memory_dim']
    env_params = variant['env_params']
    memory_aug_params = variant['memory_aug_params']

    es_params = variant['es_params']
    env_es_class = es_params['env_es_class']
    env_es_params = es_params['env_es_params']
    memory_es_class = es_params['memory_es_class']
    memory_es_params = es_params['memory_es_params']

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

    qf = RecurrentMemoryQFunction(
        int(raw_env.observation_space.flat_dim),
        int(raw_env.action_space.flat_dim),
        memory_dim,
        10,
    )
    policy = MemoryPolicy(
        int(raw_env.observation_space.flat_dim),
        int(raw_env.action_space.flat_dim),
        memory_dim,
        400,
        300,
    )
    algorithm = BpttDdpgRecurrentQ(
        env,
        exploration_strategy=es,
        qf=qf,
        policy=policy,
        **variant['algo_params']
    )
    algorithm.train()


if __name__ == "__main__":
    use_gpu = True
    H = 32
    variant = dict(
        algo_params=dict(
            num_epochs=50,
            num_steps_per_epoch=100,
            num_steps_per_eval=100,
            batch_size=H*64,
            max_path_length=H,
            use_gpu=use_gpu,
            subtraj_length=8,
        ),
        env_params=dict(
            num_steps=H,
            # horizon=H,
            # use_small_maze=True,
            # l2_action_penalty_weight=0,
        ),
        env_class=HighLow,
        # env_class=WaterMazeEasy,
        memory_dim=20,
        memory_aug_params=dict(
            max_magnitude=1,
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
    seed = random.randint(0, 9999)
    run_experiment(
        example,
        exp_prefix="dev-pytorch-rdpg",
        # exp_prefix="dev-6-12-rdpg-small-water-maze-easy",
        seed=seed,
        mode='here',
        variant=variant,
        use_gpu=use_gpu,
    )
