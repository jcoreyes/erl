"""
DDPG + memory states.
"""
from railrl.envs.memory.hidden_cartpole import NormalizedHiddenCartpoleEnv
from railrl.envs.water_maze import WaterMazeEasy, WaterMazeMemory
from railrl.launchers.launcher_util import (
    run_experiment,
    set_seed,
)


def run_linear_ocm_exp(variant):
    from railrl.algos.ddpg import DDPG
    from railrl.envs.flattened_product_box import FlattenedProductBox
    from railrl.exploration_strategies.ou_strategy import OUStrategy
    from railrl.policies.nn_policy import FeedForwardPolicy
    from railrl.qfunctions.nn_qfunction import FeedForwardCritic
    from railrl.envs.memory.continuous_memory_augmented import (
        ContinuousMemoryAugmented
    )
    from railrl.launchers.launcher_util import (
        set_seed,
    )

    """
    Set up experiment variants.
    """
    seed = variant['seed']
    algo_params = variant['algo_params']
    env_class = variant['env_class']
    env_params = variant['env_params']
    memory_dim = variant['memory_dim']
    ou_params = variant['ou_params']

    set_seed(seed)

    """
    Code for running the experiment.
    """

    env = env_class(**env_params)
    env = ContinuousMemoryAugmented(
        env,
        num_memory_states=memory_dim,
    )
    env = FlattenedProductBox(env)

    qf = FeedForwardCritic(
        name_or_scope="critic",
        env_spec=env.spec,
    )
    policy = FeedForwardPolicy(
        name_or_scope="policy",
        env_spec=env.spec,
    )
    es = OUStrategy(
        env_spec=env.spec,
        **ou_params
    )
    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
        **algo_params
    )

    algorithm.train()


if __name__ == '__main__':
    n_seeds = 1
    mode = "here"
    exp_prefix = "dev-mddpg"

    n_seeds = 10
    mode = "ec2"
    exp_prefix = "6-1-benchmark-normalized-hidden-cart-h100"

    exp_id = -1
    H = 100
    algo_params = dict(
        batch_size=32,
        n_epochs=100,
        min_pool_size=100,
        replay_pool_size=1000000,
        epoch_length=1000,
        eval_samples=400,
        max_path_length=1000,
        discount=1,
    )
    env_params = dict(
        num_steps=H,
        # use_small_maze=True,
    )
    ou_params = dict(
        max_sigma=1,
        min_sigma=None,
    )
    variant = dict(
        H=H,
        exp_prefix=exp_prefix,
        algo_params=algo_params,
        # env_class=HighLow,
        # env_class=WaterMazeEasy,
        # env_class=WaterMazeMemory,
        env_class=NormalizedHiddenCartpoleEnv,
        env_params=env_params,
        memory_dim=2,
        ou_params=ou_params,
        version="Memory DDPG",
    )
    for seed in range(n_seeds):
        exp_id += 1
        set_seed(seed)
        variant['seed'] = seed
        variant['exp_id'] = exp_id

        run_experiment(
            run_linear_ocm_exp,
            exp_prefix=exp_prefix,
            seed=seed,
            mode=mode,
            variant=variant,
            exp_id=exp_id,
        )
