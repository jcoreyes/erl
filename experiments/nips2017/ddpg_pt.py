"""
DDPG
"""
from railrl.envs.memory.high_low import HighLow
from railrl.launchers.launcher_util import (
    run_experiment,
    set_seed,
)


def run_linear_ocm_exp(variant):
    from railrl.torch.ddpg import DDPG
    from railrl.launchers.launcher_util import (
        set_seed,
    )
    from railrl.exploration_strategies.ou_strategy import OUStrategy
    from railrl.policies.torch import FeedForwardPolicy
    from railrl.qfunctions.torch import FeedForwardQFunction

    """
    Set up experiment variants.
    """
    H = variant['H']
    seed = variant['seed']
    algo_params = variant['algo_params']
    env_class = variant['env_class']
    env_params = variant['env_params']
    ou_params = variant['ou_params']

    set_seed(seed)

    """
    Code for running the experiment.
    """

    env = env_class(**env_params)

    # es = NoopStrategy(
    es = OUStrategy(
        env_spec=env.spec,
        **ou_params
    )
    qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    policy = FeedForwardPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    algorithm = DDPG(
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
    exp_prefix = "dev-pt-ddpg"

    # n_seeds = 10
    # mode = "ec2"
    # exp_prefix = "results-6-13-ddpg-highlow"

    # env_class = NormalizedHiddenCartpoleEnv
    # env_class = WaterMazeEasy
    env_class = HighLow
    H = 25
    num_steps_per_iteration = 100
    num_steps_per_eval = 1000
    num_iterations = 100
    use_gpu = True

    # noinspection PyTypeChecker
    variant = dict(
        H=H,
        algo_params=dict(
            batch_size=100,
            num_epochs=num_iterations,
            pool_size=1000000,
            num_steps_per_epoch=num_steps_per_iteration,
            num_steps_per_eval=num_steps_per_eval,
            max_path_length=H,
            discount=1,
            use_gpu=use_gpu,
        ),
        env_params=dict(
            num_steps=H,
            # use_small_maze=True,
            # l2_action_penalty_weight=0,
            # num_steps_until_reset=0,
        ),
        ou_params=dict(
            max_sigma=1,
            min_sigma=None,
        ),
        exp_prefix=exp_prefix,
        env_class=env_class,
        version="PyTorch DDPG"
    )
    exp_id = -1
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
            use_gpu=use_gpu,
        )
