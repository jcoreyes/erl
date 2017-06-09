"""
DDPG
"""
from railrl.envs.memory.hidden_cartpole import NormalizedHiddenCartpoleEnv
from railrl.envs.water_maze import WaterMazeMemory, WaterMazeEasy
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
    from railrl.policies.nn_policy import FeedForwardPolicy
    from railrl.qfunctions.nn_qfunction import FeedForwardCritic

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

    es = OUStrategy(
        env_spec=env.spec,
        **ou_params
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
    exp_prefix = "dev-pt-ddpg"

    # n_seeds = 10
    # mode = "ec2"
    exp_prefix = "6-8-ddpg-watermaze-easy"

    env_class = NormalizedHiddenCartpoleEnv
    env_class = WaterMazeEasy
    H = 100
    num_steps_per_iteration = 1000
    num_iterations = 100
    use_gpu = True

    # noinspection PyTypeChecker
    variant = dict(
        H=H,
        algo_params=dict(
            batch_size=32,
            num_epochs=num_iterations,
            pool_size=1000000,
            num_steps_per_epoch=num_steps_per_iteration,
            num_steps_per_eval=100,
            max_path_length=H,
            discount=1,
            use_gpu=use_gpu,
        ),
        env_params=dict(
            num_steps=H,
            use_small_maze=True,
            l2_action_penalty_weight=0,
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
