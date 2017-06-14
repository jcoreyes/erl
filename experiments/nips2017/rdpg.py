from railrl.envs.memory.high_low import HighLow
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import (
    run_experiment,
    set_seed,
)
from railrl.policies.torch import RecurrentPolicy
from railrl.qfunctions.torch import RecurrentQFunction
from railrl.torch.rdpg import Rdpg


def experiment(variant):
    env_class = variant['env_class']
    env = env_class(**variant['env_params'])
    es = OUStrategy(env_spec=env.spec)
    qf = RecurrentQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        10,
    )
    policy = RecurrentPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        10,
    )
    algorithm = Rdpg(
        env,
        exploration_strategy=es,
        qf=qf,
        policy=policy,
        **variant['algo_params']
    )
    algorithm.train()


if __name__ == '__main__':
    n_seeds = 1
    mode = "here"
    exp_prefix = "dev-rdpg"

    # n_seeds = 10
    # mode = "ec2"
    # exp_prefix = "6-12-ddpg-watermaze-easy"

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
            subtraj_length=25,
        ),
        env_params=dict(
            num_steps=H,
            use_small_maze=True,
            l2_action_penalty_weight=0,
            num_steps_until_reset=0,
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
            experiment,
            exp_prefix=exp_prefix,
            seed=seed,
            mode=mode,
            variant=variant,
            exp_id=exp_id,
            use_gpu=use_gpu,
        )
