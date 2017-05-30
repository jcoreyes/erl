"""
DDPG + memory states.
"""
import tensorflow as tf

from railrl.envs.memory.high_low import HighLow
from railrl.envs.water_maze import WaterMazeEasy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import (
    run_experiment,
    set_seed,
)
from railrl.policies.memory.lstm_memory_policy import (
    SeparateLstmLinearCell,
    FlatLstmMemoryPolicy,
)
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic


def run_linear_ocm_exp(variant):
    from railrl.algos.ddpg import DDPG
    from railrl.envs.flattened_product_box import FlattenedProductBox
    from railrl.envs.memory.continuous_memory_augmented import (
        ContinuousMemoryAugmented
    )
    from railrl.launchers.launcher_util import (
        set_seed,
    )

    """
    Set up experiment variants.
    """
    H = variant['H']
    seed = variant['seed']
    algo_params = variant['algo_params']
    env_class = variant['env_class']
    policy_params = variant['policy_params']
    ou_params = variant['ou_params']

    set_seed(seed)

    """
    Code for running the experiment.
    """

    env = env_class(num_steps=H)

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
    exp_prefix = "dev-ddpg"

    n_seeds = 10
    mode = "ec2"
    exp_prefix = "5-17-benchmark-ddpg-watermaze-easy"

    exp_id = -1
    algo_params = dict(
        batch_size=32,
        n_epochs=100,
        min_pool_size=100,
        replay_pool_size=100000,
        epoch_length=10000,
        eval_samples=100,
        max_path_length=1000,
        discount=1,
    )
    policy_params = dict(
        rnn_cell_class=SeparateLstmLinearCell,
        rnn_cell_params=dict(
            use_peepholes=True,
            env_noise_std=.0,
            memory_noise_std=0.,
            output_nonlinearity=tf.nn.tanh,
            env_hidden_sizes=[],
        )
    )
    ou_params = dict(
        max_sigma=1,
        min_sigma=None,
    )
    variant = dict(
        H=32,
        exp_prefix=exp_prefix,
        algo_params=algo_params,
        # env_class=HighLow,
        env_class=WaterMazeEasy,
        policy_params=policy_params,
        ou_params=ou_params,
        version="DDPG"
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
