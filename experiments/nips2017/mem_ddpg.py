"""
DDPG + memory states.
"""
import tensorflow as tf

from railrl.envs.memory.high_low import HighLow
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import (
    run_experiment,
    set_seed,
)
from railrl.policies.memory.lstm_memory_policy import (
    SeparateLstmLinearCell,
    FlatLstmMemoryPolicy,
)
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
    memory_dim = variant['memory_dim']
    policy_params = variant['policy_params']

    set_seed(seed)

    """
    Code for running the experiment.
    """

    env = env_class(num_steps=H)
    env_action_dim = env.action_space.flat_dim
    env_obs_dim = env.observation_space.flat_dim
    env = ContinuousMemoryAugmented(
        env,
        num_memory_states=memory_dim,
    )
    env = FlattenedProductBox(env)

    qf = FeedForwardCritic(
        name_or_scope="critic",
        env_spec=env.spec,
    )
    policy = FlatLstmMemoryPolicy(
        name_or_scope="policy",
        action_dim=env_action_dim,
        memory_dim=memory_dim,
        env_obs_dim=env_obs_dim,
        env_spec=env.spec,
        **policy_params
    )
    es = OUStrategy(env_spec=env.spec)
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

    n_seeds = 5
    # mode = "ec2"
    exp_prefix = "5-16-benchmark-mddpg-hl"

    exp_id = -1
    algo_params = dict(
        batch_size=32,
        n_epochs=100,
        min_pool_size=100,
        replay_pool_size=100000,
        epoch_length=1000,
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
    variant = dict(
        H=32,
        exp_prefix=exp_prefix,
        algo_params=algo_params,
        env_class=HighLow,
        memory_dim=20,
        policy_params=policy_params,
        version="Memory DDPG"
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
