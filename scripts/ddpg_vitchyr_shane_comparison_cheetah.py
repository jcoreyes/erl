"""
Compare my version of DDPG and Shane's version of DDPG on Cheetah.
"""
import tensorflow as tf
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic

from railrl.algos.ddpg import DDPG
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.exploration_strategies.gaussian_strategy import GaussianStrategy
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.instrument import run_experiment_lite, stub
from sandbox.rocky.tf.algos.ddpg import DDPG as ShaneDDPG
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.deterministic_mlp_policy import (
    DeterministicMLPPolicy
)
from sandbox.rocky.tf.q_functions.continuous_mlp_q_function import (
    ContinuousMLPQFunction
)


def main():
    stub(globals())
    env = TfEnv(HalfCheetahEnv())
    for seed in range(3):
        ddpg_params = dict(
            batch_size=128,
            n_epochs=100,
            epoch_length=10000,
            eval_samples=10000,
            discount=0.99,
            policy_learning_rate=1e-4,
            qf_learning_rate=1e-3,
            soft_target_tau=0.01,
            replay_pool_size=1000000,
            min_pool_size=256,
            scale_reward=1.0,
            max_path_length=1000,
            qf_weight_decay=0.0,
        )
        vitchyr_es = OUStrategy(env_spec=env.spec)
        vitchyr_qf = FeedForwardCritic(
            name_or_scope="critic",
            env_spec=env.spec,
        )
        vitchyr_policy = FeedForwardPolicy(
            name_or_scope="actor",
            env_spec=env.spec,
        )
        vitchyr_ddpg = DDPG(
            env,
            vitchyr_es,
            vitchyr_policy,
            vitchyr_qf,
            **ddpg_params
        )

        shane_es = GaussianStrategy(env.spec)
        shane_policy = DeterministicMLPPolicy(
            name="init_policy",
            env_spec=env.spec,
            hidden_sizes=(100, 100),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
        )
        shane_qf = ContinuousMLPQFunction(
            name="qf",
            env_spec=env.spec,
            hidden_sizes=(100, 100)
        )
        shane_ddpg = ShaneDDPG(
            env,
            shane_policy,
            shane_qf,
            shane_es,
            **ddpg_params
        )

        names_and_algos = [
            ("Vitchyr_DDPG", vitchyr_ddpg),
            ("Shane_DDPG", shane_ddpg),
        ]
        for name, algorithm in names_and_algos:
            env.reset()
            run_experiment_lite(
                algorithm.train(),
                n_parallel=1,
                snapshot_mode="last",
                exp_prefix="ddpg-comparison-cheetah",
                seed=seed,
            )

if __name__ == "__main__":
    main()
