"""
Check that having the same seed doesn't change anything for existing DDPG
implementaiton. Likewise, check that having different seeds does something
different.
"""
import tensorflow as tf

from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.misc.instrument import run_experiment_lite, stub
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.exploration_strategies.gaussian_strategy import GaussianStrategy
from sandbox.rocky.tf.policies.deterministic_mlp_policy import (
    DeterministicMLPPolicy
)
from sandbox.rocky.tf.q_functions.continuous_mlp_q_function import (
    ContinuousMLPQFunction
)
from sandbox.rocky.tf.algos.ddpg import DDPG


def main():
    stub(globals())

    for seed in range(3):
        env = TfEnv(HalfCheetahEnv())
        es = GaussianStrategy(env.spec)
        policy = DeterministicMLPPolicy(
            name="init_policy",
            env_spec=env.spec,
            hidden_sizes=(100, 100),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
        )
        qf = ContinuousMLPQFunction(
            name="qf",
            env_spec=env.spec,
            hidden_sizes=(100, 100)
        )
        ddpg_params = dict(
            batch_size=4,
            n_epochs=100,
            epoch_length=50,
            eval_samples=50,
            max_path_length=10,
            min_pool_size=5,
        )
        algorithm = DDPG(
            env,
            policy,
            qf,
            es,
            **ddpg_params
        )

        for _ in range(3):
            run_experiment_lite(
                algorithm.train(),
                n_parallel=1,
                snapshot_mode="last",
                exp_prefix="check-rllab-ddpg-seed",
                seed=seed,
                variant={"seed": seed},
            )


if __name__ == "__main__":
    main()
