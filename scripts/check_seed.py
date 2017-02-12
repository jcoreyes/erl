"""
Check that having the same seed doesn't change anything for our DDPG
implementation. Likewise, check that having different seeds does something
different.
"""
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic

from railrl.algos.ddpg import DDPG
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.instrument import run_experiment_lite
from sandbox.rocky.tf.envs.base import TfEnv


def run_task(_):
    for seed in range(3):
        env = TfEnv(HalfCheetahEnv())
        es = OUStrategy(env_spec=env.spec)
        qf = FeedForwardCritic(
            name_or_scope="critic",
            env_spec=env.spec,
        )
        policy = FeedForwardPolicy(
            name_or_scope="actor",
            env_spec=env.spec,
        )
        ddpg_params = dict(
            batch_size=16,
            n_epochs=100,
            epoch_length=100,
            eval_samples=100,
            max_path_length=10,
            min_pool_size=2,
        )
        algorithm = DDPG(
            env,
            es,
            policy,
            qf,
            **ddpg_params
        )

        algorithm.train(),


def main():
    for seed in range(3):
        run_experiment_lite(
            run_task,
            n_parallel=1,
            snapshot_mode="last",
            exp_prefix="check-ddpg-seed",
            seed=seed,
            use_cloudpickle=True,
        )

if __name__ == "__main__":
    main()
