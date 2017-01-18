"""
Run Quadratic DDPG on Cheetah.
"""
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.quadratic_naf_qfunction import QuadraticNAF

from railrl.algos.ddpg import DDPG
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.instrument import run_experiment_lite, stub
from sandbox.rocky.tf.envs.base import TfEnv


def main():
    stub(globals())
    env = TfEnv(HalfCheetahEnv())
    ddpg_params = dict(
        batch_size=128,
        n_epochs=50,
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
        qf_weight_decay=0.01,
    )
    es = OUStrategy(env_spec=env.spec)
    qf = QuadraticNAF(
        name_or_scope="quadratic_qf",
        env_spec=env.spec,
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
    )
    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
        **ddpg_params
    )

    env.reset()
    run_experiment_lite(
        algorithm.train(),
        n_parallel=1,
        snapshot_mode="last",
        exp_prefix="test-qddpg-cheetah",
        seed=1,
    )

if __name__ == "__main__":
    main()
