"""
Run Quadratic DDPG on a simple point task.
"""
from rllab.misc.instrument import run_experiment_lite


def run_task(_):
    from railrl.algos.ddpg import DDPG
    from railrl.policies.nn_policy import FeedForwardPolicy
    from railrl.qfunctions.quadratic_naf_qfunction import QuadraticNAF
    from rllab.exploration_strategies.ou_strategy import OUStrategy
    from sandbox.rocky.tf.envs.base import TfEnv
    from rllab.envs.gym_env import GymEnv

    def gym_env(name):
        return GymEnv(name,
                      record_video=False,
                      log_dir='/tmp/gym-test',  # Ignore gym log.
                      record_log=False)

    env = TfEnv(gym_env('AxeTwoDPoint-v0'))
    ddpg_params = dict(
        batch_size=128,
        n_epochs=50,
        epoch_length=1000,
        eval_samples=1000,
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
    algorithm.train()


def main():
    for seed in range(3):
        run_experiment_lite(
            run_task,
            n_parallel=1,
            snapshot_mode="last",
            exp_prefix="test-qddpg-point",
            seed=seed,
            use_cloudpickle=True,
        )

if __name__ == "__main__":
    main()
