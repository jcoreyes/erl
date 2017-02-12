"""
Compare DDPG and Quadratic-DDPG on Cheetah.
"""
from rllab.misc.instrument import run_experiment_lite


def run_task(variant):
    import tensorflow as tf
    from railrl.railrl.algos.ddpg import DDPG
    from railrl.policies.nn_policy import FeedForwardPolicy
    from railrl.qfunctions.nn_qfunction import FeedForwardCritic
    from railrl.qfunctions.quadratic_naf_qfunction import QuadraticNAF
    from rllab.exploration_strategies.ou_strategy import OUStrategy
    from sandbox.rocky.tf.envs.base import TfEnv
    from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv

    env = TfEnv(HalfCheetahEnv())
    algo_name = variant['Algorithm']
    if algo_name == 'Quadratic-DDPG':
        qf = QuadraticNAF(
            name_or_scope="quadratic_qf",
            env_spec=env.spec,
        )
    elif algo_name == 'DDPG':
        qf = FeedForwardCritic(
            name_or_scope="critic",
            env_spec=env.spec,
            embedded_hidden_sizes=(100,),
            observation_hidden_sizes=(100,),
            hidden_nonlinearity=tf.nn.relu,
        )
    else:
        raise Exception('Algo name not recognized: {0}'.format(algo_name))

    es = OUStrategy(env_spec=env.spec)
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
    )

    ddpg_params = dict(
        batch_size=128,
        n_epochs=20,
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
    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
        **ddpg_params
    )
    algorithm.train()


def main():
    exp_prefix = "ddpg-vs-qddpg"
    for seed in range(3):
        variant = {
            'Algorithm': 'DDPG',
            'Environment': 'Cheetah',
        }
        run_experiment_lite(
            run_task,
            snapshot_mode="last",
            exp_prefix=exp_prefix,
            seed=seed,
            mode="local",
            use_cloudpickle=True,
            variant=variant,
        )

        variant2 = {
            'Algorithm': 'Quadratic-DDPG',
            'Environment': 'Cheetah',
        }
        run_experiment_lite(
            run_task,
            snapshot_mode="last",
            exp_prefix=exp_prefix,
            seed=seed,
            mode="local",
            use_cloudpickle=True,
            variant=variant2,
        )

if __name__ == "__main__":
    main()
