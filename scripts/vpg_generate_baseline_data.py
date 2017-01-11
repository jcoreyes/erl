"""
Use this script to run vanilla policy gradient on various environments. Used
to create data to baseline against.
"""
from misc.scripts_util import timestamp
from rllab.misc.instrument import run_experiment_lite


def run_task(variant):
    from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
    from sandbox.rocky.tf.algos.vpg import VPG
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from rllab.envs.box2d.cartpole_env import CartpoleEnv
    from sandbox.rocky.tf.envs.base import TfEnv

    env_name = variant['Environment']
    if env_name == 'Cartpole':
        env = TfEnv(CartpoleEnv())
    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=(100, 100)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algorithm = VPG(
        env=env,
        policy=policy,
        baseline=baseline,
        n_itr=100,
        start_itr=0,
        batch_size=1000,
        max_path_length=1000,
        discount=0.99,
    )
    algorithm.train()


def main():
    exp_prefix = "vpg-generate-baseline-data-{0}".format(timestamp())
    for seed in range(3):
        variant = {
            'Environment': 'Cartpole'
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

if __name__ == "__main__":
    main()
