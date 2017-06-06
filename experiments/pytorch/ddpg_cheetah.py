"""
Run PyTorch DDPG on HalfCheetah.
"""
from railrl.envs.point_env import PointEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.ddpg import DDPG
from rllab.envs.gym_env import GymEnv

from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from railrl.envs.env_utils import gym_env
from rllab.envs.normalized_env import normalize


def example(variant):
    env = HalfCheetahEnv()
    env = normalize(env)
    es = OUStrategy(env_spec=env.spec)
    algorithm = DDPG(
        env,
        exploration_strategy=es,
        **variant['algo_params']
    )
    algorithm.train()


if __name__ == "__main__":
    for target_qf_for_policy in [True, False]:
        for normal_policy_for_qf in [True, False]:
            variant = dict(
                algo_params=dict(
                    num_epochs=100,
                    num_steps_per_epoch=10000,
                    num_steps_per_eval=1000,
                    target_hard_update_period=1,
                    use_soft_update=False,
                    # tau=1e-3,
                    batch_size=128,
                    max_path_length=1000,
                    target_qf_for_policy=target_qf_for_policy,
                    normal_policy_for_qf=normal_policy_for_qf,
                ),
                version="PyTorch - no target",
            )
            for seed in range(10):
                run_experiment(
                    example,
                    exp_prefix="6-5-modified-ddpg",
                    # exp_prefix="dev",
                    seed=seed,
                    mode='ec2',
                    # mode='here',
                    variant=variant,
                )
