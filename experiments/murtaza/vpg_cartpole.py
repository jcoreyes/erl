"""
Run PyTorch VPG on Cartpole.
"""
import random

from railrl.data_management.env_replay_buffer import VPGEnvReplayBuffer
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.algos.vpg import VPG

from railrl.sac.policies import TanhGaussianPolicy
from rllab.envs.normalized_env import normalize
from rllab.envs.box2d.cartpole_env import CartpoleEnv

def example(variant):
    env = normalize(CartpoleEnv())
    env = normalize(env)
    policy = TanhGaussianPolicy(
        hidden_sizes=[400, 300],
        obs_dim=int(env.observation_space.flat_dim),
        action_dim=int(env.action_space.flat_dim),
    )
    algorithm = VPG(
        env,
        policy=policy,
        **variant['algo_params']
    )
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    num_steps_per_epoch = 1000
    max_path_length = 100
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_eval=1000,
            num_steps_per_epoch=num_steps_per_epoch,
            max_path_length=max_path_length,
            num_updates_per_epoch=1,
            batch_size=num_steps_per_epoch*max_path_length,
            replay_buffer=VPGEnvReplayBuffer,
        ),
        use_new_version=True,
    )
    run_experiment(
        example,
        exp_prefix="vpg_test",
        seed=random.randint(0, 10010),
        mode='local',
        variant=variant,
    )
