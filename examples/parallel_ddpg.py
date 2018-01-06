"""
Run DDPG where the training happens in parallel of the environment.
"""
from railrl.envs.remote import RemoteRolloutEnv
from railrl.envs.wrappers import convert_gym_space
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.networks import FeedForwardQFunction, FeedForwardPolicy
from railrl.torch.algos.parallel_ddpg import ParallelDDPG
import railrl.torch.pytorch_util as ptu
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import (
    InvertedDoublePendulumEnv
)
from rllab.envs.normalized_env import normalize


def example(variant):
    env_class = variant['env_class']
    env_params = {}
    # Only create an env for the obs/action spaces
    env = env_class(**env_params)
    if variant['normalize_env']:
        env = normalize(env)
    obs_space = convert_gym_space(env.observation_space)
    action_space = convert_gym_space(env.action_space)
    qf = FeedForwardQFunction(
        int(obs_space.flat_dim),
        int(action_space.flat_dim),
        400,
        300,
    )
    es_class = OUStrategy
    es_params = dict(
        action_space=action_space,
    )
    policy_class = FeedForwardPolicy
    policy_params = dict(
        obs_dim=int(obs_space.flat_dim),
        action_dim=int(action_space.flat_dim),
        fc1_size=400,
        fc2_size=300,
    )
    es = es_class(**es_params)
    policy = policy_class(**policy_params)
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    remote_env = RemoteRolloutEnv(
        env,
        policy,
        exploration_policy,
        variant['max_path_length'],
        variant['normalize_env'],
    )
    algorithm = ParallelDDPG(
        remote_env,
        qf=qf,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    max_path_length = 1000
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            max_path_length=max_path_length,
            batch_size=128,
        ),
        max_path_length=max_path_length,
        env_class=InvertedDoublePendulumEnv,
        parallel=True,
        normalize_env=True,
    )
    run_experiment(
        example,
        exp_prefix="parallel-ddpg-example",
        seed=0,
        mode='here',
        variant=variant,
        use_gpu=False,
    )
