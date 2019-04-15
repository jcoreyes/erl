from gym.envs.mujoco import HopperEnv

import railrl.torch.pytorch_util as ptu
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.launchers.launcher_util import setup_logger
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
from railrl.torch.td3.td3 import TD3


def experiment(variant):
    env = NormalizedBoxEnv(HopperEnv())
    es = GaussianStrategy(
        action_space=env.action_space,
        max_sigma=0.1,
        min_sigma=0.1,  # Constant sigma
    )
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = TD3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            num_epochs=200,
            num_steps_per_epoch=5000,
            num_steps_per_eval=10000,
            max_path_length=1000,
            batch_size=100,
            discount=0.99,
            replay_buffer_size=int(1E6),
        ),
    )
    setup_logger('name-of-td3-experiment', variant=variant)
    experiment(variant)
