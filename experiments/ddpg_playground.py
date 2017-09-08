"""
Run DDPG on things.
"""
from railrl.envs.rollout_env import RemoteRolloutEnv
from railrl.envs.wrappers import convert_gym_space
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.algos.parallel_ddpg import ParallelDDPG
from railrl.torch.ddpg import DDPG
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import \
    InvertedDoublePendulumEnv
from rllab.envs.normalized_env import normalize



def example(variant):
    # env = InvertedDoublePendulumEnv()
    env = HalfCheetahEnv()
    env = normalize(env)
    obs_space = convert_gym_space(env.observation_space)
    action_space = convert_gym_space(env.action_space)
    es = OUStrategy(action_space=action_space)
    qf = FeedForwardQFunction(
        int(obs_space.flat_dim),
        int(action_space.flat_dim),
        400,
        300,
    )
    policy_class = FeedForwardPolicy
    policy_params = dict(
        obs_dim=int(obs_space.flat_dim),
        action_dim=int(action_space.flat_dim),
        fc1_size=400,
        fc2_size=300,
    )
    policy = policy_class(**policy_params)
    remote_env = RemoteRolloutEnv(
        HalfCheetahEnv,
        {},
        policy_class,
        policy_params,
        100,
    )
    algorithm = ParallelDDPG(
        remote_env,
        exploration_strategy=es,
        qf=qf,
        policy=policy,
        **variant['algo_params']
    )
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            max_path_length=100,
            batch_size=32,
        ),
    )
    run_experiment(
        example,
        exp_prefix="dev-ddpg",
        seed=0,
        mode='here',
        variant=variant,
        use_gpu=False,
    )
