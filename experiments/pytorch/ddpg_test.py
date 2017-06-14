"""
Exampling of running DDPG on HalfCheetah.
"""
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG
from rllab.envs.box2d.cartpole_env import CartpoleEnv

from rllab.envs.normalized_env import normalize


def example(variant):
    # env = HalfCheetahEnv()
    # env = PointEnv()
    # env = gym_env("Pendulum-v0")
    env = CartpoleEnv()
    env = normalize(env)
    es = OUStrategy(env_spec=env.spec)
    qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    policy = FeedForwardPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    algorithm = DDPG(
        env,
        exploration_strategy=es,
        qf=qf,
        policy=policy,
        **variant['algo_params']
    )
    algorithm.train()


if __name__ == "__main__":
    use_gpu = True
    variant = dict(
        algo_params=dict(
            num_epochs=10,
            num_steps_per_epoch=1000,
            num_steps_per_eval=100,
            target_hard_update_period=10000,
            # use_soft_update=True,
            batch_size=32,
            max_path_length=100,
            use_gpu=use_gpu,
        )
    )
    run_experiment(
        example,
        exp_prefix="dev-pytorch-ddpg",
        seed=0,
        mode='here',
        variant=variant,
        use_gpu=use_gpu,
    )
