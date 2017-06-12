"""
RDPG Experiments
"""
from railrl.envs.memory.high_low import HighLow
from railrl.envs.point_env import PointEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import RecurrentPolicy
from railrl.qfunctions.torch import RecurrentQFunction
from railrl.torch.rdpg import Rdpg
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.gym_env import GymEnv

from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from railrl.envs.env_utils import gym_env
from rllab.envs.normalized_env import normalize


def example(variant):
    env_class = variant['env_class']
    env = env_class(**variant['env_params'])
    es = OUStrategy(env_spec=env.spec)
    qf = RecurrentQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        4,
    )
    policy = RecurrentPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
    )
    algorithm = Rdpg(
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
            num_steps_per_epoch=100,
            num_steps_per_eval=100,
            batch_size=32,
            max_path_length=100,
            use_gpu=use_gpu,
        ),
        env_params=dict(
            num_steps=2,
        ),
        env_class=HighLow,
    )
    run_experiment(
        example,
        exp_prefix="dev-pytorch-rdpg",
        seed=0,
        mode='here',
        variant=variant,
        use_gpu=use_gpu,
    )
