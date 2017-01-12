from main import gym_env
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import \
    InvertedDoublePendulumEnv
from rllab.envs.normalized_env import normalize


def get_env_settings(env_id="", normalize_env=True, gym_name=None):
    if env_id == 'cart':
        env = CartpoleEnv()
        name = "Cartpole"
    elif env_id == 'cheetah':
        env = HalfCheetahEnv()
        name = "HalfCheetah"
    elif env_id == 'ant':
        env = AntEnv()
        name = "Ant"
    elif env_id == 'point':
        env = gym_env("Pointmass-v1")
        name = "Pointmass"
    elif env_id == 'pt':
        env = gym_env("PointmassTarget-v1")
        name = "PointmassTarget"
    elif env_id == 'reacher':
        env = gym_env("Reacher-v1")
        name = "Reacher"
    elif env_id == 'idp':
        env = InvertedDoublePendulumEnv()
        name = "InvertedDoublePendulum"
    elif env_id == 'gym':
        if gym_name is None or gym_name == "":
            raise Exception("Must provide a gym name")
        env = gym_env(gym_name)
        name = gym_name
    else:
        raise Exception("Unknown env: {0}".format(env_id))
    if normalize_env:
        env = normalize(env)
        name += "-normalized"
    return dict(
        env=env,
        name=name,
        was_env_normalized=normalize_env,
    )
