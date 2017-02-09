from railrl.envs.env_utils import gym_env
from railrl.envs.memory.one_char_memory import OneCharMemory
from railrl.envs.memory.continuous_memory_augmented import ContinuousMemoryAugmented
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import \
    InvertedDoublePendulumEnv
from rllab.envs.normalized_env import normalize

# Although this import looks like it does nothing, but this is needed to use
# the envs in this package, because this call will register the environments.
from rllab.misc import logger
from rllab.misc.instrument import run_experiment_lite


def get_env_settings(
        env_id="",
        normalize_env=True,
        gym_name="",
        env_params=None,
        num_memory_states=0,
):
    """

    :param env_id: Env ID. See code for acceptable IDs.
    :param normalize_env: Boolean. If true, normalize the environment.
    :param gym_name: Gym environment name if env_id is "gym".
    :param env_params: Parameters to pass to the environment's constructor.
    :param num_memory_states: Number of memory states. If positive, then the
    environment is wrapped in a ContinuousMemoryAugmented env with this many
    memory states.
    :return:
    """
    assert num_memory_states >= 0
    if env_params is None:
        env_params = {}

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
        env = gym_env("OneDPoint-v0")
        name = "OneDPoint"
    elif env_id == 'reacher':
        env = gym_env("Reacher-v1")
        name = "Reacher"
    elif env_id == 'idp':
        env = InvertedDoublePendulumEnv()
        name = "InvertedDoublePendulum"
    elif env_id == 'ocm':
        env = OneCharMemory(**env_params)
        name = "OneCharMemory"
    elif env_id == 'gym':
        if gym_name == "":
            raise Exception("Must provide a gym name")
        env = gym_env(gym_name)
        name = gym_name
    else:
        raise Exception("Unknown env: {0}".format(env_id))
    if normalize_env:
        env = normalize(env)
        name += "-normalized"
    if num_memory_states > 0:
        env = ContinuousMemoryAugmented(
            env,
            num_memory_states=num_memory_states,
        )
    return dict(
        env=env,
        name=name,
        was_env_normalized=normalize_env,
    )


def run_experiment(
        task,
        exp_prefix,
        seed,
        variant,
        time=True,
        save_profile=False,
        profile_file='time_log.prof',
        **kwargs):
    """

    :param task:
    :param exp_prefix:
    :param seed:
    :param variant:
    :param time: Add a "time" command to the python command?
    :param save_profile: Create a cProfile log?
    :param kwargs:
    :return:
    """
    variant['seed'] = str(seed)
    logger.log("Variant:")
    logger.log(str(variant))
    command_words = []
    if time:
        command_words.append('time')
    command_words.append('python')
    if save_profile:
        command_words += ['-m cProfile -o', profile_file]
    run_experiment_lite(
        task,
        snapshot_mode="last",
        exp_prefix=exp_prefix,
        variant=variant,
        seed=seed,
        use_cloudpickle=True,
        python_command=' '.join(command_words),
        **kwargs
    )