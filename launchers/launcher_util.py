import datetime
import dateutil.tz
import os
import os.path as osp

from railrl.envs.env_utils import gym_env
from railrl.envs.memory.one_char_memory import OneCharMemory
from railrl.envs.memory.continuous_memory_augmented import (
    ContinuousMemoryAugmented
)
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import (
    InvertedDoublePendulumEnv
)
from rllab.envs.normalized_env import normalize

# Although this import looks like it does nothing, but this is needed to use
# the envs in this package, because this call will register the environments.
from rllab import config
from rllab.misc import logger
from rllab.misc.instrument import run_experiment_lite


def get_env_settings(
        env_id="",
        normalize_env=True,
        gym_name="",
        init_env_params=None,
        num_memory_states=0,
):
    """

    :param env_id: Env ID. See code for acceptable IDs.
    :param normalize_env: Boolean. If true, normalize the environment.
    :param gym_name: Gym environment name if env_id is "gym".
    :param init_env_params: Parameters to pass to the environment's constructor.
    :param num_memory_states: Number of memory states. If positive, then the
    environment is wrapped in a ContinuousMemoryAugmented env with this many
    memory states.
    :return:
    """
    if init_env_params is None:
        init_env_params = {}
    assert num_memory_states >= 0

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
        env = OneCharMemory(**init_env_params)
        name = "OneCharMemory"
    elif env_id == 'gym':
        if gym_name == "":
            raise Exception("Must provide a gym name")
        env = gym_env(gym_name)
        name = gym_name
    else:
        raise Exception("Unknown env: {0}".format(env_id))
    if normalize_env and env_id != 'ocm':
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
    :param profile_file: Where to save the cProfile log.
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


now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')


def create_exp_name(exp_prefix="default", exp_count=0):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_prefix:
    :param exp_count:
    :return:
    """
    return "%s_%s_%04d" % (exp_prefix, timestamp, exp_count)


def create_log_dir(exp_prefix="default", exp_count=0):
    """
    Creates and returns a unique log directory.

    :param exp_prefix: All experiments with this prefix will have log
    directories be under this directory.
    :param exp_count: Different exp_counts will be in different directories.
    :return:
    """
    exp_name = create_exp_name(exp_prefix=exp_prefix, exp_count=exp_count)
    log_dir = osp.join(
        config.LOG_DIR,
        'local',
        exp_prefix.replace("_", "-"),
        exp_name,
    )
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def setup_logger(
        exp_prefix=None,
        exp_count=0,
        variant=None,
        log_dir=None,
        text_log_file="debug.log",
        variant_log_file="variant.json",
        tabular_log_file="progress.csv",
        snapshot_mode="last",
        log_tabular_only=False,
        snapshot_gap=1,
):
    """
    Set up logger to have some reasonable default settings.

    :param exp_prefix:
    :param exp_count:
    :param variant_data:
    :param log_dir:
    :param text_log_file:
    :param variant_log_file:
    :param tabular_log_file:
    :param snapshot_mode:
    :param log_tabular_only:
    :param snapshot_gap:
    :return:
    """
    if log_dir is None:
        assert exp_prefix is not None
        log_dir = create_log_dir(exp_prefix, exp_count=exp_count)
    tabular_log_path = osp.join(log_dir, tabular_log_file)
    text_log_path = osp.join(log_dir, text_log_file)

    if variant is not None:
        variant_log_path = osp.join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    logger.add_text_output(text_log_path)
    logger.add_tabular_output(tabular_log_path)
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
