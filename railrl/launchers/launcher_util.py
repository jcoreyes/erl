import __main__ as main
import time
import sys
import datetime
import os
import os.path as osp
import random
import uuid
import git
import json
import base64
import cloudpickle

import dateutil.tz
import numpy as np
import tensorflow as tf

from railrl.envs.env_utils import gym_env
from railrl.envs.memory.continuous_memory_augmented import (
    ContinuousMemoryAugmented
)
from railrl.envs.memory.one_char_memory import (
    OneCharMemory,
    OneCharMemoryEndOnly,
    OneCharMemoryOutputRewardMag,
)
from railrl.launchers import config
from railrl.pythonplusplus import dict_to_safe_json
from railrl.torch.pytorch_util import set_gpu_mode
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import (
    InvertedDoublePendulumEnv
)
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize
from rllab.misc import logger
from rllab.misc.instrument import run_experiment_lite, query_yes_no

import doodad
import doodad.mode
import doodad.mount as mount
from doodad.utils import REPO_DIR

def get_standard_env(normalized=True):
    envs = [
        HalfCheetahEnv(),
        CartpoleEnv(),
        InvertedDoublePendulumEnv(),
        HalfCheetahEnv(),
        AntEnv(),
        SwimmerEnv(),
    ]
    if normalized:
        envs = [normalize(e) for e in envs]
    return envs


def get_standard_env_ids():
    return [
        'cart',
        'cheetah',
        'ant',
        'reacher',
        'idp',
        'swimmer',
    ]


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
    elif env_id == 'random2d':
        env = gym_env("TwoDPointRandomInit-v0")
        name = "TwoDPoint-RandomInit"
    elif env_id == 'reacher':
        env = gym_env("Reacher-v1")
        name = "Reacher"
    elif env_id == 'idp':
        env = InvertedDoublePendulumEnv()
        name = "InvertedDoublePendulum"
    elif env_id == 'swimmer':
        env = SwimmerEnv()
        name = "Swimmer"
    elif env_id == 'ocm':
        env = OneCharMemory(**init_env_params)
        name = "OneCharMemory"
    elif env_id == 'ocme':
        env = OneCharMemoryEndOnly(**init_env_params)
        name = "OneCharMemoryEndOnly"
    elif env_id == 'ocmr':
        env = OneCharMemoryOutputRewardMag(**init_env_params)
        name = "OneCharMemoryOutputRewardMag"
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
        method_call,
        mode='local',
        exp_prefix='default',
        seed=None,
        variant=None,
        exp_id=0,
        unique_id=None,
        prepend_date_to_exp_prefix=True,
        use_gpu=False,
        snapshot_mode='last',
        snapshot_gap=1,
        n_parallel=0,
        base_log_dir=None,
        sync_interval=180,
):
    # Modify some of the inputs
    if seed is None:
        seed = random.randint(0, 100000)
    if variant is None:
        variant = {}
    if unique_id is None:
        unique_id = str(uuid.uuid4())
    if prepend_date_to_exp_prefix:
        exp_prefix = time.strftime("%m-%d") + "-" + exp_prefix
    variant['seed'] = str(seed)
    variant['exp_id'] = str(exp_id)
    variant['unique_id'] = str(unique_id)
    logger.log("Variant:")
    logger.log(json.dumps(dict_to_safe_json(variant), indent=2))

    mode_str_to_doodad_mode = {
        'local': doodad.mode.Local(),
        'local_docker': doodad.mode.LocalDocker(
            image=config.DOODAD_DOCKER_IMAGE,
        ),
        'ec2': doodad.mode.EC2AutoconfigDocker(
            image=config.DOODAD_DOCKER_IMAGE,
            region='us-west-1',
            instance_type='c4.large',
            spot_price=0.03,
            s3_log_prefix=exp_prefix,
            s3_log_name="{}-id{}-s{}".format(exp_prefix, exp_id, seed),
        ),
    }

    if base_log_dir is None:
        local_output_dir = config.LOCAL_LOG_DIR
    else:
        local_output_dir = '{}/{}'.format(base_log_dir, exp_prefix)
    mounts = [
        mount.MountLocal(local_dir=REPO_DIR, pythonpath=True),
    ]
    for code_dir in config.CODE_DIRS_TO_MOUNT:
        mounts.append(mount.MountLocal(local_dir=code_dir, pythonpath=True))

    if mode != 'local':
        for non_code_mapping in config.DIR_AND_MOUNT_POINT_MAPPINGS:
            mounts.append(mount.MountLocal(**non_code_mapping))

    if mode == 'ec2':
        if not query_yes_no(
                "EC2 costs money. Are you sure you want to run?"
        ):
            sys.exit(1)
        if use_gpu:
            if not query_yes_no(
                    "EC2 is more expensive with GPUs. Confirm?"
            ):
                sys.exit(1)
        output_mount = mount.MountS3(
            s3_path='',
            mount_point=config.OUTPUT_DIR_FOR_DOODAD_TARGET,
            output=True,
            sync_interval=sync_interval,
        )
    else:
        output_mount = mount.MountLocal(
            local_dir=local_output_dir,
            mount_point=config.OUTPUT_DIR_FOR_DOODAD_TARGET,
            output=True,
        )
    mounts.append(output_mount)

    repo = git.Repo(os.getcwd())
    run_experiment_kwargs = dict(
        exp_prefix=exp_prefix,
        variant=variant,
        exp_id=exp_id,
        seed=seed,
        use_gpu=use_gpu,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        code_diff=repo.git.diff(None),
        commit_hash=repo.head.commit.hexsha,
        script_name=main.__file__,
        n_parallel=n_parallel,
        base_log_dir=config.OUTPUT_DIR_FOR_DOODAD_TARGET,
    )
    doodad.launch_python(
        target=config.RUN_DOODAD_EXPERIMENT_SCRIPT_PATH,
        mode=mode_str_to_doodad_mode[mode],
        mount_points=mounts,
        args={
            'method_call': method_call,
            'output_dir': config.OUTPUT_DIR_FOR_DOODAD_TARGET,
            'run_experiment_kwargs': run_experiment_kwargs,
        }
    )


def run_experiment_old(
        task,
        exp_prefix='default',
        seed=None,
        variant=None,
        time_it=True,
        save_profile=False,
        profile_file='time_log.prof',
        mode='here',
        exp_id=0,
        unique_id=None,
        prepend_date_to_exp_prefix=True,
        use_gpu=False,
        snapshot_mode='last',
        snapshot_gap=1,
        n_parallel=0,
        base_log_dir=None,
        **run_experiment_lite_kwargs
):
    """
    Run a task via the rllab interface, i.e. serialize it and then run it via
    the run_experiment_lite script.

    :param task:
    :param exp_prefix:
    :param seed:
    :param variant:
    :param time_it: Add a "time" command to the python command?
    :param save_profile: Create a cProfile log?
    :param profile_file: Where to save the cProfile log.
    :param mode: 'here' will run the code in line, without any serialization
    Other options include 'local', 'local_docker', and 'ec2'. See
    run_experiment_lite documentation to learn what those modes do.
    :param exp_id: Experiment ID. Should be unique across all
    experiments. Note that one experiment may correspond to multiple seeds.
    :param unique_id: Unique ID should be unique across all runs--even different
    seeds!
    :param prepend_date_to_exp_prefix: If True, prefix "month-day_" to
    exp_prefix
    :param run_experiment_lite_kwargs: kwargs to be passed to
    `run_experiment_lite`
    :return:
    """
    if seed is None:
        seed = random.randint(0, 100000)
    if variant is None:
        variant = {}
    if unique_id is None:
        unique_id = str(uuid.uuid4())
    if prepend_date_to_exp_prefix:
        exp_prefix = time.strftime("%m-%d") + "_" + exp_prefix
    variant['seed'] = str(seed)
    variant['exp_id'] = str(exp_id)
    variant['unique_id'] = str(unique_id)
    logger.log("Variant:")
    logger.log(json.dumps(dict_to_safe_json(variant), indent=2))
    command_words = []
    if time_it:
        command_words.append('time')
    command_words.append('python')
    if save_profile:
        command_words += ['-m cProfile -o', profile_file]
    repo = git.Repo(os.getcwd())
    diff_string = repo.git.diff(None)
    commit_hash = repo.head.commit.hexsha
    script_name = main.__file__
    if mode == 'here':
        run_experiment_here(
            task,
            exp_prefix=exp_prefix,
            variant=variant,
            exp_id=exp_id,
            seed=seed,
            use_gpu=use_gpu,
            snapshot_mode=snapshot_mode,
            snapshot_gap=snapshot_gap,
            code_diff=diff_string,
            commit_hash=commit_hash,
            script_name=script_name,
            n_parallel=n_parallel,
            base_log_dir=base_log_dir,
        )
    else:
        if mode == "ec2" and use_gpu:
            if not query_yes_no(
                "EC2 is more expensive with GPUs. Confirm?"
            ):
                sys.exit(1)
        code_diff = (
            base64.b64encode(cloudpickle.dumps(diff_string)).decode("utf-8")
        )
        run_experiment_lite(
            task,
            snapshot_mode=snapshot_mode,
            snapshot_gap=snapshot_gap,
            exp_prefix=exp_prefix,
            variant=variant,
            seed=seed,
            use_cloudpickle=True,
            python_command=' '.join(command_words),
            mode=mode,
            use_gpu=use_gpu,
            script="railrl/scripts/run_experiment_lite.py",
            code_diff=code_diff,
            commit_hash=commit_hash,
            script_name=script_name,
            n_parallel=n_parallel,
            **run_experiment_lite_kwargs
        )


def run_experiment_here(
        experiment_function,
        exp_prefix="default",
        variant=None,
        exp_id=0,
        seed=0,
        use_gpu=True,
        snapshot_mode='last',
        snapshot_gap=1,
        code_diff=None,
        commit_hash=None,
        script_name=None,
        n_parallel=0,
        base_log_dir=None,
        snapshot_dir=None,
):
    """
    Run an experiment locally without any serialization.

    :param experiment_function: Function. `variant` will be passed in as its
    only argument.
    :param exp_prefix: Experiment prefix for the save file.
    :param variant: Dictionary passed in to `experiment_function`.
    :param exp_id: Experiment ID. Should be unique across all
    experiments. Note that one experiment may correspond to multiple seeds,.
    :param seed: Seed used for this experiment.
    :param use_gpu: Run with GPU. By default False.
    :param script_name: Name of the running script
    :return:
    """
    if variant is None:
        variant = {}
    if seed is None and 'seed' not in variant:
        seed = random.randint(0, 100000)
        variant['seed'] = str(seed)
    if n_parallel > 0:
        from rllab.sampler import parallel_sampler
        parallel_sampler.initialize(n_parallel=n_parallel)
        parallel_sampler.set_seed(seed)
    variant['exp_id'] = str(exp_id)
    reset_execution_environment()
    set_seed(seed)
    setup_logger(
        exp_prefix=exp_prefix,
        variant=variant,
        exp_id=exp_id,
        seed=seed,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
        base_log_dir=base_log_dir,
        snapshot_dir=snapshot_dir,
    )
    log_dir = logger.get_snapshot_dir()
    if code_diff is not None:
        with open(osp.join(log_dir, "code.diff"), "w") as f:
            f.write(code_diff)
    if commit_hash is not None:
        with open(osp.join(log_dir, "commit_hash.txt"), "w") as f:
            f.write(commit_hash)
    if script_name is not None:
        with open(osp.join(log_dir, "script_name.txt"), "w") as f:
            f.write(script_name)
    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    else:
        set_gpu_mode(use_gpu)
    return experiment_function(variant)


def create_exp_name(exp_prefix, exp_id=0, seed=0):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_prefix:
    :param exp_id:
    :return:
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    return "%s_%s_%04d--s-%d" % (exp_prefix, timestamp, exp_id, seed)


def create_log_dir(exp_prefix, exp_id=0, seed=0, base_log_dir=None):
    """
    Creates and returns a unique log directory.

    :param exp_prefix: All experiments with this prefix will have log
    directories be under this directory.
    :param exp_id: Different exp_ids will be in different directories.
    :return:
    """
    exp_name = create_exp_name(exp_prefix, exp_id=exp_id,
                               seed=seed)
    if base_log_dir is None:
        base_log_dir = config.LOCAL_LOG_DIR
    log_dir = osp.join(base_log_dir, exp_prefix.replace("_", "-"), exp_name)
    if osp.exists(log_dir):
        raise Exception(
            "Log directory already exists. Will no overwrite: {0}".format(
                log_dir
            )
        )
    os.makedirs(log_dir, exist_ok=True)
    return log_dir, exp_name


def setup_logger(
        exp_prefix="default",
        exp_id=0,
        seed=0,
        variant=None,
        base_log_dir=None,
        text_log_file="debug.log",
        variant_log_file="variant.json",
        tabular_log_file="progress.csv",
        snapshot_mode="last",
        snapshot_gap=1,
        log_tabular_only=False,
        snapshot_dir=None,
):
    """
    Set up logger to have some reasonable default settings.

    Will save log output to

    based_log_dir/exp_prefix/exp_name.

    :param exp_prefix: The sub-directory for this specific experiment.
    :param exp_id: The number of the specific experiment run within this
    experiment.
    :param variant:
    :param base_log_dir: The directory where all log should be saved.
    :param text_log_file:
    :param variant_log_file:
    :param tabular_log_file:
    :param snapshot_mode:
    :param log_tabular_only:
    :param snapshot_gap:
    :return:
    """
    log_dir, exp_name = create_log_dir(exp_prefix, exp_id=exp_id, seed=seed,
                                       base_log_dir=base_log_dir)
    if snapshot_dir is not None:
        log_dir = snapshot_dir
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
    logger.push_prefix("[%s] " % exp_name)


def set_seed(seed):
    """
    Set the seed for all the possible random number generators.

    :param seed:
    :return: None
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def reset_execution_environment():
    """
    Call this between calls to separate experiments.
    :return:
    """
    tf.reset_default_graph()
    import importlib
    importlib.reload(logger)


def create_run_experiment_multiple_seeds(n_seeds, experiment, **kwargs):
    """
    Run a function multiple times over different seeds and return the average
    score.
    :param n_seeds: Number of times to run an experiment.
    :param experiment: A function that returns a score.
    :param kwargs: keyword arguements to pass to experiment.
    :return: Average score across `n_seeds`.
    """
    def run_experiment_with_multiple_seeds(variant):
        seed = int(variant['seed'])
        scores = []
        for i in range(n_seeds):
            variant['seed'] = str(seed + i)
            scores.append(run_experiment(
                experiment,
                variant=variant,
                exp_id=i,
                mode='here',
                **kwargs
            ))
        return np.mean(scores)

    return run_experiment_with_multiple_seeds
