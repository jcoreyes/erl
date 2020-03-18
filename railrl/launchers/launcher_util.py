import os
from typing import NamedTuple
import random

import __main__ as main
import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.core import logger, setup_logger
from railrl.launchers import config


GitInfo = NamedTuple(
    'GitInfo',
    [
        ('directory', str),
        ('code_diff', str),
        ('code_diff_staged', str),
        ('commit_hash', str),
        ('branch_name', str),
    ],
)


def run_experiment(
        method_call,
        exp_name='default',
        mode='local',
        variant=None,
        use_gpu=False,
        gpu_id=0,
        wrap_fn_with_auto_setup=True,
        unpack_variant=True,
        **kwargs
):
    if wrap_fn_with_auto_setup:
        method_call = auto_setup(method_call, unpack_variant=unpack_variant)
    if mode == 'here_no_doodad':
        setup_experiment(
            variant=variant,
            exp_name=exp_name,
            base_log_dir=config.LOCAL_LOG_DIR,
            git_infos=generate_git_infos(),
            script_name=main.__file__,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
        )
        method_call(None, variant)
    else:
        from doodad.easy_launch.python_function import (
            run_experiment as doodad_run_experiment
        )
        doodad_run_experiment(
            method_call,
            exp_name=exp_name,
            mode=mode,
            variant=variant,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
            **kwargs
        )


def auto_setup(exp_function, unpack_variant=True):
    """
    Automatically set up:
    1. the logger
    2. the GPU mode
    3. the seed

    :param exp_function: some function that should not depend on `logger_config`
    nor `seed`.
    :param unpack_variant: do you call exp_function with `**variant`?
    nor `seed`.
    :return: function output
    """
    def run_experiment_compatible_function(doodad_config, variant):
        # See doodad.easy_launch.python_function.DoodadConfig
        if doodad_config:
            variant_to_save = variant.copy()
            variant_to_save['doodad_info'] = doodad_config.extra_launch_info
            setup_experiment(
                variant=variant_to_save,
                exp_name=doodad_config.exp_name,
                base_log_dir=doodad_config.base_log_dir,
                git_infos=doodad_config.git_infos,
                script_name=doodad_config.script_name,
                use_gpu=doodad_config.use_gpu,
                gpu_id=doodad_config.gpu_id,
            )
        variant.pop('logger_config', None)
        variant.pop('seed', None)
        if unpack_variant:
            exp_function(**variant)
        else:
            exp_function(variant)

    return run_experiment_compatible_function


def setup_experiment(
        variant,
        exp_name,
        base_log_dir,
        git_infos,
        script_name,
        use_gpu,
        gpu_id,
):
    logger_config = variant.get('logger_config', {})
    seed = variant.get('seed', random.randint(0, 99999))
    set_seed(seed)
    ptu.set_gpu_mode(use_gpu, gpu_id)
    os.environ['gpu_id'] = str(gpu_id)
    setup_logger(
        logger,
        exp_name=exp_name,
        base_log_dir=base_log_dir,
        variant=variant,
        git_infos=git_infos,
        script_name=script_name,
        **logger_config)


def set_seed(seed):
    """
    Set the seed for all the possible random number generators.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)


def generate_git_infos():
    try:
        import git
        dirs = config.CODE_DIRS_TO_MOUNT

        git_infos = []
        for directory in dirs:
            # Idk how to query these things, so I'm just doing try-catch
            try:
                repo = git.Repo(directory)
                try:
                    branch_name = repo.active_branch.name
                except TypeError:
                    branch_name = '[DETACHED]'
                git_infos.append(GitInfo(
                    directory=directory,
                    code_diff=repo.git.diff(None),
                    code_diff_staged=repo.git.diff('--staged'),
                    commit_hash=repo.head.commit.hexsha,
                    branch_name=branch_name,
                ))
            except git.exc.InvalidGitRepositoryError:
                pass
    except (ImportError, UnboundLocalError, NameError):
        git_infos = None
    return git_infos
