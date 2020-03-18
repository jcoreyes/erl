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
        **kwargs
):
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
