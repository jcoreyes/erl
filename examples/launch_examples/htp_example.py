"""
Example of a script to generate a script that can be run on slurm + singularity.
"""
import time

from railrl.core import logger
from railrl.launchers.launcher_util import run_experiment
from datetime import datetime
from pytz import timezone
import pytz


def example(variant):
    import platform
    logger.log("python {}".format(platform.python_version()))
    logger.log("platform path {}".format(platform.__file__))
    import torch
    logger.log("torch version {}".format(torch.__version__))
    import mujoco_py
    logger.log("mujoco version {}".format(mujoco_py.__version__))
    import gym
    logger.log("gym version {}".format(gym.__version__))
    date_format = '%m/%d/%Y %H:%M:%S %Z'
    date = datetime.now(tz=pytz.utc)
    logger.log("start")
    logger.log('Current date & time is: {}'.format(date.strftime(date_format)))
    logger.log('torch.cuda.is_available() {}'.format(
        torch.cuda.is_available()
    ))
    if torch.cuda.is_available():
        x = torch.randn(3)
        logger.log(str(x.cuda()))

    date = date.astimezone(timezone('US/Pacific'))
    logger.log('Local date & time is: {}'.format(date.strftime(date_format)))
    for i in range(variant['num_seconds']):
        logger.log("Tick, {}".format(i))
        time.sleep(1)
    logger.log("end")
    logger.log('Local date & time is: {}'.format(date.strftime(date_format)))

    logger.log("start mujoco")
    from gym.envs.mujoco import HalfCheetahEnv
    e = HalfCheetahEnv()
    img = e.sim.render(32, 32)
    logger.log(str(sum(img)))
    logger.log("end mujoco")


if __name__ == "__main__":
    # noinspection PyTypeChecker
    date_format = '%m/%d/%Y %H:%M:%S %Z'
    date = datetime.now(tz=pytz.utc)
    logger.log("start")
    variant = dict(
        num_seconds=10,
        launch_time=str(date.strftime(date_format)),
    )
    for _ in range(5):
        run_experiment(
            example,
            exp_prefix='htp-mode-example',
            mode='htp',
            variant=variant,
            use_gpu=False,
            time_in_mins=20,
        )
