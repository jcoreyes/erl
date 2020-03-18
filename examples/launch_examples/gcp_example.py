"""
Example of running stuff on EC2
"""
import time
from datetime import datetime

import pytz
import torch
from pytz import timezone

import railrl.torch.pytorch_util as ptu
from railrl.core import logger
from railrl.launchers.doodad_util import auto_setup
from railrl.launchers.launcher_util import run_experiment


def example(num_seconds, launch_time):
    logger.log(torch.__version__)
    date_format = '%m/%d/%Y %H:%M:%S %Z'
    date = datetime.now(tz=pytz.utc)
    logger.log("start")
    logger.log('Saved launch time {}'.format(launch_time))
    logger.log('Current date & time is: {}'.format(date.strftime(date_format)))
    if torch.cuda.is_available():
        x = torch.randn(3)
        logger.log(str(x.to(ptu.device)))

    date = date.astimezone(timezone('US/Pacific'))
    logger.log('Local date & time is: {}'.format(date.strftime(date_format)))
    for i in range(num_seconds):
        logger.log("Tick, {}".format(i))
        time.sleep(1)
    logger.log("end")
    logger.log('Local date & time is: {}'.format(date.strftime(date_format)))

    logger.log("start mujoco")
    from gym.envs.mujoco import HalfCheetahEnv
    e = HalfCheetahEnv()
    img = e.sim.render(32, 32)
    logger.log(str(sum(img)))
    logger.log("end mujoco_py")


if __name__ == "__main__":
    # noinspection PyTypeChecker
    date_format = '%m/%d/%Y %H:%M:%S %Z'
    date = datetime.now(tz=pytz.utc)
    for seed in range(5):
        variant = dict(
            num_seconds=10,
            launch_time=str(date.strftime(date_format)),
            logger_config=dict(
            ),
            seed=seed,
        )
        run_experiment(
            auto_setup(example),
            exp_name='gcp-doodad-easy-launch-example',
            mode='gcp',
            variant=variant,
            use_gpu=False,
        )
