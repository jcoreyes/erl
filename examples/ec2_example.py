"""
Example of running stuff on EC2
"""
import time

from railrl.launchers.launcher_util import run_experiment
from rllab.misc import logger
from datetime import datetime
from pytz import timezone
import pytz


def example(variant):
    import torch
    logger.log(torch.__version__)
    date_format = '%m/%d/%Y %H:%M:%S %Z'
    date = datetime.now(tz=pytz.utc)
    logger.log("start")
    logger.log('Current date & time is: {}'.format(date.strftime(date_format)))
    x = torch.randn(3)
    logger.log(str(x.cuda()))

    date = date.astimezone(timezone('US/Pacific'))
    logger.log('Local date & time is: {}'.format(date.strftime(date_format)))
    for i in range(variant['num_seconds']):
        logger.log("Tick, {}".format(i))
        time.sleep(1)
    logger.log("end")
    logger.log('Local date & time is: {}'.format(date.strftime(date_format)))


if __name__ == "__main__":
    # noinspection PyTypeChecker
    date_format = '%m/%d/%Y %H:%M:%S %Z'
    date = datetime.now(tz=pytz.utc)
    logger.log("start")
    variant = dict(
        num_seconds=10000,
        launch_time=str(date.strftime(date_format)),
    )
    run_experiment(
        example,
        # exp_prefix="ec2-ami-dc4e75bc-gpu-short-3",
        exp_prefix="ec2-ami-874378e7-gpu-longer-4",
        mode='ec2',
        variant=variant,
        use_gpu=True,
    )
