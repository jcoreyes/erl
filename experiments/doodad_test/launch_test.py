import argparse
from railrl.launchers.launcher_util import run_experiment


def run_task(variant):
    from rllab.misc import logger
    print(variant)
    logger.log("Hello from script")
    logger.log("variant: " + str(variant))

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='local')
args = parser.parse_args()

run_experiment(
    run_task,
    mode=args.mode,
    exp_prefix='test-doodad',
    variant=dict(
        test=2
    ),
)