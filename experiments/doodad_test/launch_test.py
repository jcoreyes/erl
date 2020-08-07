import argparse
from rlkit.launchers.launcher_util import run_experiment


def run_task(variant):
    from rlkit.core import logger
    print(variant)
    logger.log("Hello from script")
    logger.log("variant: " + str(variant))
    logger.record_tabular("value", 1)
    logger.dump_tabular()
    logger.log("snapshot_dir:", logger.get_snapshot_dir())

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='local')
args = parser.parse_args()

run_experiment(
    run_task,
    mode=args.mode,
    exp_prefix='test-doodad-launch-test-script',
    variant=dict(
        test=2
    ),
)
