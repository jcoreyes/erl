from railrl.launchers.launcher_util import run_experiment_doodad


def run_task(variant):
    from rllab.misc import logger
    print(variant)
    logger.log("Hello from script")

run_experiment_doodad(
    run_task,
    mode='ec2',
    variant=dict(
        test=2
    ),
)