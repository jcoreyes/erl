from railrl.launchers.launcher_util import run_experiment_doodad


def run_task(variant):
    from rllab.misc import logger
    print(variant)
    logger.log("Hello from script")
    logger.log("variant: " + str(variant))

run_experiment_doodad(
    run_task,
    mode='local_docker',
    exp_prefix='test-doodad',
    variant=dict(
        test=2
    ),
)