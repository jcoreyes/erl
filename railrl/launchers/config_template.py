CODE_DIRS_TO_MOUNT = [
    '/home/user/python/module/one',
    '/home/user/python/module/two',
]
DIR_AND_MOUNT_POINT_MAPPINGS = [
    dict(
        local_dir='/home/user/.mujoco/',
        mount_point='/root/.mujoco',
    ),
]
LOCAL_LOG_DIR = '/home/user/git/path/to/save/data/'
RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = (
    '/home/user/path/to/railrl/scripts/run_experiment_from_doodad.py'
)
DOODAD_DOCKER_IMAGE = 'vitchyr/rllab-vitchyr'

# This really shouldn't matter and in theory could be whatever
OUTPUT_DIR_FOR_DOODAD_TARGET = '/tmp/doodad-output/'
