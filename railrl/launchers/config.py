CODE_DIRS_TO_MOUNT = [
    '/home/vitchyr/git/rllab-rail/railrl/',
    '/home/vitchyr/git/rllab-rail/',
    '/home/vitchyr/git/hw4_fall2017/',
]
DIR_AND_MOUNT_POINT_MAPPINGS = [
    dict(
        local_dir='/home/vitchyr/.mujoco/',
        mount_point='/root/.mujoco',
    ),
]
LOCAL_LOG_DIR = '/home/vitchyr/git/rllab-rail/railrl/data/local/'
RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = (
    '/home/vitchyr/git/rllab-rail/railrl/scripts/run_experiment_from_doodad.py'
)
DOODAD_DOCKER_IMAGE = 'vitchyr/rllab-vitchyr'

# This really shouldn't matter and in theory could be whatever
OUTPUT_DIR_FOR_DOODAD_TARGET = '/tmp/doodad-output/'
# OUTPUT_DIR_FOR_DOODAD_TARGET = '/tmp/dir/from/railrl-config/'
