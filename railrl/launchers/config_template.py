# Change these things
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
# If not set, default will be chosen by doodad
# AWS_S3_PATH = 's3://bucket/directory

AWS_S3_PATH = 's3://2-12-2017.railrl.vitchyr.rail.bucket/doodad/logs-12-01-2017'

# You probably don't need to change things below
# Specifically, the docker image is looked up on dockerhub.com.
DOODAD_DOCKER_IMAGE = 'vitchyr/railrl-vitchyr'
INSTANCE_TYPE = 'c4.large'
SPOT_PRICE = 0.03

GPU_DOODAD_DOCKER_IMAGE = 'vitchyr/railrl-vitchyr-gpu'
GPU_INSTANCE_TYPE = 'g2.2xlarge'
GPU_SPOT_PRICE = 0.5
GPU_AWS_IMAGE_ID = "ami-874378e7"

# This really shouldn't matter and in theory could be whatever
OUTPUT_DIR_FOR_DOODAD_TARGET = '/tmp/doodad-output/'
