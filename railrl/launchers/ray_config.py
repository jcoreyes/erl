DIR_AND_MOUNT_POINT_MAPPINGS = [
    dict(
        local_dir='/home/steven/.mujoco',
        remote_dir='/home/ubuntu/.mujoco',
        mount_point='/root/.mujoco',
    ),
    dict(
        local_dir='/home/steven/res/railrl-private',
        remote_dir='/home/ubuntu/res/railrl-private',
        mount_point='/home/steven/res/railrl-private',
    ),
    dict(
        local_dir='/home/steven/res/multiworld',
        remote_dir='/home/ubuntu/res/multiworld',
        mount_point='/home/steven/res/multiworld',
    ),
    # dict(
        # local_dir='/home/steven/res/rllab-curriculum',
        # remote_dir='/home/ubuntu/res/rllab-curriculum',
        # mount_point='/home/steven/res/rllab-curriculum',
    # ),
    dict(
        local_dir='/tmp/local_exp.pkl',
        remote_dir='/home/ubuntu/local_exp.pkl',
        mount_point='/tmp/local_exp.pkl',
    ),
]
# This can basically be anything. Used for launching on instances. The
# local launch parameters (exo_func, exp_variant, etc) are saved at this location
# on the local machine and then transfered to the remote machine.
EXPERIMENT_INFO_PKL_FILEPATH = '/tmp/local_exp.pkl'
# Again, can be anything. The Ray autoscaler yaml file is saved to this location
# before launching.
LAUNCH_FILEPATH = '/tmp/autoscaler_launch.yaml'

LOCAL_LOG_DIR = '/home/steven/logs'
RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = (
    '/home/steven/res/railrl-private/scripts/run_experiment_from_doodad.py'
)

AWS_CONFIG_NO_GPU=dict(
    REGION='us-west-2',
    INSTANCE_TYPE = 'g2.2xlarge',
    SPOT_PRICE = 0.2,
    REGION_TO_AWS_IMAGE_ID = {
        'us-east-1': "ami-ce73adb1",
        # 'us-west-2': 'ami-0eb3c807c9088a837'
        'us-west-2': 'ami-0d5bb58171e5325a8'
    },
    REGION_TO_AWS_AVAIL_ZONE = {
        'us-east-1': "us-east-1a",
        'us-west-2': 'us-west-2a,us-west-2b'
    },

)

AWS_CONFIG_GPU = dict(
    # REGION='us-east-1',
    REGION='us-west-2',
    INSTANCE_TYPE = 'g3.4xlarge',
    SPOT_PRICE = 0.6,
    REGION_TO_AWS_IMAGE_ID = {
        'us-east-1': "ami-ce73adb1",
        # 'us-west-2': 'ami-0b294f219d14e6a82'
        'us-west-2': 'ami-0d5bb58171e5325a8'
    },
    REGION_TO_AWS_AVAIL_ZONE = {
        'us-east-1': "us-east-1a",
        'us-west-2': 'us-west-2a,us-west-2b'
    },
)

GCP_CONFIG_GPU = dict(
    ZONE='us-west2-c',
    INSTANCE_TYPE='n1-highmem-8',
    IMAGE_PROJECT='railrl-private-gcp',
    PROJECT_ID='railrl-private-gcp',
    gpu_kwargs=dict(
        num_gpu=1,
    )
)


AWS_CONFIG = {
    True: AWS_CONFIG_GPU,
    False: AWS_CONFIG_NO_GPU,
}
GCP_CONFIG = {
    True: GCP_CONFIG_GPU,
    False: GCP_CONFIG_GPU,
}

DOODAD_DOCKER_IMAGE = 'vitchyr/railrl-torch4cuda9'
# If not set, default will be chosen by doodad
# AWS_S3_PATH = 's3://bucket/directory
GPU_DOODAD_DOCKER_IMAGE = 'stevenlin598/ray_railrl'
DOCKER_IMAGE = {
    True: GPU_DOODAD_DOCKER_IMAGE,
    False: GPU_DOODAD_DOCKER_IMAGE
}


# You probably don't need to change things below
# Specifically, the docker image is looked up on dockerhub.com.
GPU_SINGULARITY_IMAGE = None
# GPU_DOODAD_DOCKER_IMAGE = 'mdalal/railrl_auto_goal_gan_v2'
#ec2 image
# GPU_DOODAD_DOCKER_IMAGE = 'vitchyr/railrl-torch4cuda9'
# These AMI images have the docker images already installed.

# BEGIN GCP
GCP_IMAGE_NAME = 'railrl-torch-4-cpu'
# GCP_GPU_IMAGE_NAME = 'railrl-torch4cuda9'
GCP_GPU_IMAGE_NAME = 'stevenjqcuda9nvidia396'
GCP_BUCKET_NAME='railrl-steven'
AWS_S3_PATH = 's3://steven.railrl/ray'

LOG_BUCKET = 's3://steven.railrl/ray'

GCP_PREEMPTION_BUCKET_NAME='preemptible-restarts'



# This really shouldn't matter and in theory could be whatever
OUTPUT_DIR_FOR_DOODAD_TARGET = '/tmp/doodad-output/'
SSH_HOSTS = {'':''}
SSH_DEFAULT_HOST = ""


