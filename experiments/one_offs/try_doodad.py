import os

import doodad as pd
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.ec2.credentials import AWSCredentials


THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.dirname(THIS_FILE_DIR)

# or this! Run experiment via docker on another machine through SSH
mode_ssh = pd.mode.SSHDocker(
    image='vitchyr/rllab-vitchyr:latest',
    credentials=ssh.SSHCredentials(hostname='newton4.banatao.berkeley.edu',
                                   username='rail', identity_file='~/.ssh/rail_lab_0617'),
)

mode_local = pd.mode.LocalDocker(
    image='vitchyr/rllab-vitchyr:latest',
)

credentials = AWSCredentials(from_env=True)
mode_ec2 = pd.mode.EC2SpotDocker(
    credentials,
    image='vitchyr/rllab-vitchyr:latest',
    region='us-west-1',
    instance_type='c4.large',
    spot_price=0.03,
    s3_bucket="2-12-2017.railrl.vitchyr.rail.bucket",
    terminate=True,
    image_id="ami-ad81c8cd",
    aws_key_name="rllab-vitchyr-us-west-1",
    iam_instance_profile_name='rllab',
    s3_log_prefix='custom_experiment',
)

# Set up code and output directories
OUTPUT_DIR = '/mount/outputs'  # this is the directory visible to the target
mounts = [
    # mount.MountLocal(local_dir=REPO_DIR, pythonpath=True),
    # mount.MountLocal(local_dir='~/install/rllab', pythonpath=True),  # rllab
    # mount.MountLocal(local_dir='~/install/gym/.mujoco', mount_point='/root/.mujoco'),  # mujoco
    # mount.MountLocal(local_dir='~/code/doodad', pythonpath=True),  # rllab
    # mount.MountLocal(local_dir='~/data/vitchyr', mount_point=OUTPUT_DIR,
    #                  read_only=False),  # mujoco
    mount.MountS3(
        s3_path="test",
        s3_bucket="2-12-2017.railrl.vitchyr.rail.bucket",
        mount_point=OUTPUT_DIR,
        output=True,
    )
]

script_path = os.path.join(THIS_FILE_DIR, 'test_script.py')
print(script_path)
pd.launch_python(
    target=script_path,
    # script. If running remotely, this will be copied over
    #target='/media/sg2/scripts/swimmer_data.py',
    mode=mode_ec2,
    # mode=mode_ssh,
    # mode=mode_local,
    mount_points=mounts,
    args={'output_dir': OUTPUT_DIR}
)