import boto3
from subprocess import Popen
from multiprocessing import Process

from railrl.launchers.config import LOCAL_LOG_DIR
import os

def sync_down(path, check_exists=True):
    local_path = "%s/%s" % (LOCAL_LOG_DIR, path)

    if check_exists and os.path.isfile(local_path):
        return local_path

    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    is_docker = os.path.isfile("/.dockerenv")
    if is_docker:
        from doodad.ec2.autoconfig import AUTOCONFIG
        os.environ["AWS_ACCESS_KEY_ID"] = AUTOCONFIG.aws_access_key()
        os.environ["AWS_SECRET_ACCESS_KEY"] = AUTOCONFIG.aws_access_secret()
        aws = "/env/bin/aws"
    else:
        aws = "aws"

    cmd = "%s s3 cp s3://s3doodad/doodad/logs/%s %s" % (aws, path, local_path)
    cmd_list = cmd.split(" ")
    try:
        p = Popen(cmd_list).wait()
    except:
        print("Failed to sync!....", path)
    return local_path

if __name__ == "__main__":
    p = sync_down("ashvin/vae/new-point2d/run0/id1/params.pkl")
    print("got", p)
