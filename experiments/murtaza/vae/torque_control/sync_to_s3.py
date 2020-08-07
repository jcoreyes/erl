import os
import sys

from rlkit.launchers import config

file_path = sys.argv[1]
s3_path = config.AWS_S3_PATH
os.system("aws s3 sync " + file_path + ' ' + s3_path + file_path)
