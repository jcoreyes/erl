"""
See start_experiment.py to see what to do first

Run

$ python this_script.py path/to/snapshot/dir
"""
import argparse
from rlkit.launchers.launcher_util import (
    continue_experiment_simple,
    resume_torch_algorithm_simple,
)

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()

continue_experiment_simple(args.path, resume_torch_algorithm_simple)