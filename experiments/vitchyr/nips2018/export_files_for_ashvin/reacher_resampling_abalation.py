import os
import shutil
from rlkit.misc.data_processing import get_trials
from subprocess import call


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
base_dir = '/home/vitchyr/git/railrl/data/papers/nips2018/for-ashvin' \
             '/reacher-abalation-resample-strategy'


trials = get_trials(
    '/home/vitchyr/git/railrl/data/papers/nips2018/reacher-abalation-resample-strategy/',
    criteria={
        'replay_kwargs.fraction_resampled_goals_are_env_goals': 0.5,
        'replay_kwargs.fraction_goals_are_rollout_goals': 0.2,
        'algo_kwargs.num_updates_per_env_step': 5,
    }
)
output_dir = base_dir + "/rollout-0.2--env-goals-0.5"

os.makedirs(output_dir, exist_ok=True)
print("Making dir", output_dir)
for trial in trials:
    dir = trial[2]
    print("cp -r {} {}".format(dir, output_dir))
    call(["cp", "-r", dir, output_dir])


trials = get_trials(
    '/home/vitchyr/git/railrl/data/papers/nips2018/reacher-abalation-resample-strategy/',
    criteria={
        'replay_kwargs.fraction_resampled_goals_are_env_goals': 1.0,
        'replay_kwargs.fraction_goals_are_rollout_goals': 0.2,
        'algo_kwargs.num_updates_per_env_step': 5,
    }
)
output_dir = base_dir + "/rollout-0.2--env-goals-1.0"

os.makedirs(output_dir, exist_ok=True)
print("Making dir", output_dir)
for trial in trials:
    dir = trial[2]
    print("cp -r {} {}".format(dir, output_dir))
    call(["cp", "-r", dir, output_dir])


trials = get_trials(
    '/home/vitchyr/git/railrl/data/papers/nips2018/reacher-abalation-resample-strategy/',
    criteria={
        'replay_kwargs.fraction_resampled_goals_are_env_goals': 0.0,
        'replay_kwargs.fraction_goals_are_rollout_goals': 0.2,
        'algo_kwargs.num_updates_per_env_step': 5,
    }
)
output_dir = base_dir + "/rollout-0.2--env-goals-0.0"

os.makedirs(output_dir, exist_ok=True)
print("Making dir", output_dir)
for trial in trials:
    dir = trial[2]
    print("cp -r {} {}".format(dir, output_dir))
    call(["cp", "-r", dir, output_dir])


trials = get_trials(
    '/home/vitchyr/git/railrl/data/papers/nips2018/reacher-abalation-resample-strategy/',
    criteria={
        'replay_kwargs.fraction_resampled_goals_are_env_goals': 0.0,
        'replay_kwargs.fraction_goals_are_rollout_goals': 1.0,
        'algo_kwargs.num_updates_per_env_step': 5,
    }
)
output_dir = base_dir + "/rollout-0.0--env-goals-0.0"

os.makedirs(output_dir, exist_ok=True)
print("Making dir", output_dir)
for trial in trials:
    dir = trial[2]
    print("cp -r {} {}".format(dir, output_dir))
    call(["cp", "-r", dir, output_dir])
