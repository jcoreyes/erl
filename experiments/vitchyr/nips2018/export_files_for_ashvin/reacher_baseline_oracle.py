import os
import shutil
from rlkit.misc.data_processing import get_trials
from subprocess import call


base_dir = '/home/vitchyr/git/railrl/data/papers/nips2018/for-ashvin' \
             '/reacher-baseline-oracle'


trials = get_trials(
    '/home/vitchyr/git/railrl/data/doodads3/05-16-paper-reacher-results-full-state-oracle-ish/',
    criteria={
        'replay_buffer_kwargs.fraction_resampled_goals_are_env_goals': 1.0,
        'algo_kwargs.num_updates_per_env_step': 5,
        'exploration_type': 'epsilon',
    }
)
output_dir = base_dir

os.makedirs(output_dir, exist_ok=True)
print("Making dir", output_dir)
for trial in trials:
    dir = trial[2]
    print("cp -r {} {}".format(dir, output_dir))
    call(["cp", "-r", dir, output_dir])
