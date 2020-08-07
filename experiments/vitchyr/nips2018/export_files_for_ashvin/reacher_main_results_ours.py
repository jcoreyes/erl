import os
from rlkit.misc.data_processing import get_trials
from subprocess import call


base_dir = '/home/vitchyr/git/railrl/data/papers/nips2018/for-ashvin' \
           '/reacher-main-results-ours'


trials = get_trials(
    '/home/vitchyr/git/railrl/data/papers/nips2018/reacher-abalation-resample-strategy/',
    criteria={
        'replay_kwargs.fraction_resampled_goals_are_env_goals': 0.5,
        'replay_kwargs.fraction_goals_are_rollout_goals': 0.2,
        'algo_kwargs.num_updates_per_env_step': 5,
    }
)
output_dir = base_dir

os.makedirs(output_dir, exist_ok=True)
print("Making dir", output_dir)
for trial in trials:
    dir = trial[2]
    print("cp -r {} {}".format(dir, output_dir))
    call(["cp", "-r", dir, output_dir])
