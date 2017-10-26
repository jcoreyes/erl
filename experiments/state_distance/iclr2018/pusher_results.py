from railrl.misc.data_processing import Experiment
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ddpg_path = "/home/vitchyr/git/rllab-rail/railrl/data/doodads3/10-25-ddpg-pusher-again-baseline-with-reward-bonus/"
ddpg_criteria = {
    'algo_params.num_updates_per_env_step': 5,
    'algo_params.scale_reward': 1,
    'algo_params.tau': 0.01,
}
mb_path = "/home/vitchyr/git/rllab-rail/railrl/data/doodads3/10-25-abhishek-mb-baseline-pusher-again-shaped/"
mb_criteria = None
our_path = "/home/vitchyr/git/rllab-rail/railrl/data/doodads3/10-25-get-results-take1/"
our_criteria = {
    'env_class.$class':
        "railrl.envs.multitask.pusher2d.HandCylinderXYPusher2DEnv",
    'algo_params.goal_dim_weights': [1, 1, 1, 1],
    'epoch_discount_schedule_params.value': 10,
}

ddpg_exp = Experiment(ddpg_path)
mb_exp = Experiment(mb_path)
our_exp = Experiment(our_path)

ddpg_trials = ddpg_exp.get_trials(ddpg_criteria)
mb_trials = mb_exp.get_trials(mb_criteria)
our_trials = our_exp.get_trials(our_criteria)

print(len(ddpg_trials))
print(len(mb_trials))
print(len(our_trials))

key = 'Final_Euclidean_distance_to_goal_Mean'
plt.figure()
for trials, name in [
    (ddpg_trials, 'DDPG'),
    (mb_trials, 'Model Based'),
    (our_trials, 'TDM'),
]:
    all_values = []
    min_len = np.inf
    for trial in trials:
        values_ts = trial.data[key]
        min_len = min(min_len, len(values_ts))
        all_values.append(values_ts)
    costs = np.vstack([
        values[:min_len]
        for values in all_values
    ])
    mean = np.mean(costs, axis=0)
    std = np.std(costs, axis=0)
    epochs = np.arange(0, len(costs[0]))
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.1)
    plt.plot(epochs, mean, label=name)


# plt.xscale('log')
plt.xlabel("Epoch (1000 steps)")
plt.ylabel("Distance to Goal")
plt.legend()
plt.savefig('results/iclr2018/pusher.jpg')
plt.show()
