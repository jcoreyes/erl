from railrl.misc.data_processing import Experiment
import matplotlib.pyplot as plt
import numpy as np


her_path = "/home/vitchyr/git/rllab-rail/railrl/data/papers/iclr2018/pusher2d/her-reacher/reacher_curve.npy"
her_data = np.load(her_path)

ddpg_path = "/home/vitchyr/git/rllab-rail/railrl/data/doodads3/10-25-ddpg" \
            "-reacher-pusher-baseline/"
ddpg_criteria = {
    'algo_params.num_updates_per_env_step': 1,
    'algo_params.scale_reward': 1,
    'algo_params.tau': 0.001,
    'env_class.$class': "railrl.envs.multitask.her_reacher_7dof_env.Reacher7Dof"
}
mb_path = "/home/vitchyr/git/rllab-rail/railrl/data/doodads3/10-25-abhishek" \
          "-mb-baseline-pusher-reacher-300-300-net/"
mb_criteria = {
    'env_name_or_class.$class':
        'railrl.envs.multitask.her_reacher_7dof_env.Reacher7Dof',
}
our_path = "/home/vitchyr/git/rllab-rail/railrl/data/doodads3/10-24-sdql-reacher7everything-vtau/"
our_criteria = {
    'raw_explore_policy': 'oc',
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
t1 = our_trials[0]
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
    if name == 'TDM':
        epochs = epochs / 10
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.1)
    plt.plot(epochs, mean, label=name)


# Stop at 200 epochs
her_data = her_data[:, :100]
her_mean = np.mean(her_data, axis=0)
her_std = np.mean(her_data, axis=0)
epochs = 2 * np.arange(0, len(her_mean))
plt.fill_between(epochs, her_mean - her_std, her_mean + her_std, alpha=0.1)
plt.plot(epochs, her_mean, label="HER")
# for run in her_data:
#     plt.plot(epochs, run, label="HER")

# plt.xscale('log')
plt.xlabel("Epoch (1000 steps)")
plt.ylabel("Distance to Goal")
plt.legend()
plt.savefig('results/iclr2018/reacher.jpg')
plt.show()
