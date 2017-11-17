from railrl.misc.data_processing import Experiment
import matplotlib.pyplot as plt
import numpy as np

her_path = "/home/vitchyr/git/rllab-rail/railrl/data/papers/iclr2018/pusher2d/her-pusher/pusher_curve.npy"
her_data = np.load(her_path)

ddpg_path = "/home/vitchyr/git/rllab-rail/railrl/data/doodads3/10-25-ddpg-pusher-again-baseline-with-reward-bonus/"
ddpg_criteria = {
    'algo_params.num_updates_per_env_step': 5,
    'algo_params.scale_reward': 1,
    'algo_params.tau': 0.01,
}
# DDPG policy: /home/vitchyr/git/rllab-rail/railrl/data/doodads3/10-25-ddpg-pusher-again-baseline-with-reward-bonus/10-25-ddpg-pusher-again-baseline-with-reward-bonus-id0-s362488/
mb_path = "/home/vitchyr/git/rllab-rail/railrl/data/doodads3/10-25-abhishek-mb-baseline-pusher-again-shaped/"
mb_criteria = None
our_path = "/home/vitchyr/git/rllab-rail/railrl/data/doodads3/11-02-get-results-handxyxy-small-sweep"
our_criteria = {
    'algo_params.num_updates_per_env_step': 5,
    'epoch_discount_schedule_params.value': 5,
    'algo_params.tau': 0.001,
}

her_dense_path = "/home/vitchyr/git/rllab-rail/railrl/data/doodads3/10-27-her-baseline-shaped-rewards-no-clipping-300-300-right-discount-and-tau/"
her_dense_criteria = {
    'algo_params.num_updates_per_env_step': 1,
    'algo_params.scale_reward': 1,
    'env_class.$class':
        "railrl.envs.multitask.pusher2d.CylinderXYPusher2DEnv"
}

ddpg_exp = Experiment(ddpg_path)
mb_exp = Experiment(mb_path)
our_exp = Experiment(our_path)
her_dense_exp = Experiment(her_dense_path)

ddpg_trials = ddpg_exp.get_trials(ddpg_criteria)
mb_trials = mb_exp.get_trials(mb_criteria)
our_trials = our_exp.get_trials(our_criteria)
her_dense_trials = her_dense_exp.get_trials(her_dense_criteria)
MAX_ITERS = 100

base_key = 'Final_Euclidean_distance_to_goal_Mean'
plt.figure()
for trials, name, key in [
    (ddpg_trials, 'DDPG', base_key),
    (mb_trials, 'Model Based', base_key),
    (our_trials, 'TDM', 'test_'+base_key),
    (her_dense_trials, 'HER - Dense', 'test_'+base_key),
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
    costs = costs[:, :min(costs.shape[1], MAX_ITERS)]
    mean = np.mean(costs, axis=0)
    std = np.std(costs, axis=0)
    epochs = np.arange(0, len(costs[0]))
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.1)
    plt.plot(epochs, mean, label=name)


# Stop at 200 epochs
her_data = her_data[:, :50]
her_mean = np.mean(her_data, axis=0)
her_std = np.mean(her_data, axis=0)
epochs = 2 * np.arange(0, len(her_mean))
# plt.fill_between(epochs, her_mean - her_std, her_mean + her_std, alpha=0.1)
# plt.plot(epochs, her_mean, label="HER - Sparse")
# for run in her_data:
#     plt.plot(epochs, run, label="HER - Sparse")

# plt.xscale('log')
plt.xlabel("Environment Samples (x1000)")
plt.ylabel("Final Distance to Goal")
# plt.title(r"Pusher: Distance to Goal vs Environment Samples")
plt.legend()
plt.savefig('results/iclr2018/pusher.jpg')
plt.show()
