from railrl.misc.data_processing import Experiment
import matplotlib.pyplot as plt
import numpy as np

her_path = "/home/vitchyr/git/rllab-rail/railrl/data/papers/iclr2018/pusher2d" \
           "/her-cheetah/cheetah_curve.npy"
her_data = np.load(her_path)

ddpg_path = "/home/vitchyr/git/rllab-rail/railrl/data/doodads3/10-26-ddpg-half-cheetah/"
ddpg_criteria = {
    'algo_params.num_updates_per_env_step': 5,
    'algo_params.scale_reward': 1,
    'algo_params.tau': 0.001,
}
mb_path = "/home/vitchyr/git/rllab-rail/railrl/data/doodads3/10-26-abhishek-mb-cheetah-target-reset/"
mb_criteria = None
our_path = "/home/vitchyr/git/rllab-rail/railrl/data/local/10-26-sdql-cheetah-xvel/10-26-sdql-cheetah-xvel_2017_10_26_16_11_42_0000--s-5011/"
our_criteria = None
her_dense_path = "/home/vitchyr/git/rllab-rail/railrl/data/doodads3/10-27-her-baseline-shaped-rewards-no-clipping-300-300-right-discount-and-tau/"
her_dense_criteria = {
    'algo_params.num_updates_per_env_step': 1,
    'algo_params.scale_reward': 10,
    'env_class.$class':
        "railrl.envs.multitask.half_cheetah.GoalXVelHalfCheetah"
}

ddpg_exp = Experiment(ddpg_path)
mb_exp = Experiment(mb_path)
our_exp = Experiment(our_path)
her_dense_exp = Experiment(her_dense_path)

ddpg_trials = ddpg_exp.get_trials(ddpg_criteria)
mb_trials = mb_exp.get_trials(mb_criteria)
our_trials = our_exp.get_trials(our_criteria)
her_dense_trials = her_dense_exp.get_trials(her_dense_criteria)

print(len(ddpg_trials))
print(len(mb_trials))
print(len(our_trials))

def smooth(data):
    box = np.ones(10) / 10
    new_data = []
    for d in data:
        new_data.append(np.convolve(d, box, mode='same'))
    return np.vstack(new_data)

plt.figure()
for trials, name, key in [
    (ddpg_trials, 'DDPG', 'Final_xvel_errors_Mean'),
    (mb_trials, 'Model Based', 'Final_xvel_errors_Mean'),
    (our_trials, 'TDM', 'test_Multitask_distance_to_goal_Mean'),
    (her_dense_trials, 'HER - dense', 'test_Final_xvel_errors_Mean'),
]:
    all_values = []
    min_len = np.inf
    for trial in trials:
        try:
            values_ts = trial.data[key]
        except:
            import ipdb; ipdb.set_trace()
        min_len = min(min_len, len(values_ts))
        all_values.append(values_ts)
    costs = np.vstack([
        values[:min_len]
        for values in all_values
    ])
    costs = smooth(costs)
    mean = np.mean(costs, axis=0)
    std = np.std(costs, axis=0)
    epochs = np.arange(0, len(costs[0]))
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.1)
    plt.plot(epochs, mean, label=name)


# Stop at 200 epochs
her_data = her_data[:, :100]
her_mean = np.mean(her_data, axis=0)
her_std = np.mean(her_data, axis=0)
epochs = 2 * np.arange(0, len(her_mean))
# plt.fill_between(epochs, her_mean - her_std, her_mean + her_std, alpha=0.1)
# plt.plot(epochs, her_mean, label="HER")
# for run in her_data:
#     plt.plot(epochs, run, label="HER")

# plt.xscale('log')
plt.xlabel("Environment Samples (x1000)")
plt.ylabel("Distance to Goal")
plt.title(r"Distance to Goal vs Environment Samples")
plt.legend()
plt.savefig('results/iclr2018/cheetah.jpg')
plt.show()
