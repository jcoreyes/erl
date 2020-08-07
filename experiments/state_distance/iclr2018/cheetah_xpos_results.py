from rlkit.misc.data_processing import Experiment
import matplotlib.pyplot as plt
import numpy as np

from rlkit.misc.visualization_util import sliding_mean

mb_trials = Experiment(
    "/home/vitchyr/git/railrl/data/doodads3/12-30-cheetah-xpos-increase-distance/",
    criteria={
        'algorithm': 'Model-Based-Dagger',
        'env_kwargs.max_distance': 40,
    },
).get_trials()
ddpg_trials = Experiment(
    "/home/vitchyr/git/railrl/data/doodads3/12-30-cheetah-xpos-increase-distance/",
    criteria={
        'algorithm': 'DDPG',
        'env_kwargs.max_distance': 40,
        'exp_id': '10',
    },
).get_trials()

MAX_ITERS = 10000
plt.figure()
base_key = 'Final_Distance_to_goal_Mean'
for trials, name, key in [
    (mb_trials, 'Model Based', base_key),
    (ddpg_trials, 'DDPG', base_key),
]:
    all_values = []
    min_len = np.inf
    if len(trials) == 0:
        print(name)
        import ipdb; ipdb.set_trace()
    for trial in trials:
        try:
            values_ts = trial.data[key]
        except:
            import ipdb; ipdb.set_trace()
        min_len = min(min_len, len(values_ts))
        values_ts = sliding_mean(values_ts, window=10)
        all_values.append(values_ts)
    costs = np.vstack([
        values[:min_len]
        for values in all_values
    ])
    costs = costs[:, :min(costs.shape[1], MAX_ITERS)]
    # if name != 'TDM':
    # costs = smooth(costs)
    mean = np.mean(costs, axis=0)
    std = np.std(costs, axis=0)
    epochs = np.arange(0, len(costs[0]))
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.1)
    plt.plot(epochs, mean, label=name)


# plt.xscale('log')
plt.xlabel("Environment Samples (x1,000)")
plt.ylabel("Velocity Error")
# plt.title(r"Half Cheetah: Velocity Error vs Environment Samples")
plt.legend()
plt.savefig('results/iclr2018/cheetah-xpos.jpg')
plt.show()
