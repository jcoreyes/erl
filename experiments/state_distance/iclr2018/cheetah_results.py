from railrl.misc.data_processing import Experiment
import matplotlib.pyplot as plt
import numpy as np

mb_trials = Experiment(
    "/home/vitchyr/git/railrl/data/doodads3/12-13-abhishek-mb-cheetah-max-vel5/"
).get_trials()
ddpg_trials = Experiment(
    "/home/vitchyr/git/railrl/data/doodads3/12-12-tdm-half-cheetah-short-epoch-nupo-sweep/"
).get_trials({
    'exp_id': '5',
    'algorithm': 'DDPG',
})
tdm_trials = Experiment(
    "/home/vitchyr/git/railrl/data/doodads3/12-12-tdm-half-cheetah-short-epoch-nupo-sweep/"
).get_trials({
    'exp_id': '8',
    'algorithm': 'DDPG-TDM',
})

MAX_ITERS = 150
plt.figure()
for trials, name, key in [
    (mb_trials, 'Model Based', 'Final_xvel_errors_Mean'),
    (ddpg_trials, 'DDPG', 'Final_xvel_errors_Mean'),
    (tdm_trials, 'TDM', 'Final_xvel_errors_Mean'),
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
    costs = costs[:, :min(costs.shape[1], MAX_ITERS)]
    # if name != 'TDM':
    # costs = smooth(costs)
    mean = np.mean(costs, axis=0)
    std = np.std(costs, axis=0)
    epochs = np.arange(0, len(costs[0]))
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.1)
    plt.plot(epochs, mean, label=name)


# plt.xscale('log')
plt.xlabel("Environment Samples (x1000)")
plt.ylabel("Velocity Error")
# plt.title(r"Half Cheetah: Velocity Error vs Environment Samples")
plt.legend()
plt.savefig('results/iclr2018/cheetah.jpg')
plt.show()
