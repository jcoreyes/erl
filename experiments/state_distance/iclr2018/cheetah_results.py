from rlkit.misc.data_processing import Experiment
import matplotlib.pyplot as plt
import numpy as np

mb_trials = Experiment(
    "/home/vitchyr/git/railrl/data/doodads3/12-24-dagger-mb-ant-cheetah-pos-and-vel/",
    criteria={
        'exp_id': '0',
    },
).get_trials()
ddpg_trials = Experiment(
    "/home/vitchyr/git/railrl/data/doodads3/12-12-tdm-half-cheetah-short-epoch-nupo-sweep/",
    criteria={
        'exp_id': '5',
        'algorithm': 'DDPG',
    },
).get_trials()
tdm_trials = Experiment(
    "/home/vitchyr/git/railrl/data/doodads3/12-12-tdm-half-cheetah-short-epoch-nupo-sweep/",
    criteria={
        'exp_id': '8',
        'algorithm': 'DDPG-TDM',
    }
).get_trials()
ddpg_indicator_trials = Experiment(
    "/home/vitchyr/git/railrl/data/doodads3/12-24-ddpg-sparse-no-relabel-cheetah-xvel/",
    criteria={
        'exp_id': '7',
    }
).get_trials()
her_andry_trials = Experiment(
    "/home/vitchyr/git/railrl/data/doodads3/12-24-her-andrychowicz-cheetah-xvel-rebutal/",
    criteria={
        'exp_id': '6',
    }
).get_trials()

MAX_ITERS = 100
plt.figure()
for trials, name, key in [
    (tdm_trials, 'TDM', 'Final_xvel_errors_Mean'),
    (ddpg_trials, 'DDPG', 'Final_xvel_errors_Mean'),
    (her_andry_trials, 'HER', 'Final_xvel_errors_Mean'),
    (ddpg_indicator_trials, 'DDPG-Sparse', 'Final_xvel_errors_Mean'),
    (mb_trials, 'Model Based', 'Final_xvel_errors_Mean'),
]:
    if len(trials) == 0:
        print(name)
        import ipdb;

        ipdb.set_trace()
    all_values = []
    min_len = np.inf
    for trial in trials:
        try:
            values_ts = trial.data[key]
        except:
            import ipdb;

            ipdb.set_trace()
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
plt.ylabel("Final Velocity Error")
# plt.title(r"Half Cheetah: Velocity Error vs Environment Samples")
plt.legend(loc='upper right')
plt.savefig('results/iclr2018/cheetah.jpg')
plt.show()
