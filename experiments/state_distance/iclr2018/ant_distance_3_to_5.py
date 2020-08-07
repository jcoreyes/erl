from rlkit.misc.data_processing import Experiment
import matplotlib.pyplot as plt
import numpy as np

from rlkit.misc.visualization_util import sliding_mean


def main():
    tdm_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/01-02-ddpg-tdm-ant-nupo-sweep/",
        criteria={
            'exp_id': '27',  # 23 for NUPO = 20, 27 for NUPO = 10
        }
    ).get_trials()
    mb_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/01-02-ant-distance-3-to-5/",
        criteria={
            'exp_id': '0',
            'algorithm': 'Model-Based-Dagger',
        }
    ).get_trials()
    ddpg_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/01-02-ant-distance-3-to-5/",
        criteria={
            'exp_id': '3',
            'algorithm': 'DDPG',
        }
    ).get_trials()

    MAX_ITERS = 1000000

    plt.figure()
    base_key = 'Final Distance to goal Mean'
    for trials, name, key in [
        (tdm_trials, 'TDMs', base_key),
        (ddpg_trials, 'DDPG', base_key),
        (mb_trials, 'Model-Based', base_key),
    ]:
        key = key.replace(" ", "_")
        all_values = []
        for trial in trials:
            try:
                values_ts = trial.data[key]
            except:
                import ipdb; ipdb.set_trace()
            values_ts = sliding_mean(values_ts, window=10)
            all_values.append(values_ts)
        min_len = min(map(len, all_values))
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

    plt.xlabel("Environment Samples (x1,000)")
    plt.ylabel("Final Euclidean Distance to Goal Position")
    plt.legend()
    plt.savefig('results/iclr2018/ant-distance-3-to-5.jpg')
    plt.show()


if __name__ == '__main__':
    main()
