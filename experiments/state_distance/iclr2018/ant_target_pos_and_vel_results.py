import matplotlib.pyplot as plt
import numpy as np

from rlkit.misc.data_processing import get_trials
from rlkit.misc.visualization_util import sliding_mean


def main():
    tdm_trials = get_trials(
        "/home/vitchyr/git/railrl/data/doodads3/01-03-final-ant-pos-and-vel/",
        criteria={
            'algorithm': 'DDPG-TDM',
            'exp_id': '17',
        }
    )
    ddpg_trials = get_trials(
        "/home/vitchyr/git/railrl/data/doodads3/01-03-final-ant-pos-and-vel/",
        criteria={
            'algorithm': 'DDPG',
            'exp_id': '5',
        }
    )
    ddpg_sparse_trials = get_trials(
        "/home/vitchyr/git/railrl/data/doodads3/01-03-final-ant-pos-and-vel/",
        criteria={
            'algorithm': 'DDPG-Sparse',
            'exp_id': '22',
        }
    )
    her_trials = get_trials(
        "/home/vitchyr/git/railrl/data/doodads3/01-03-final-ant-pos-and-vel/",
        criteria={
            'algorithm': 'HER-Andrychowicz',
            'exp_id': '26',
        }
    )
    mb_trials = get_trials(
        "/home/vitchyr/git/railrl/data/doodads3/01-03-final-ant-pos-and-vel/",
        criteria={
            'algorithm': 'Model-Based-Dagger',
            'exp_id': '4',
        }
    )

    MAX_ITERS = 300

    plt.figure()
    key = 'Final_Weighted_Error_Mean'
    for trials, name in [
        (tdm_trials, 'TDM'),
        (ddpg_trials, 'DDPG'),
        (her_trials, 'HER'),
        (ddpg_sparse_trials, 'DDPG-Sparse'),
        (mb_trials, 'Model-Based'),
    ]:
        all_values = []
        for trial in trials:
            try:
                values_ts = trial.data[key]
                values_ts = sliding_mean(values_ts, window=5)
            except:
                import ipdb; ipdb.set_trace()
            all_values.append(values_ts)
        min_len = min(map(len, all_values))
        # max_len = max(map(len, all_values))
        # all_values = [
        #     np.pad(values, (0, max_len - len(values)), 'constant',
        #            constant_values=np.nan)
        #     for values in all_values
        # ]
        costs = np.vstack([
            values[:min_len]
            for values in all_values
        ])
        costs = costs[:, :min(costs.shape[1], MAX_ITERS)]
        mean = np.nanmean(costs, axis=0)
        std = np.nanstd(costs, axis=0)
        epochs = np.arange(0, len(costs[0]))
        plt.fill_between(epochs, mean - std, mean + std, alpha=0.1)
        plt.plot(epochs, mean, label=name)

    plt.xlabel("Environment Samples (x1,000)")
    plt.ylabel("Final Distance to Goal Position and Velocity")
    plt.legend()
    plt.savefig('results/iclr2018/ant-pos-and-vel.jpg')
    plt.show()


if __name__ == '__main__':
    main()
