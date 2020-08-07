from rlkit.misc.data_processing import get_trials
import matplotlib.pyplot as plt
import numpy as np

from rlkit.misc.visualization_util import sliding_mean


def main():
    tdm_trials = get_trials(
        "/home/vitchyr/git/railrl/data/doodads3/01-03-final-gym-pusher3d/",
        criteria={
            'exp_id': '16',
            'algorithm': 'DDPG-TDM',
        }
    )
    mb_trials = get_trials(
        "/home/vitchyr/git/railrl/data/doodads3/01-03-final-gym-pusher3d/",
        criteria={
            'exp_id': '2',
            'algorithm': 'Model-Based-Dagger',
        }
    )
    ddpg_trials = get_trials(
        "/home/vitchyr/git/railrl/data/doodads3/01-03-final-gym-pusher3d/",
        criteria={
            'exp_id': '7',
            'algorithm': 'DDPG',
        }
    )
    her_trials = get_trials(
        "/home/vitchyr/git/railrl/data/doodads3/01-03-final-gym-pusher3d/",
        criteria={
            'exp_id': '10',
            'algorithm': 'HER-Andrychowicz',
        }
    )
    ddpg_sparse_trials = get_trials(
        "/home/vitchyr/git/railrl/data/doodads3/01-03-final-gym-pusher3d/",
        criteria={
            'exp_id': '8',
            'algorithm': 'DDPG-Sparse',
        }
    )

    MAX_ITERS = 1000000

    plt.figure()
    base_key = 'Multitask Final L2 distance to goal Mean'
    for trials, name, key in [
        (tdm_trials, 'TDMs', base_key),
        (ddpg_trials, 'DDPG', base_key),
        (ddpg_sparse_trials, 'DDPG-Sparse', base_key),
        (her_trials, 'HER', base_key),
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
    plt.savefig('results/iclr2018/gym-pusher-3d.jpg')
    plt.show()


if __name__ == '__main__':
    main()
