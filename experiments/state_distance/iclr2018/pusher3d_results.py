from rlkit.misc.data_processing import Experiment
import matplotlib.pyplot as plt
import numpy as np


def main():
    ddpg_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-29-find-pusher3d-mismatch-2",
        criteria={
            'env_kwargs.reward_coefs': [1, 0, 0],
            'exp_id': '1',
        }
    ).get_trials()
    mb_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-30-mb-dagger-pusher3d-fixed-2/",
    ).get_trials()

    MAX_ITERS = 10000

    plt.figure()
    base_key = 'Final Distance to goal Mean'
    for trials, name, key in [
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
    plt.savefig('results/iclr2018/pusher3d.jpg')
    plt.show()


def average_every_n_elements(arr, n):
    return np.nanmean(
        np.pad(
            arr.astype(float),
            (0, n - arr.size % n),
            mode='constant',
            constant_values=np.NaN,
        ).reshape(-1, n),
        axis=1
    )


if __name__ == '__main__':
    main()
