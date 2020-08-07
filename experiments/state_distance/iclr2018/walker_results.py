from rlkit.misc.data_processing import Experiment
import matplotlib.pyplot as plt
import numpy as np


def main():
    tdm_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-25-tdm-ddpg-walker-position-long/"
    ).get_trials({
        'exp_id': '1',
    })
    trpo_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-25-trpo-walker-position-long/"
    ).get_trials({
        'exp_id': '4',
    })
    mb_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-25-mb-dagger-walker-position-long-take2/"
    ).get_trials({
        'exp_id': '2',
    })
    ddpg_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-25-ddpg-walker-position-long-take2/"
    ).get_trials({
        'exp_id': '1',
    })

    MAX_ITERS = 10001

    plt.figure()
    base_key = 'Final xpos errors Mean'
    for trials, name, key in [
        (tdm_trials, 'TDMs', base_key),
        (mb_trials, 'Model-Based', base_key),
        (ddpg_trials, 'DDPG', base_key),
        (trpo_trials, 'TRPO', base_key),
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

    plt.xlabel("Environment Samples (x10,000)")
    plt.ylabel("Final Distance to Goal Position")
    plt.legend()
    plt.savefig('results/iclr2018/walker.jpg')
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
