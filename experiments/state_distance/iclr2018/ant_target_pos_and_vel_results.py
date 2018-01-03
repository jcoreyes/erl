from railrl.misc.data_processing import Experiment
import matplotlib.pyplot as plt
import numpy as np

from railrl.misc.visualization_util import sliding_mean


def main():
    tdm_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/01-02-ant-pos-and-vel-target/",
        criteria={
            # 'exp_id': '33',  # weight 0.99
            'exp_id': '22',  # weight 0.9
            'algorithm': 'DDPG-TDM',
        }
    ).get_trials()
    mb_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/01-02-ant-pos-and-vel-target/",
        criteria={
            # 'exp_id': '0',  #weight 0.99
            'exp_id': '1',  #weight 0.9
            'algorithm': 'Model-Based-Dagger',
        }
    ).get_trials()
    tmp = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-30-ant-pos-vel/",
    )
    ddpg_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-30-ant-pos-vel/",
        criteria={
            # 'exp_id': '6',  # weight 0.99
            'exp_id': '8',  # weight 0.9
            'algorithm': 'DDPG',
        }
    ).get_trials()

    MAX_ITERS = 1000000

    plt.figure()
    key1 = 'Final_weighted_vel_error_Mean'
    key2 = 'Final_weighted_pos_error_Mean'
    for trials, name in [
        (tdm_trials, 'TDMs'),
        (ddpg_trials, 'DDPG'),
        (mb_trials, 'Model-Based'),
    ]:
        all_values = []
        for trial in trials:
            try:
                values_ts = trial.data[key1] + trial.data[key2]
                values_ts = sliding_mean(values_ts, window=10)
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
    plt.ylabel("Final Weighted Distance to Goal Position")
    plt.legend()
    plt.savefig('results/iclr2018/ant-pos-and-vel.jpg')
    plt.show()


if __name__ == '__main__':
    main()
