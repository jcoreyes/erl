from rlkit.misc.data_processing import Experiment
import matplotlib.pyplot as plt
import numpy as np

def main():
    ddpg_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-23-ddpg-nupo-sweep-ant/",
        criteria={
            'exp_id': '16',
        },
    ).get_trials()
    her_andrychowicz_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-23-her-andrychowicz-ant-rebutal/",
        criteria={
            'exp_id': '14',
        },
    ).get_trials()
    # Ant results with batch size of 128
    tdm_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-24-ddpg-nupo-sweep-ant/",
        criteria={
            'exp_id': '16',
        }
    ).get_trials()
    # Accidentally called this pusher, but it's really ant
    # Here, x-axis is 10k steps.
    # tdm_trials = Experiment(
    #     "/home/vitchyr/git/rlkit/data/doodads3/12-27-pusher-reward-scale-tau-uniform-or-truncated-geo-sweep-2/",
    #     criteria={
    #         'ddpg_tdm_kwargs.base_kwargs.reward_scale': 100,
    #         'ddpg_tdm_kwargs.tdm_kwargs.tau_sample_strategy':
    #             'truncated_geometric',
    #     }
    # ).get_trials()
    ddpg_indicator_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-23-ddpg-sparse-sweep-4/",
        criteria={
            'env_class.$class': 'railrl.envs.multitask.ant_env.GoalXYPosAnt',
            'ddpg_tdm_kwargs.base_kwargs.num_updates_per_env_step': 1,
        },
    ).get_trials()
    mb_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-24-dagger-mb-ant-cheetah-pos-and-vel/",
        criteria={
            'exp_id': '1',
        },
    ).get_trials()

    # MAX_ITERS = 10001
    MAX_ITERS = 200

    plt.figure()
    base_key = 'Final Distance to goal Mean'
    for trials, name, key in [
        (tdm_trials, 'TDM', base_key),
        (mb_trials, 'Model-Based', base_key),
        (ddpg_trials, 'DDPG', base_key),
        (her_andrychowicz_trials, 'HER', base_key),
        (ddpg_indicator_trials, 'DDPG-Sparse', base_key),
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
    plt.savefig('results/iclr2018/ant.jpg')
    # plt.show()


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
