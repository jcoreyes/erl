from railrl.misc.data_processing import Experiment
import matplotlib.pyplot as plt
import numpy as np

def main():
    ant_her_final_exp = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-23-her-andrychowicz-ant-rebutal/"
    )
    ant_tdm_exp = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-24-ddpg-nupo-sweep-ant/"
    )

    ddpg_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-23-ddpg-nupo-sweep-ant/"
    ).get_trials({
        'exp_id': '19',
    })
    her_andrychowicz_trials = ant_her_final_exp.get_trials({
        'exp_id': '10',
    })
    ddpg_tdm_trials = ant_tdm_exp.get_trials({
        'exp_id': '12',
    })
    ddpg_indicator_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-23-ddpg-sparse-sweep-4/"
    ).get_trials({
        'env_class.$class': 'railrl.envs.multitask.ant_env.GoalXYPosAnt',
        'ddpg_tdm_kwargs.base_kwargs.num_updates_per_env_step': 1,
    })
    mb_trials = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-24-dagger-mb-ant-cheetah-pos-and-vel/"
    ).get_trials({
        'exp_id': '1',
    })

    MAX_ITERS = 10001

    plt.figure()
    base_key = 'Final Distance to goal Mean'
    for trials, name, key in [
        (her_andrychowicz_trials, 'HER', base_key),
        (ddpg_trials, 'DDPG', base_key),
        (ddpg_tdm_trials, 'TDMs', base_key),
        (ddpg_indicator_trials, 'DDPG-Sparse', base_key),
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
        # if name != 'TDM':
        # costs = smooth(costs)
        mean = np.mean(costs, axis=0)
        std = np.std(costs, axis=0)
        epochs = np.arange(0, len(costs[0]))
        plt.fill_between(epochs, mean - std, mean + std, alpha=0.1)
        plt.plot(epochs, mean, label=name)

    plt.xlabel("Environment Samples (x1,000)")
    plt.ylabel("Final Euclidean Distance to Goal Position")
    plt.legend()
    plt.savefig('results/iclr2018/ant.jpg')
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
