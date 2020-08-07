from rlkit.misc.data_processing import Experiment
import matplotlib.pyplot as plt
import numpy as np

from rlkit.misc.visualization_util import sliding_mean


def main():
    relabel_exp = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/01-02-ddpg-tdm-ant-nupo-sweep/",
        criteria={
            'relabel': True,
            'ddpg_tdm_kwargs.base_kwargs.reward_scale': 1,
        }
    )
    no_relabel_exp = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/01-02-ddpg-tdm-ant-nupo-sweep/",
        criteria={
            'relabel': False,
            'ddpg_tdm_kwargs.base_kwargs.reward_scale': 1,
        }
    )

    MAX_ITERS = 100

    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)
    plot_key = 'Final Distance to goal Mean'.replace(' ', '_')
    # for ax, exp, name in [
    #     (ax1, relabel_exp, 'Relabel'),
    #     (ax2, no_relabel_exp, 'No Relabel'),
    # ]:
    for exp, name in [
        (relabel_exp, 'Relabel'),
        (no_relabel_exp, 'No Relabel'),
    ]:
        fig = plt.figure()
        for nupo in [1, 5, 10]:
            trials = exp.get_trials({
                'ddpg_tdm_kwargs.base_kwargs.num_updates_per_env_step': nupo
            })
            all_values = []
            for trial in trials:
                try:
                    values_ts = trial.data[plot_key]
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
            plt.plot(epochs, mean, label="{} updates per step".format(
                nupo
            ))
        # plt.title(name)

        plt.xlabel("Environment Samples (x1,000)")
        plt.ylabel("Final Distance to Goal Position")
        plt.legend()
        # print(fig.get_size_inches())
        # fig.set_size_inches(6.4*1, 4.8*2)
        plt.savefig(
            'results/iclr2018/ant-nupo-sweep-{}.jpg'.format(
                name.lower().replace(' ', '-')
            ),
            # transparent=True, bbox_inches='tight', pad_inches=0,
        )
        plt.show()


if __name__ == '__main__':
    main()
