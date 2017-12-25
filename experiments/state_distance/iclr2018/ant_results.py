from railrl.misc.data_processing import Experiment
import matplotlib.pyplot as plt
import numpy as np

def main():
    # ant_12_21_her_exp = Experiment(
    #     "/home/vitchyr/git/railrl/data/doodads3/12-21-her-sac-ant-sweep/"
    # )
    # ant_12_21_tdm_her_exp = Experiment(
    #     "/home/vitchyr/git/railrl/data/doodads3/12-21-tdm-her-sac-ant-sweep/"
    # )
    # ant_12_20_exp = Experiment(
    #     "/home/vitchyr/git/railrl/data/doodads3/12-20-tdm-ant/"
    # )
    # sac_her_andrychowicz_trials = ant_12_21_her_exp.get_trials({
    #     'sac_tdm_kwargs.tdm_kwargs.finite_horizon': False,
    #     'sac_tdm_kwargs.tdm_kwargs.reward_type': 'sparse',
    #     'sac_tdm_kwargs.tdm_kwargs.vectorized': False,
    #     'sac_tdm_kwargs.tdm_kwargs.terminate_when_goal_reached': True,
    #     'sac_tdm_kwargs.base_kwargs.reward_scale': 10,
    # })
    # sac_her_distance_trials = ant_12_21_her_exp.get_trials({
    #     'sac_tdm_kwargs.tdm_kwargs.finite_horizon': False,
    #     'sac_tdm_kwargs.tdm_kwargs.reward_type': 'distance',
    #     'sac_tdm_kwargs.tdm_kwargs.vectorized': False,
    #     'sac_tdm_kwargs.tdm_kwargs.terminate_when_goal_reached': False,
    #     'sac_tdm_kwargs.base_kwargs.reward_scale': 10,
    # })
    # sac_trials = ant_12_19_exp.get_trials({
    #     'algorithm': 'SAC',
    #     'algo_params.reward_scale': 10,
    # })
    # sac_tdm_trials = ant_12_21_tdm_her_exp.get_trials({
    #     'sac_tdm_kwargs.tdm_kwargs.finite_horizon': True,
    #     'sac_tdm_kwargs.tdm_kwargs.reward_type': 'distance',
    #     'sac_tdm_kwargs.tdm_kwargs.sample_train_goals_from': 'her',
    #     'sac_tdm_kwargs.tdm_kwargs.dense_rewards': False,
    #     'sac_tdm_kwargs.base_kwargs.reward_scale': 1000,
    # })
    #
    # trpo_trials = ant_12_19_exp.get_trials({
    #     'algorithm': 'TRPO',
    #     'trpo_params.step_size': 0.001,
    # })
    # ddpg_her_distance_trials = ant_12_20_exp.get_trials({
    #     'algorithm': 'HER-Dense DDPG',
    #     'ddpg_tdm_kwargs.base_kwargs.reward_scale': 100,
    #     'ddpg_tdm_kwargs.base_kwargs.discount': 0.95,
    # })

    ant_ddpg_exp = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-23-ddpg-nupo-sweep-ant/"
    )
    ant_her_final_exp = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-23-her-andrychowicz-ant-rebutal/"
    )
    ant_tdm_exp = Experiment(
        "/home/vitchyr/git/railrl/data/doodads3/12-24-ddpg-nupo-sweep-ant/"
    )

    ddpg_trials = ant_ddpg_exp.get_trials({
        'exp_id': '19',
    })
    ddpg_her_andrychowicz_trials = ant_her_final_exp.get_trials({
        'exp_id': '20',
    })
    ddpg_tdm_trials = ant_tdm_exp.get_trials({
        'exp_id': '12',
    })

    MAX_ITERS = 200

    plt.figure()
    for trials, name, key in [
        (
                ddpg_her_andrychowicz_trials,
                'HER',
                'Final Distance to goal Mean',
        ),
        (ddpg_trials, 'DDPG', 'Final Distance to goal Mean'),
        (
             ddpg_tdm_trials,
             'TDMs',
             'Final Distance to goal Mean',
        ),
        # (
        #     ddpg_her_distance_trials,
        #     'DDPG-HER: Dense, Distance',
        #     'Final Distance to goal Mean',
        # ),
        # (sac_trials, 'SAC', 'Final Distance to goal Mean'),
        # (
        #      sac_tdm_trials,
        #      'SAC-HER: Sparse, Distance (TDMs)',
        #      'Final Distance to goal Mean',
        # ),
        # (
        #     sac_her_andrychowicz_trials,
        #     'SAC-HER: Dense, Indicator (Andrychowicz)',
        #     'Final Distance to goal Mean',
        # ),
        # (
        #     sac_her_distance_trials,
        #     'SAC-HER: Dense, Distance',
        #     'Final Distance to goal Mean',
        # ),
    ]:
        key = key.replace(" ", "_")
        all_values = []
        for trial in trials:
            try:
                values_ts = trial.data[key]
            except:
                import ipdb; ipdb.set_trace()
            all_values.append(values_ts)
        if name != 'DDPG':
            all_values = [
                average_every_n_elements(values, 10)
                for values in all_values
            ]
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

    plt.xlabel("Environment Samples (x10,000)")
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
