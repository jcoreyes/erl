from rlkit.misc.data_processing import Experiment, get_trials
import matplotlib.pyplot as plt
import numpy as np

from rlkit.misc.visualization_util import sliding_mean

def main():
    her_andry_trials = get_trials(
        "/home/vitchyr/git/railrl/data/doodads3/12-24-her-andrychowicz-pusher-rebutal/",
        criteria={
            'exp_id': '11',
        },
    )

    ddpg_path = "/mnt/data-backup-12-02-2017/doodads3/10-25-ddpg-pusher-again-baseline-with-reward-bonus/"
    ddpg_criteria = {
        'algo_params.num_updates_per_env_step': 5,
        'algo_params.scale_reward': 1,
        'algo_params.tau': 0.01,
    }
    mb_path = "/mnt/data-backup-12-02-2017/doodads3/10-25-abhishek-mb-baseline-pusher-again-shaped/"
    mb_criteria = None
    our_path = "/mnt/data-backup-12-02-2017/doodads3/11-02-get-results-handxyxy-small-sweep"
    our_criteria = {
        'algo_params.num_updates_per_env_step': 5,
        'epoch_discount_schedule_params.value': 5,
        'algo_params.tau': 0.001,
    }
    ddpg_trials = get_trials(ddpg_path, criteria=ddpg_criteria)
    mb_trials = get_trials(mb_path, criteria=mb_criteria)
    our_trials = get_trials(our_path, criteria=our_criteria)
    ddpg_sparse_trials = get_trials(
        "/home/vitchyr/git/railrl/data/doodads3/12-23-ddpg-sparse-sweep-4/",
        criteria={
            'env_class.$class': 'railrl.envs.multitask.pusher2d.CylinderXYPusher2DEnv',
            'ddpg_tdm_kwargs.base_kwargs.num_updates_per_env_step': 1,
        }
    )
    MAX_ITERS = 100

    base_key = 'Final_Euclidean_distance_to_goal_Mean'
    plt.figure()
    for trials, name, key in [
        (our_trials, 'TDM', 'test_'+base_key),
        (ddpg_trials, 'DDPG', base_key),
        (her_andry_trials, 'HER', "Final Distance object to goal Mean"),
        (ddpg_sparse_trials, 'DDPG-Sparse', "Final Distance object to goal Mean"),
        (mb_trials, 'Model Based', base_key),
    ]:
        key = key.replace(" ", "_")
        all_values = []
        min_len = np.inf
        for trial in trials:
            values_ts = trial.data[key]
            min_len = min(min_len, len(values_ts))
            all_values.append(values_ts)
        try:
            costs = np.vstack([
                values[:min_len]
                for values in all_values
            ])
        except ValueError as e:
            import ipdb; ipdb.set_trace()
        costs = costs[:, :min(costs.shape[1], MAX_ITERS)]
        if name == 'HER':
            costs = sliding_mean(costs, 20)
        mean = np.mean(costs, axis=0)
        std = np.std(costs, axis=0)
        epochs = np.arange(0, len(costs[0]))
        plt.fill_between(epochs, mean - std, mean + std, alpha=0.1)
        plt.plot(epochs, mean, label=name)



    # plt.xscale('log')
    plt.xlabel("Environment Samples (x1000)")
    plt.ylabel("Final Distance to Goal Position")
    plt.legend()
    plt.savefig('results/iclr2018/pusher.jpg')
    plt.show()

if __name__ == '__main__':
    main()
