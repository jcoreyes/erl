from railrl.misc.data_processing import Experiment
import matplotlib.pyplot as plt
import numpy as np


her_exp = Experiment(
    "/home/vitchyr/git/railrl/data/doodads3/12-22-her-andrychowicz-try-hard-2/"
)
her_andry_trials = her_exp.get_trials({
    'env_class.$class':
        "railrl.envs.multitask.reacher_7dof.Reacher7DofXyzGoalState",
    'ddpg_tdm_kwargs.base_kwargs.reward_scale': 100,
    'ddpg_tdm_kwargs.base_kwargs.num_updates_per_env_step': 1,
})

ddpg_path = "/mnt/data-backup-12-02-2017/doodads3/10-25-ddpg" \
            "-reacher-pusher-baseline/"
ddpg_criteria = {
    'algo_params.num_updates_per_env_step': 1,
    'algo_params.scale_reward': 1,
    'algo_params.tau': 0.001,
    'env_class.$class': "railrl.envs.multitask.her_reacher_7dof_env.Reacher7Dof"
}
# Example DDPG policy: /home/vitchyr/git/rllab-rail/railrl/data/doodads3/10-25-ddpg-reacher-pusher-baseline/10-25-ddpg-reacher-pusher-baseline-id4-s740143/

mb_path = "/mnt/data-backup-12-02-2017/doodads3/10-25-abhishek" \
          "-mb-baseline-pusher-reacher-300-300-net/"
mb_criteria = {
    'env_name_or_class.$class':
        'railrl.envs.multitask.her_reacher_7dof_env.Reacher7Dof',
}
# our_path = "/home/vitchyr/git/rllab-rail/railrl/data/doodads3/10-24-sdql-reacher7everything-vtau/"
# our_criteria = {
#     'raw_explore_policy': 'oc',
# }
our_path = "/mnt/data-backup-12-02-2017/doodads3/10-27-sdql-reacher-get-long-results"
our_criteria = {
    'epoch_discount_schedule_params.value': 15,
    'eval_with_oc_policy': False,
    'algo_params.num_updates_per_env_step': 25,
}

our_exp = Experiment(our_path)
our_trials = our_exp.get_trials(our_criteria)
t1 = our_trials[0]

ddpg_exp = Experiment(ddpg_path)
mb_exp = Experiment(mb_path)
# her_dense_exp = Experiment(her_dense_path)

ddpg_trials = ddpg_exp.get_trials(ddpg_criteria)
mb_trials = mb_exp.get_trials(mb_criteria)
# her_dense_trials = her_dense_exp.get_trials(her_dense_criteria)

MAX_ITERS = 1000

base_key = 'Final_Euclidean_distance_to_goal_Mean'
plt.figure()
for trials, name, key in [
    (ddpg_trials, 'DDPG', base_key),
    (mb_trials, 'Model Based', base_key),
    (our_trials, 'TDM', 'test_'+base_key),
    (her_andry_trials, 'HER', base_key),
]:
    all_values = []
    min_len = np.inf
    for trial in trials:
        values_ts = trial.data[key]
        min_len = min(min_len, len(values_ts))
        all_values.append(values_ts)
    costs = np.vstack([
        values[:min_len]
        for values in all_values
    ])
    costs = costs[:, :min(costs.shape[1], MAX_ITERS)]
    mean = np.mean(costs, axis=0)
    std = np.std(costs, axis=0)
    epochs = np.arange(0, len(costs[0]))
    # if name == 'TDM':
    #     epochs = epochs / 10
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.1)
    plt.plot(epochs, mean, label=name)


plt.xlabel("Environment Steps (x1000)")
plt.ylabel("Final Distance to Goal")
plt.legend()
plt.savefig('results/iclr2018/reacher.jpg')
plt.show()
