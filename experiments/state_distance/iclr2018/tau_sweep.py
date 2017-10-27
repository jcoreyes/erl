from railrl.misc.data_processing import Experiment
import matplotlib.pyplot as plt
import numpy as np

# path = "/home/vitchyr/git/rllab-rail/railrl/data/doodads3/10-25-get-results-take1/"
path = "/home/vitchyr/git/rllab-rail/railrl/data/doodads3/10-27-get-results-handxyxy-best-hp-no-oc-sampling-nspe1000/"
exp = Experiment(path)
# base_criteria = {
#     'algo_params.goal_dim_weights': [1] * 17,
#     'env_class.$class': "railrl.envs.multitask.reacher_7dof.Reacher7DofGoalStateEverything"
# }
base_criteria = {
    'algo_params.num_updates_per_env_step': 5,
    # 'algo_params.goal_dim_weights': [1] * 17,
    # 'env_class.$class': "railrl.envs.multitask.reacher_7dof.Reacher7DofGoalStateEverything"
}
tau_to_criteria = {}
# taus = [1, 5, 10, 25]
taus = [1, 5, 15, 50]
for tau in taus:
    criteria = base_criteria.copy()
    criteria['epoch_discount_schedule_params.value'] = tau
    tau_to_criteria[tau] = criteria


tau_to_trials = {}
for tau in taus:
    tau_to_trials[tau] = exp.get_trials(tau_to_criteria[tau])

# key = 'Final_Euclidean_distance_to_goal_Mean'
key = 'test_Final_Euclidean_distance_to_goal_Mean'
for tau, trials in tau_to_trials.items():
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
    mean = np.mean(costs, axis=0)
    std = np.std(costs, axis=0)
    epochs = np.arange(0, len(costs[0])) / 10
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.1)
    plt.plot(epochs, mean, label=r"$\tau = {}$".format(str(tau)))


plt.xlabel("Environment Steps (x1000)")
plt.ylabel("Distance to Goal")
plt.title("Distance to Goal vs Environment Samples")
plt.legend()
plt.savefig('results/iclr2018/tau_sweep.jpg')
plt.show()
