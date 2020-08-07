from rlkit.misc.data_processing import Experiment
import matplotlib.pyplot as plt
import numpy as np

path = "/mnt/data-backup-12-02-2017/doodads3/10-27-get-results-handxyxy-best-hp-no-oc-sampling-nspe1000/"
exp = Experiment(path)
base_criteria = {
    'algo_params.num_updates_per_env_step': 25,
}
tau_to_criteria = {}
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
MAX_ITERS = 35
for tau in taus:
    trials = tau_to_trials[tau]
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
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.1)
    plt.plot(epochs, mean, label=r"$\tau = {}$".format(str(tau)))


plt.xlabel("Environment Steps (x1000)")
plt.ylabel("Distance to Goal")
# plt.title("Pusher: Distance to Goal vs Environment Samples for varying ")
plt.legend()
plt.savefig('results/iclr2018/tau-sweep.jpg')
plt.show()
