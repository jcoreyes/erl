from rlkit.misc.data_processing import Experiment, get_trials
import matplotlib.pyplot as plt
import numpy as np

path = "/mnt/data-backup-12-02-2017/doodads3/10-21-sdql-compare-vectorized-delta-normal-big-sweep/"
exp = Experiment(path)
base_criteria = {
    'env_class.$class':
        "railrl.envs.multitask.reacher_7dof.Reacher7DofFullGoalState"
}
algos = [
    'railrl.algos.state_distance.state_distance_q_learning.HorizonFedStateDistanceQLearning',
    'railrl.algos.state_distance.vectorized_sdql.VectorizedTauSdql',
]
algo_to_trials = {}
for algo in algos:
    criteria = base_criteria.copy()
    criteria['algo_class.$class'] = algo
    algo_to_trials[algo] = exp.get_trials(criteria)

key = 'Final_Euclidean_distance_to_goal_Mean'
MAX_ITERS = 50
for algo, trials in algo_to_trials.items():
    if algo == 'railrl.algos.state_distance.state_distance_q_learning' \
               '.HorizonFedStateDistanceQLearning':
        name = 'Scalar'
    else:
        name = 'Vector'
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
    epochs = np.arange(0, len(costs[0])) / 10
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.1)
    plt.plot(epochs, mean, label=name)


plt.xlabel("Environment Steps (x1000)")
plt.ylabel("Distance to Goal")
# plt.title("Distance to Goal vs Environment Samples for Scalar and Vector "
#           "Supervision")
plt.legend()
plt.savefig('results/iclr2018/vectorized-vs-scalar.jpg')
plt.show()
