from railrl.misc.data_processing import Experiment
import matplotlib.pyplot as plt
import numpy as np

ddpg_trials = Experiment(
    "/home/vitchyr/git/railrl/data/doodads3/12-17-tdm-half-cheetah-xpos/"
).get_trials({
    'algorithm': 'DDPG',
    'multitask': True,
    'exp_id': '0',
})
tdm_trials = Experiment(
    "/home/vitchyr/git/railrl/data/doodads3/12-17-tdm-half-cheetah-xpos/"
).get_trials({
    'algorithm': 'DDPG-TDM',
    'multitask': True,
    'exp_id': '3',
})
mb_trials = Experiment(
    "/home/vitchyr/git/railrl/data/doodads3/12-24-dagger-mb-ant-cheetah-pos-and-vel/"
).get_trials({
    'exp_id': '2'
})
ddpg_indicator_trials = Experiment(
    "/home/vitchyr/git/railrl/data/doodads3/12-23-ddpg-sparse-sweep-4/"
).get_trials({
    'env_class.$class':
        'railrl.envs.multitask.half_cheetah.GoalXPosHalfCheetah',
    'exp_id': '13',
})
her_andry_trials = Experiment(
    "/home/vitchyr/git/railrl/data/doodads3/12-24-her-andrychowicz-cheetah-xpos-rebutal/"
).get_trials({
    'exp_id': '5',
})

MAX_ITERS = 10000
plt.figure()
base_key = 'Final_Distances_to_goal_Mean'
key2 = 'Final_Distance_to_goal_Mean'
for trials, name, key in [
    (mb_trials, 'Model Based', key2),
    (ddpg_trials, 'DDPG', base_key),
    (tdm_trials, 'TDM', base_key),
    (her_andry_trials, 'HER', base_key),
    (ddpg_indicator_trials, 'DDPG-Sparse', base_key),
]:
    all_values = []
    min_len = np.inf
    if len(trials) == 0:
        print(name)
        import ipdb; ipdb.set_trace()
    for trial in trials:
        try:
            values_ts = trial.data[key]
        except:
            import ipdb; ipdb.set_trace()
        min_len = min(min_len, len(values_ts))
        all_values.append(values_ts)
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


# plt.xscale('log')
plt.xlabel("Environment Samples (x1,000)")
plt.ylabel("Velocity Error")
# plt.title(r"Half Cheetah: Velocity Error vs Environment Samples")
plt.legend()
plt.savefig('results/iclr2018/cheetah-xpos.jpg')
plt.show()
