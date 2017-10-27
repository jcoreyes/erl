from railrl.misc.data_processing import get_all_csv
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np

naf_path = "/home/vitchyr/git/rllab-rail/railrl/data/papers/iclr2018/sawyer/naf/"
ddpg_path = "/home/vitchyr/git/rllab-rail/railrl/data/papers/iclr2018/sawyer/ddpg"
tdm_path = "/home/vitchyr/git/rllab-rail/railrl/data/papers/iclr2018/sawyer/tdm/"

ddpg_csvs = get_all_csv(ddpg_path)
naf_csvs = get_all_csv(naf_path)
tdm_csvs = get_all_csv(tdm_path)

plt.figure()
for trials, name, key in [
    (ddpg_csvs, 'DDPG', 'Test_Last_N_Step_Distance_from_Desired_End_Effector_Position_Mean'),
    (naf_csvs, 'NAF',
     'Test_Last_N_Step_Distance_from_Desired_End_Effector_Position_Mean'),
    (tdm_csvs, 'TDM',
     'Test_Distance_from_Desired_End_Effector_Position_Mean'),
]:
    all_values = []
    min_len = np.inf
    for trial in trials:
        values_ts = trial[key]
        min_len = min(min_len, len(values_ts))
        all_values.append(values_ts)
    costs = np.vstack([
        values[:min_len]
        for values in all_values
    ])
    mean = np.mean(costs, axis=0)
    std = np.std(costs, axis=0)
    epochs = np.arange(0, len(costs[0]))
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.1)
    plt.plot(epochs, mean, label=name)


# plt.xscale('log')
plt.xlabel("Epoch (1000 steps)")
plt.ylabel("Distance to Goal")
plt.legend()
plt.savefig('results/iclr2018/sawyer.jpg')
plt.show()
