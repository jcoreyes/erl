from rlkit.misc.data_processing import get_all_csv
import matplotlib.pyplot as plt
import numpy as np

# naf_path = "/home/vitchyr/git/rllab-rail/rlkit/data/papers/iclr2018/sawyer/naf/"
# ddpg_path = "/home/vitchyr/git/rllab-rail/rlkit/data/papers/iclr2018/sawyer" \
#             "/ddpg-new"
# tdm_path = "/home/vitchyr/git/rllab-rail/rlkit/data/papers/iclr2018/sawyer" \
#            "/tdm-new/"
ddpg_path = "/mnt/data-backup-12-02-2017/papers/iclr2018/sawyer_new/ddpg/"
tdm_path = "/mnt/data-backup-12-02-2017/papers/iclr2018/sawyer_new/tdm/"

ddpg_csvs = get_all_csv(ddpg_path)
# naf_csvs = get_all_csv(naf_path)
tdm_csvs = get_all_csv(tdm_path)


MAX_ITERS = 100
plt.figure()
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
for trials, name, key in [
    (tdm_csvs, 'TDM',
     'Test_Final_Distance_from_Desired_End_Effector_Position_Mean'),
    (ddpg_csvs, 'DDPG',
     'Test_Final_Distance_from_Desired_End_Effector_Position_Mean'),
    # (naf_csvs, 'NAF',
    #  'Test_Last_N_Step_Distance_from_Desired_End_Effector_Position_Mean'),
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
    if name == 'TDM':
        # Murtaza said the last trial was messed up by external factors
        costs = costs[:, :-1]
    costs = costs[:, :min(costs.shape[1], MAX_ITERS)]
    mean = np.mean(costs, axis=0)
    std = np.std(costs, axis=0)
    epochs = np.arange(0, len(costs[0]))
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.1)
    if name == 'TDM':
        plt.plot(epochs, mean, label=name)#, color=colors[0])
    else:
        plt.plot(epochs, mean, label=name)#, color=colors[1])
    # plt.plot(epochs, costs[0, :], label=name)
    # plt.plot(epochs, costs[1, :], label=name)
    # plt.plot(epochs, costs[2, :], label=name)


plt.xlabel("Environment Steps (x1000)")
plt.ylabel("Final Distance to Goal (m)")
# plt.title(r"Mean Distance to Goal vs Environment Samples")
plt.legend()
plt.savefig('results/iclr2018/sawyer.jpg')
plt.show()
