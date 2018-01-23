import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim

import railrl.torch.pytorch_util as ptu
from railrl.policies.simple import RandomPolicy
from railrl.samplers.util import rollout
from railrl.state_distance.util import merge_into_flat_obs
from railrl.torch.core import PyTorchModule

TDM_PATH = '/home/vitchyr/git/railrl/data/local/01-22-dev-sac-tdm-launch/01' \
           '-22-dev-sac-tdm-launch_2018_01_22_13_31_47_0000--s-3096/params.pkl'
MODEL_PATH = '/home/vitchyr/git/railrl/data/local/01-19-reacher-model-based' \
             '/01-19-reacher-model-based_2018_01_19_15_54_27_0000--s-983077/params.pkl'


K = 100

class ImplicitModel(PyTorchModule):
    def __init__(self, qf, vf):
        self.quick_init(locals())
        super().__init__()
        self.qf = qf
        self.vf = vf

    def forward(self, obs, goals, taus, actions):
        flat_obs = merge_into_flat_obs(obs, goals, taus)
        return self.qf(flat_obs, actions) - self.vf(flat_obs)


def expand_np_to_var(array, requires_grad=False):
    array_expanded = np.repeat(
        np.expand_dims(array, 0),
        K,
        axis=0
    )
    return ptu.np_to_var(array_expanded, requires_grad=requires_grad)


def get_feasible_goal_states(tdm, ob, action):
    obs = expand_np_to_var(ob)
    actions = expand_np_to_var(action)
    taus = expand_np_to_var(np.array([1]))
    goal_states = expand_np_to_var(ob.copy(), requires_grad=True)
    goal_states.data = goal_states.data #+ torch.randn(goal_states.shape)
    optimizer = optim.RMSprop([goal_states], lr=1e-1)
    print("--")
    for _ in range(5):
        distances = - tdm(obs, goal_states, taus, actions)
        distance = distances.mean()
        print(ptu.get_numpy(distance.mean())[0])
        optimizer.zero_grad()
        distance.backward()
        optimizer.step()

    goal_states_np = ptu.get_numpy(goal_states)
    # distances_np = ptu.get_numpy(distances)
    # min_i = np.argmin(distances_np.sum(axis=1))
    min_i = 0
    return goal_states_np[min_i, :]


def main():
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    env = model_data['env']
    qf = joblib.load(TDM_PATH)['qf']
    vf = joblib.load(TDM_PATH)['vf']
    tdm = ImplicitModel(qf, vf)
    random_policy = RandomPolicy(env.action_space)
    path = rollout(env, random_policy, max_path_length=100)

    model_distance_preds = []
    tdm_distance_preds = []
    for ob, action, next_ob in zip(
            path['observations'],
            path['actions'],
            path['next_observations'],
    ):
        obs = ob[None]
        actions = action[None]
        next_obs = next_ob[None]
        model_next_ob_pred = ob + model.eval_np(obs, actions)
        model_distance_pred = np.abs(
            model_next_ob_pred - next_obs
        )[0]

        tdm_next_ob_pred = get_feasible_goal_states(tdm, ob, action)
        tdm_distance_pred = np.abs(
            tdm_next_ob_pred - next_obs
        )[0]
        # model_distance_pred = np.abs(
        #     (model_next_ob_pred - next_obs)[0]
        # )
        # tdm_distance_pred = -(
        #     tdm.eval_np(obs, actions, next_obs, np.zeros((1, 1)))
        # )[0]


        model_distance_preds.append(model_distance_pred)
        tdm_distance_preds.append(tdm_distance_pred)

    model_distances = np.array(model_distance_preds)
    tdm_distances = np.array(tdm_distance_preds)
    ts = np.arange(len(model_distance_preds))
    num_dim = model_distances[0].size
    ind = np.arange(num_dim)
    width = 0.35

    # ax = plt.subplot(2, 1, 1)
    fig, ax = plt.subplots()
    means = model_distances.mean(axis=0)
    stds = model_distances.std(axis=0)
    rects1 = ax.bar(ind, means, width, color='r', yerr=stds)

    # ax = plt.subplot(2, 1, 2)
    means = tdm_distances.mean(axis=0)
    stds = tdm_distances.std(axis=0)
    rects2 = ax.bar(ind + width, means, width, color='y', yerr=stds)
    ax.legend((rects1[0], rects2[0]), ('Model', 'TDM'))
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Absolute Error")
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(list(map(str, ind)))

    plt.show()

    plt.subplot(2, 1, 1)
    for i in range(num_dim):
        plt.plot(
            ts,
            model_distances[:, i],
            label=str(i),
        )
    plt.legend()

    plt.subplot(2, 1, 2)
    for i in range(tdm_distances[0].size):
        plt.plot(
            ts,
            tdm_distances[:, i],
            label=str(i),
        )
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
