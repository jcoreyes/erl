import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from torch import optim

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.simple import RandomPolicy
from rlkit.samplers.util import rollout
from rlkit.state_distance.tdm_networks import TdmQf
from rlkit.state_distance.util import merge_into_flat_obs
from rlkit.torch.core import PyTorchModule

TDM_PATH = '/home/vitchyr/git/railrl/data/local/01-22-dev-sac-tdm-launch/01' \
           '-22-dev-sac-tdm-launch_2018_01_22_13_31_47_0000--s-3096/params.pkl'
# ddpg TDM trained with only mtau = 0
TDM_PATH = '/home/vitchyr/git/railrl/data/doodads3/01-23-reacher-full-ddpg' \
           '-tdm-mtau-0/01-23-reacher-full-ddpg-tdm-mtau-0-id1-s49343/params.pkl'
MODEL_PATH = '/home/vitchyr/git/railrl/data/local/01-19-reacher-model-based' \
             '/01-19-reacher-model-based_2018_01_19_15_54_27_0000--s-983077/params.pkl'

TDM_PATH = '/home/vitchyr/git/railrl/data/doodads3/02-08-reacher7dof-sac-squared-distance-sweep-qf-activation-2/02-08-reacher7dof-sac-squared-distance-sweep-qf-activation-2-id1-s5793/params.pkl'
MODEL_PATH = '/home/vitchyr/git/railrl/data/local/01-27-reacher-full-mpcnn-H1/01-27-reacher-full-mpcnn-H1_2018_01_27_17_59_04_0000--s-96642/params.pkl'


K = 100

class ImplicitModel(PyTorchModule):
    def __init__(self, qf, vf):
        super().__init__()
        self.qf = qf
        self.vf = vf

    def forward(self, obs, goals, taus, actions):
        flat_obs = merge_into_flat_obs(obs, goals, taus)
        if self.vf is None:
            return self.qf(flat_obs, actions)
        else:
            return self.qf(flat_obs, actions) - self.vf(flat_obs)


def expand_np_to_var(array, requires_grad=False):
    array_expanded = np.repeat(
        np.expand_dims(array, 0),
        K,
        axis=0
    )
    return ptu.np_to_var(array_expanded, requires_grad=requires_grad)


def get_feasible_goal(env, tdm, ob, action):
    obs = expand_np_to_var(ob)
    actions = expand_np_to_var(action)
    taus = expand_np_to_var(np.array([0]))
    goals = expand_np_to_var(
        env.convert_ob_to_goal(ob), requires_grad=True
    )
    goals.data = goals.data + torch.randn(goals.shape)
    optimizer = optim.RMSprop([goals], lr=1e-2)
    print("--")
    for _ in range(1000):
        distances = - tdm(obs, goals, taus, actions)
        distance = distances.mean()
        print(ptu.get_numpy(distance.mean())[0])
        optimizer.zero_grad()
        distance.backward()
        optimizer.step()

    goals = ptu.get_numpy(goals)
    min_i = 0
    if isinstance(tdm.qf, TdmQf):
        return tdm.qf.eval_np(
            np.hstack((
                ob,
                goals[min_i, :],
                np.zeros(1)
            ))[None],
            action[None],
            return_internal_prediction=True,
        )
    return goals[min_i, :]


def main():
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    tdm_data = joblib.load(TDM_PATH)
    env = tdm_data['env']
    qf = tdm_data['qf']
    variant_path = Path(TDM_PATH).parents[0] / 'variant.json'
    variant = json.load(variant_path.open())
    reward_scale = variant['sac_tdm_kwargs']['base_kwargs']['reward_scale']
    tdm = ImplicitModel(qf, None)
    random_policy = RandomPolicy(env.action_space)
    H = 10
    path = rollout(env, random_policy, max_path_length=H)

    model_distance_preds = []
    tdm_distance_preds = []
    for ob, action, next_ob in zip(
            path['observations'],
            path['actions'],
            path['next_observations'],
    ):
        obs = ob[None]
        actions = action[None]
        next_feature = env.convert_ob_to_goal(next_ob)
        model_next_ob_pred = ob + model.eval_np(obs, actions)[0]
        model_distance_pred = np.abs(
            env.convert_ob_to_goal(model_next_ob_pred) -
            next_feature
        )

        tdm_next_feature_pred = get_feasible_goal(env, tdm, ob, action)
        tdm_distance_pred = np.abs(
            tdm_next_feature_pred - next_feature
        )

        model_distance_preds.append(model_distance_pred)
        tdm_distance_preds.append(tdm_distance_pred)

    model_distances = np.array(model_distance_preds)
    tdm_distances = np.array(tdm_distance_preds)
    ts = np.arange(len(model_distance_preds))
    num_dim = model_distances[0].size
    ind = np.arange(num_dim)
    width = 0.35

    fig, ax = plt.subplots()
    means = model_distances.mean(axis=0)
    stds = model_distances.std(axis=0)
    rects1 = ax.bar(ind, means, width, color='r', yerr=stds)

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
    plt.xlabel("Time")
    plt.ylabel("Absolute Error")
    plt.title("Model")
    plt.legend()

    plt.subplot(2, 1, 2)
    for i in range(num_dim):
        plt.plot(
            ts,
            tdm_distances[:, i],
            label=str(i),
        )
    plt.xlabel("Time")
    plt.ylabel("Absolute Error")
    plt.title("TDM")
    plt.legend()
    plt.show()

    goal = env.convert_ob_to_goal(path['observations'][H//2].copy())
    path = rollout(env, random_policy, max_path_length=H)

    model_distance_preds = []
    tdm_distance_preds = []
    for ob, action, next_ob in zip(
            path['observations'],
            path['actions'],
            path['next_observations'],
    ):
        model_next_ob_pred = ob + model.eval_np(ob[None], action[None])[0]
        model_distance_pred = np.linalg.norm(
            env.convert_ob_to_goal(model_next_ob_pred)
            - goal
        )

        tdm_distance_pred = tdm.eval_np(
            ob[None],
            goal[None],
            np.zeros((1, 1)),
            action[None],
        )[0] / reward_scale

        model_distance_preds.append(model_distance_pred)
        tdm_distance_preds.append(tdm_distance_pred)

    fig, ax = plt.subplots()
    means = model_distances.mean(axis=0)
    stds = model_distances.std(axis=0)
    rects1 = ax.bar(ind, means, width, color='r', yerr=stds)

    means = tdm_distances.mean(axis=0)
    stds = tdm_distances.std(axis=0)
    rects2 = ax.bar(ind + width, means, width, color='y', yerr=stds)
    ax.legend((rects1[0], rects2[0]), ('Model', 'TDM'))
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Error To Random Goal State")
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(list(map(str, ind)))

    plt.show()


if __name__ == '__main__':
    main()
