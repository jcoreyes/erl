from multiworld.envs.pygame.pick_and_place import PickAndPlaceEnv

import numpy as np
import time

from rlkit.misc.asset_loader import local_path_from_s3_or_local_path
import joblib
import os.path as osp

from rlkit.torch.core import torch_ify, np_ify

import rlkit.torch.pytorch_util as ptu
import os

def create_demo_dataset(env, n=1000, render=False):
    state_dim = env.observation_space.spaces['state_achieved_goal'].low.size

    list_of_waypoints = np.zeros((n, 5, state_dim))

    # data collection
    for i in range(n):
        obs_dict = env.reset()
        obs = obs_dict['state_observation']
        goal = obs_dict['state_desired_goal']

        for step in range(5):
            wp = obs.copy()
            if step > 0:
                start_idx = step * 2
                end_idx = step * 2 + 2
                wp[0:2] = goal[start_idx:end_idx] # move hand
                wp[2:end_idx] = goal[2:end_idx] # move objs

            list_of_waypoints[i][step] = wp

            if render:
                env.set_to_goal({
                    'state_desired_goal': wp
                })
                env.render()
                time.sleep(1.5)

    return list_of_waypoints


env_class=PickAndPlaceEnv
env_kwargs=dict(
    num_objects=4,

    # Environment dynamics
    action_scale=1.0,
    ball_radius=0.75, #1.
    boundary_dist=4,
    object_radius=0.50,
    min_grab_distance=0.5,
    walls=None,
    # Rewards
    action_l2norm_penalty=0,
    reward_type="dense", #dense_l1
    success_threshold=0.60,
    # Reset settings
    fixed_goal=None,
    # Visualization settings
    images_are_rgb=True,
    render_dt_msec=0,
    render_onscreen=False,
    render_size=256,
    show_goal=True,
    # get_image_base_render_size=(48, 48),
    # Goal sampling
    goal_samplers=None,
    goal_sampling_mode='random',
    num_presampled_goals=10000,
    object_reward_only=True,

    init_position_strategy='random',
)

env = env_class(**env_kwargs)

ckpts = [
    'pg-4obj/06-06-larger-nupo/06-06-larger-nupo_2020_06_07_03_24_58_id000--s75821',
    'pg-4obj/06-06-larger-nupo/06-06-larger-nupo_2020_06_07_03_24_59_id000--s96595',
    'pg-4obj/06-06-larger-nupo/06-06-larger-nupo_2020_06_07_03_24_58_id000--s16595',
]

epoch = None

qfs = []
policies = []
for ckpt in ckpts:
    if epoch is not None:
        filename = local_path_from_s3_or_local_path(osp.join(ckpt, 'itr_%d.pkl' % epoch))
    else:
        filename = local_path_from_s3_or_local_path(osp.join(ckpt, 'params.pkl'))

    print("Loading ckpt from", filename)
    data = joblib.load(filename)
    qfs.append(data['trainer/qf1'])
    policies.append(data['evaluation/policy'])

n = 1
list_of_waypoints = create_demo_dataset(env, n=n, render=False)

gpu_id = 1
ptu.set_gpu_mode(True, gpu_id=gpu_id)
os.environ['gpu_id'] = str(gpu_id)
print(ptu.gpu_enabled())


v_val_means = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        states, goals = list_of_waypoints[:,i,:], list_of_waypoints[:,j,:]
        if i > j:
            states, goals = goals, states
        masks = np.zeros(states.shape)
        start_idx = 2
        end_idx = 10
        # if i < j:
        #     pass
        # elif i == j:
        #     # end_idx = end_idx - 2
        #     start_idx = start_idx - 2
        # if i > j:
        #     start_idx, end_idx = end_idx, start_idx
        # print(i, j, start_idx, end_idx)
        masks[:,start_idx:end_idx] = 1
        obs = np.concatenate((states, goals, masks), axis=1)
        list_of_v_vals = []
        for k in range(len(ckpts)):
            actions = policies[k].get_actions(obs)

            v_vals = qfs[k](
                torch_ify(obs),
                torch_ify(actions),
            )
            v_vals = np_ify(v_vals) / 100
            list_of_v_vals.append(v_vals)
        v_val_means[i][j] = np.mean(list_of_v_vals)

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

df_cm = pd.DataFrame(v_val_means.astype(int), range(5), range(5))
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.0) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, fmt='g') # font size

# if epoch is not None:
#     plt.title('%d k' % epoch)
# else:
#     plt.title('3000 k')

plt.title('demo 5')

plt.show()

