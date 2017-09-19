import os.path as osp
import time

import joblib
import numpy as np
import tensorflow as tf
from gym.monitoring.video_recorder import ImageEncoder

n_paths = 5
path_length = 300
frame_size = (500, 500)
viewer_settings = dict(
    distance=5,
    trackbodyid=0
)

ddpg_name_to_snapshot_path = {
    'push-middle': '/home/vitchyr/git/rllab-rail/railrl/data/papers/icra2017/09-11_pusher-3dof-vertical-2_2017_09_11_23_24_08_0017/itr_50.pkl',
    'push-left': '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-12-pusher-3dof-vertical-l2-left/09-12_pusher-3dof-vertical-l2-left_2017_09_12_15_56_43_0001/itr_40.pkl',
    'push-right': '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-12-pusher-3dof-vertical-l2-right/09-12_pusher-3dof-vertical-l2-right_2017_09_12_15_57_16_0001/itr_40.pkl',
    'push-bottom': '/home/vitchyr/git/rllab-rail/railrl/data/papers/icra2017/09-11_pusher-3dof-horizontal-2_2017_09_11_23_23_50_0039/itr_50.pkl',
    'push-bottom-left': '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-bottom-left/09-14_pusher-3dof-reacher-bottom-left_2017_09_14_17_12_41_0001/params.pkl',
    'push-bottom-middle': '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-bottom-middle/09-14_pusher-3dof-reacher-bottom-middle_2017_09_14_17_13_07_0001/params.pkl',
    'push-bottom-right': '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-bottom-right/09-14_pusher-3dof-reacher-bottom-right_2017_09_14_17_13_22_0001/params.pkl',
    'merge-bottom-left': '/home/vitchyr/git/rllab-rail/railrl/data/local/09-14-combine-policies--left-bottom/09-14_combine-policies--left-bottom_2017_09_14_14_39_36_0000--s-4401/params.pkl',
    'merge-bottom-middle': '/home/vitchyr/git/rllab-rail/railrl/data/local/09-14-combine-policies--middle-bottom/09-14_combine-policies--middle-bottom_2017_09_14_14_39_01_0000--s-2893/params.pkl',
    'merge-bottom-right': '/home/vitchyr/git/rllab-rail/railrl/data/local/09-14-combine-policies--right-bottom/09-14_combine-policies--right-bottom_2017_09_14_14_38_47_0000--s-984/params.pkl',
}
naf_name_to_snapshot_path = dict(
    # push_left=(
    #     '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-naf-yolo-left/09-14_pusher-3dof-reacher-naf-yolo_left_2017_09_14_17_52_45_0010/params.pkl'
    # ),
    # push_right=(
    #     '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-naf-yolo-right/09-14_pusher-3dof-reacher-naf-yolo_right_2017_09_14_17_52_45_0016/params.pkl'
    # ),
    # push_middle=(
    #     '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-naf-yolo-middle/09-14_pusher-3dof-reacher-naf-yolo_middle_2017_09_14_17_52_45_0013/params.pkl'
    # ),
    # push_bottom=(
    #     '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-naf-yolo-bottom/09-14_pusher-3dof-reacher-naf-yolo_bottom_2017_09_14_17_52_45_0019/params.pkl'
    # ),
    merge_bottom_left=(
        '/home/vitchyr/git/rllab-rail/railrl/data/local/09-14-1-combine-naf-policies-left/09-14_1-combine-naf-policies-left_2017_09_14_21_42_24_0000--s-68077/params.pkl'
    ),
    merge_bottom_right=(
        '/home/vitchyr/git/rllab-rail/railrl/data/local/09-14-1-combine-naf-policies-right/09-14_1-combine-naf-policies-right_2017_09_14_21_42_29_0000--s-42677/params.pkl'
    ),
    merge_bottom_middle=(
        '/home/vitchyr/git/rllab-rail/railrl/data/local/09-14-1-combine-naf-policies-middle/09-14_1-combine-naf-policies-middle_2017_09_14_21_42_27_0000--s-91696/params.pkl'
    ),
    # push_bottom_left=(
    #     '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-naf-yolo-bottom-left/09-14_pusher-3dof-reacher-naf-yolo_bottom-left_2017_09_14_17_52_45_0001/params.pkl'
    # ),
    # push_bottom_right=(
    #     '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-naf-yolo-bottom-right/09-14_pusher-3dof-reacher-naf-yolo_bottom-right_2017_09_14_17_52_45_0007/params.pkl'
    # ),
    # push_bottom_middle=(
    #     '/home/vitchyr/git/rllab-rail/railrl/data/s3/09-14-pusher-3dof-reacher-naf-yolo-bottom-middle/09-14_pusher-3dof-reacher-naf-yolo_bottom-middle_2017_09_14_17_52_45_0005/params.pkl'
    # ),
)

naf_output_path = '/home/vitchyr/git/rllab-rail/railrl/data/papers/icra2017/results/pusher/naf/videos/'
ddpg_output_path = '/home/vitchyr/git/rllab-rail/railrl/data/papers/icra2017/results/pusher/ddpg/videos/'


def rollout(env, policy, path_length, render=False, speedup=10, callback=None,
            render_mode='human', viewer_kwargs=None):
    if viewer_kwargs is None:
        viewer_kwargs = {}

    if render_mode == 'rgb_array':
        ims = list()

    Da = env.action_space.flat_dim
    Do = env.observation_space.flat_dim

    observation = env.reset()
    policy.reset()

    observations = np.zeros((path_length, Do))
    actions = np.zeros((path_length, Da))
    terminals = np.zeros((path_length,))
    rewards = np.zeros((path_length,))
    all_infos = list()
    t = 0  # To make edge case path_length=0 work.
    for t in range(t, path_length):

        action, _ = policy.get_action(observation)

        if callback is not None:
            callback(observation, action)

        next_obs, reward, terminal, info = env.step(action)

        all_infos.append(info)

        import ipdb; ipdb.set_trace()
        actions[t, :] = action
        terminals[t] = terminal
        rewards[t] = reward
        observations[t, :] = observation

        observation = next_obs

        if render:
            if render_mode == 'rgb_array':
                ims.append(env.render(
                    mode=render_mode,
                    viewer_kwargs=viewer_kwargs
                ))
                # import ipdb; ipdb.set_trace()
            else:
                env.render(render_mode)
                time_step = 0.05
                time.sleep(time_step / speedup)

        if terminal:
            break

    last_obs = observation

    concat_infos = dict()
    for key in all_infos[0].keys():
        all_vals = [np.array(info[key])[None] for info in all_infos]
        concat_infos[key] = np.concatenate(all_vals)

    path = dict(
        last_obs=last_obs,
        dones=terminals[:t+1],
        actions=actions[:t+1],
        observations=observations[:t+1],
        rewards=rewards[:t+1],
        env_infos=concat_infos
    )

    if render_mode == 'rgb_array':
        # import ipdb; ipdb.set_trace()
        path['ims'] = np.stack(ims, axis=0)

    return path


def rollouts(env, policy, path_length, n_paths, render=False):
    paths = list()
    for i in range(n_paths):
        paths.append(rollout(env, policy, path_length, render))

    return paths


name_to_snapshot_path = ddpg_name_to_snapshot_path
output_path = ddpg_output_path
name_to_snapshot_path = naf_name_to_snapshot_path
# output_path = naf_output_path
output_path = '/tmp'

for name, snapshot_path in name_to_snapshot_path.items():
    data = joblib.load(snapshot_path)
    if "naf_policy" in data:
        policy = data["naf_policy"]
    else:
        policy = data["policy"]
    env = data["env"]
    ims = list()

    # Dummy rollout for the camera to settle
    path = rollout(env, policy, 30, True, 999, None, 'rgb_array', viewer_settings)

    for p in range(n_paths):
        path = rollout(
            env=env,
            policy=policy,
            path_length=path_length,
            speedup=9999,
            render=True,
            render_mode='rgb_array',
            viewer_kwargs=viewer_settings
        )
        ims.append(path['ims'])
    ims = np.concatenate(ims, axis=0)

    video_file = osp.join(output_path, name + '.mp4')

    encoder = ImageEncoder(
        output_path=video_file,
        frame_shape=frame_size + (3,),
        frames_per_sec=20
    )

    for im in ims:
        encoder.capture_frame(im)

    encoder.close()

    tf.reset_default_graph()

