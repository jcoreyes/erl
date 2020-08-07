from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
import numpy as np
from rlkit.demos.collect_demo import collect_demos
from rlkit.misc.asset_loader import load_local_or_remote_file

if __name__ == '__main__':
    data = load_local_or_remote_file('/home/murtaza/research/railrl/data/doodads3/11-16-pusher-state-td3-sweep-params-policy-update-period/11-16-pusher_state_td3_sweep_params_policy_update_period_2019_11_17_00_28_45_id000--s62098/params.pkl')
    env = data['evaluation/env']
    policy = data['trainer/trained_policy']
    image_env = ImageEnv(
        env,
        48,
        init_camera=sawyer_init_camera_zoomed_in,
        transpose=True,
        normalize=True,
    )
    collect_demos(image_env, policy, "data/local/demos/pusher_demos_action_noise_1000.npy", N=1000, horizon=50, threshold=.1, add_action_noise=False, key='puck_distance', render=True, noise_sigma=0.0)
    # data = load_local_or_remote_file("demos/pusher_demos_1000.npy")
    # for i in range(100):
    #     goal = data[i]['observations'][49]['desired_goal']
    #     o = env.reset()
    #     path_length = 0
    #     while path_length < 50:
    #         env.set_goal({'state_desired_goal':goal})
    #         o = o['state_observation']
    #         new_obs = np.hstack((o, goal))
    #         a, agent_info = policy.get_action(new_obs)
    #         o, r, d, env_info = env.step(a)
    #         path_length += 1
    #     print(i, env_info['puck_distance'])
