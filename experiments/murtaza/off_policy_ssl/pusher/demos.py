from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in

from railrl.demos.collect_demo import collect_demos
from railrl.misc.asset_loader import load_local_or_remote_file

if __name__ == '__main__':
    data = load_local_or_remote_file('/home/murtaza/research/railrl/data/doodads3/11-16-pusher-state-td3-sweep-params-policy-update-period/11-16-pusher_state_td3_sweep_params_policy_update_period_2019_11_17_00_29_02_id000--s90018/params.pkl')
    env = data['evaluation/env']
    policy = data['trainer/trained_policy']
    image_env = ImageEnv(
        env,
        48,
        init_camera=sawyer_init_camera_zoomed_in,
        transpose=True,
        normalize=True,
    )
    collect_demos(image_env, policy, "data/local/demos/pusher_demos_1000.npy", N=1000, horizon=50, threshold=.01, add_action_noise=False, key='puck_distance')
