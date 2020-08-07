from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera
from rlkit.demos.collect_demo import collect_demos
from rlkit.misc.asset_loader import load_local_or_remote_file

if __name__ == '__main__':
    data = load_local_or_remote_file('/home/murtaza/research/railrl/data/doodads3/11-16-pickup-state-td3-sweep-params-policy-update-period/11-16-pickup_state_td3_sweep_params_policy_update_period_2019_11_17_00_26_46_id000--s87995/params.pkl')
    env = data['evaluation/env']
    policy = data['trainer/trained_policy']
    image_env = ImageEnv(
        env,
        48,
        init_camera=sawyer_pick_and_place_camera,
        transpose=True,
        normalize=True,
    )
    collect_demos(image_env, policy, "data/local/demos/pickup_demos_action_noise_1000.npy", N=1000, horizon=50, threshold=.02, add_action_noise=False, key='obj_distance', render=False, noise_sigma=.5)
