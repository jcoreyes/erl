from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from railrl.demos.collect_demo import collect_demos_fixed
from railrl.misc.asset_loader import load_local_or_remote_file

if __name__ == '__main__':
    data = load_local_or_remote_file('/home/mdalal/research/railrl/data/local/01-11-dev/01-11-dev_2020_01_11_22_58_04_id000--s80227/params.pkl')
    env = data['evaluation/env']
    policy = data['trainer/trained_policy']
    collect_demos_fixed(env, policy, "data/local/demos/gym_1000.npy", N=1000, horizon=50, threshold=.1, add_action_noise=False, render=False, noise_sigma=0.0)
