from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from railrl.demos.collect_demo import collect_demos_fixed
from railrl.misc.asset_loader import load_local_or_remote_file

if __name__ == '__main__':
    data = load_local_or_remote_file('/home/murtaza/research/railrl/data/doodads3/12-05-sac-mujoco-envs-v1/12-05-sac_mujoco_envs_v1_2019_12_06_02_08_27_id000--s22176/params.pkl')
    env = data['evaluation/env']
    policy = data['exploration/policy']
    collect_demos_fixed(env, policy, "data/local/demos/gym_1000.npy", N=1000, horizon=1000, threshold=9000, add_action_noise=False, render=False, noise_sigma=0.0)
