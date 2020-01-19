from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
import numpy as np
from railrl.demos.collect_demo import collect_demos_fixed
from railrl.misc.asset_loader import load_local_or_remote_file

from railrl.launchers.experiments.ashvin.awr_sac_rl import ENV_PARAMS

if __name__ == '__main__':
    data = load_local_or_remote_file('ashvin/icml2020/mujoco/reference/run1/id35/itr_1860.pkl')
    env = data['evaluation/env']
    policy = data['evaluation/policy']
    policy.to("cpu")
    env_name = "ant"
    outfile = "/home/ashvin/data/s3doodad/demos/icml2020/mujoco/%s.npy" % env_name
    horizon = ENV_PARAMS[env_name]['max_path_length']
    collect_demos_fixed(env, policy, outfile, N=100, horizon=horizon) # , threshold=.1, add_action_noise=False, key='puck_distance', render=True, noise_sigma=0.0)
