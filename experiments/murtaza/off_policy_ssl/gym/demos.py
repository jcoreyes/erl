from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from railrl.demos.collect_demo import collect_demos_fixed
from railrl.misc.asset_loader import load_local_or_remote_file

if __name__ == '__main__':
    data = load_local_or_remote_file('01-12-sac-mujoco-envs/01-12-sac_mujoco_envs_2020_01_12_22_34_03_id000--s52204/params.pkl')
    env = data['evaluation/env']
    policy = data['exploration/policy']
    collect_demos_fixed(env, policy, "data/local/demos/hc_action_noise_1000.npy", N=1000, horizon=1000, threshold=10000, render=False)
    #
    data = load_local_or_remote_file(
        '01-12-sac-mujoco-envs/01-12-sac_mujoco_envs_2020_01_12_22_34_04_id005--s76863/params.pkl')
    env = data['evaluation/env']
    policy = data['exploration/policy']
    collect_demos_fixed(env, policy, "data/local/demos/hopper_action_noise_1000.npy", N=1000, horizon=1000, threshold=3500, render=False)

    data = load_local_or_remote_file(
        '01-12-sac-mujoco-envs/01-12-sac_mujoco_envs_2020_01_12_22_34_02_id003--s37740/params.pkl')
    env = data['evaluation/env']
    policy = data['exploration/policy']
    collect_demos_fixed(env, policy, "data/local/demos/ant_action_noise_1000.npy", N=1000, horizon=1000, threshold=6000, render=False)

    data = load_local_or_remote_file(
        '01-12-sac-mujoco-envs/01-12-sac_mujoco_envs_2020_01_12_22_34_03_id004--s15589/params.pkl')
    env = data['evaluation/env']
    policy = data['exploration/policy']
    collect_demos_fixed(env, policy, "data/local/demos/walker_action_noise_1000.npy", N=1000, horizon=1000, threshold=6000, render=False)

    data = load_local_or_remote_file(
        '01-12-sac-mujoco-envs/01-12-sac_mujoco_envs_2020_01_12_22_34_03_id004--s15589/params.pkl')
    env = data['evaluation/env']
    policy = data['exploration/policy']
    collect_demos_fixed(env, policy, "data/local/demos/humanoid_action_noise_1000.npy", N=1000, horizon=1000,
                        threshold=6000, render=False)

