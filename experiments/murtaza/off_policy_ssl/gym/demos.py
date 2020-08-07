from rlkit.demos.collect_demo import collect_demos_fixed
from rlkit.misc.asset_loader import load_local_or_remote_file
import gym

if __name__ == '__main__':
    # data = load_local_or_remote_file('01-12-sac-mujoco-envs/01-12-sac_mujoco_envs_2020_01_12_22_34_03_id000--s52204/params.pkl')
    # data = load_local_or_remote_file('02-20-sac-mujoco-envs-unnormalized-run-longer/02-20-sac_mujoco_envs_unnormalized_run_longer_2020_02_20_23_55_13_id000--s39214/params.pkl')
    # # env = data['exploration/env']
    # env = gym.make('HalfCheetah-v2')
    # policy = data['exploration/policy']
    # collect_demos_fixed(env, policy, "data/local/demos/hc_action_noise_1000.npy", N=1000, horizon=1000, threshold=9000, render=False)
    #
    data = load_local_or_remote_file(
        '02-17-sac-mujoco-envs-unnormalized/02-17-sac_mujoco_envs_unnormalized_2020_02_18_01_07_08_id005--s24204/params.pkl')
    env = data['evaluation/env']
    policy = data['exploration/policy']
    collect_demos_fixed(env, policy, "data/local/demos/hopper_action_noise_1000.npy", N=1000, horizon=1000, threshold=3000, render=False)
    
    data = load_local_or_remote_file(
        '02-17-sac-mujoco-envs-unnormalized/02-17-sac_mujoco_envs_unnormalized_2020_02_18_01_07_32_id003--s29410/params.pkl')
    env = data['evaluation/env']
    policy = data['exploration/policy']
    collect_demos_fixed(env, policy, "data/local/demos/ant_action_noise_1000.npy", N=1000, horizon=1000, threshold=5000, render=False)
    
    data = load_local_or_remote_file(
        '02-17-sac-mujoco-envs-unnormalized/02-17-sac_mujoco_envs_unnormalized_2020_02_18_01_07_08_id004--s82441/params.pkl')
    env = data['evaluation/env']
    policy = data['exploration/policy']
    collect_demos_fixed(env, policy, "data/local/demos/walker_action_noise_1000.npy", N=1000, horizon=1000, threshold=5000, render=False)
    #
    # data = load_local_or_remote_file(
    #     '01-12-sac-mujoco-envs/01-12-sac_mujoco_envs_2020_01_12_22_34_03_id004--s15589/params.pkl')
    # env = data['evaluation/env']
    # policy = data['exploration/policy']
    # collect_demos_fixed(env, policy, "data/local/demos/humanoid_action_noise_1000.npy", N=1000, horizon=1000,
    #                     threshold=6000, render=False)

