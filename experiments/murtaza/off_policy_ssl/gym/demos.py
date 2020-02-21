from railrl.demos.collect_demo import collect_demos_fixed
from railrl.misc.asset_loader import load_local_or_remote_file
import gym

if __name__ == '__main__':
    # data = load_local_or_remote_file('01-12-sac-mujoco-envs/01-12-sac_mujoco_envs_2020_01_12_22_34_03_id000--s52204/params.pkl')
    data = load_local_or_remote_file('02-17-sac-mujoco-envs-unnormalized/02-17-sac_mujoco_envs_unnormalized_2020_02_18_01_07_07_id000--s36081/params.pkl')
    # env = data['exploration/env']
    env = gym.make('HalfCheetah-v2')
    policy = data['exploration/policy']
    collect_demos_fixed(env, policy, "data/local/demos/hc_action_noise_1000.npy", N=1000, horizon=1000, threshold=9000, render=False)
    #
    # data = load_local_or_remote_file(
    #     '01-12-sac-mujoco-envs/01-12-sac_mujoco_envs_2020_01_12_22_34_04_id005--s76863/params.pkl')
    # env = data['evaluation/env']
    # policy = data['exploration/policy']
    # collect_demos_fixed(env, policy, "data/local/demos/hopper_action_noise_1000.npy", N=1000, horizon=1000, threshold=3500, render=False)
    #
    # data = load_local_or_remote_file(
    #     '01-12-sac-mujoco-envs/01-12-sac_mujoco_envs_2020_01_12_22_34_02_id003--s37740/params.pkl')
    # env = data['evaluation/env']
    # policy = data['exploration/policy']
    # collect_demos_fixed(env, policy, "data/local/demos/ant_action_noise_1000.npy", N=1000, horizon=1000, threshold=6000, render=False)
    #
    # data = load_local_or_remote_file(
    #     '01-12-sac-mujoco-envs/01-12-sac_mujoco_envs_2020_01_12_22_34_03_id004--s15589/params.pkl')
    # env = data['evaluation/env']
    # policy = data['exploration/policy']
    # collect_demos_fixed(env, policy, "data/local/demos/walker_action_noise_1000.npy", N=1000, horizon=1000, threshold=6000, render=False)
    #
    # data = load_local_or_remote_file(
    #     '01-12-sac-mujoco-envs/01-12-sac_mujoco_envs_2020_01_12_22_34_03_id004--s15589/params.pkl')
    # env = data['evaluation/env']
    # policy = data['exploration/policy']
    # collect_demos_fixed(env, policy, "data/local/demos/humanoid_action_noise_1000.npy", N=1000, horizon=1000,
    #                     threshold=6000, render=False)

