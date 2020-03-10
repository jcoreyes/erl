from railrl.demos.collect_demo import collect_demos_fixed
from railrl.misc.asset_loader import load_local_or_remote_file
import gym
from railrl.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic

if __name__ == '__main__':
    # data = load_local_or_remote_file('02-17-sac-mujoco-envs-unnormalized/02-17-sac_mujoco_envs_unnormalized_2020_02_18_01_07_08_id004--s82441/params.pkl')
    # env = data['exploration/env']
    # policy = data['exploration/policy']
    # collect_demos_fixed(env, policy, "data/local/demos/walker_action_noise_10.npy", N=10, horizon=1000, threshold=5000, render=False)

    # data = load_local_or_remote_file('02-17-sac-mujoco-envs-unnormalized/02-17-sac_mujoco_envs_unnormalized_2020_02_18_01_07_08_id004--s82441/params.pkl')
    # env = data['exploration/env']
    # policy = data['exploration/policy']
    # collect_demos_fixed(env, policy, "data/local/demos/walker_action_noise_15.npy", N=15, horizon=1000, threshold=5000, render=False)
    
    data = load_local_or_remote_file('02-17-sac-mujoco-envs-unnormalized/02-17-sac_mujoco_envs_unnormalized_2020_02_18_01_07_08_id004--s82441/params.pkl')
    env = data['exploration/env']
    policy = data['exploration/policy']
    # collect_demos_fixed(env, policy, "data/local/demos/walker_action_noise_25.npy", N=25, horizon=1000, threshold=5000, render=False)

    data = load_local_or_remote_file(
        '/home/murtaza/research/railrl/data/doodads3/03-08-bc-walker-gym-v1/03-08-bc_walker_gym_v1_2020_03_08_19_22_00_id000--s52831/bc.pkl')
    policy = data.cpu()
    collect_demos_fixed(env, policy, "data/local/demos/walker_off_policy_10_demos_100.npy", N=100, horizon=1000, threshold=-1,
                        render=False)

    data = load_local_or_remote_file(
        '/home/murtaza/research/railrl/data/doodads3/03-08-bc-walker-gym-v1/03-08-bc_walker_gym_v1_2020_03_08_19_22_10_id000--s29670/bc.pkl')
    policy = data.cpu()
    collect_demos_fixed(env, policy, "data/local/demos/walker_off_policy_15_demos_100.npy", N=100, horizon=1000, threshold=-1,
                        render=False)

    data = load_local_or_remote_file(
        '/home/murtaza/research/railrl/data/doodads3/03-08-bc-walker-gym-v1/03-08-bc_walker_gym_v1_2020_03_08_19_22_18_id000--s69927/bc.pkl')
    policy = data.cpu()
    collect_demos_fixed(env, policy, "data/local/demos/walker_off_policy_25_demos_100.npy", N=100, horizon=1000, threshold=-1,
                        render=False)

