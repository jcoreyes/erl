from railrl.demos.collect_demo import collect_demos_fixed
from railrl.misc.asset_loader import load_local_or_remote_file
import gym
from railrl.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic

if __name__ == '__main__':
    data = load_local_or_remote_file('/home/murtaza/research/railrl/data/doodads3/03-09-sac-mujoco-envs-unnormalized-run-longer/03-09-sac_mujoco_envs_unnormalized_run_longer_2020_03_10_00_01_32_id000--s80895/params.pkl')
    env = data['exploration/env']
    policy = data['exploration/policy']
    collect_demos_fixed(env, policy, "data/local/demos/walker_action_noise_1000.npy", N=1000, horizon=1000, threshold=4500, render=False)

    # data = load_local_or_remote_file('/home/murtaza/research/railrl/data/doodads3/03-09-sac-mujoco-envs-unnormalized-run-longer/03-09-sac_mujoco_envs_unnormalized_run_longer_2020_03_10_00_01_32_id000--s80895/params.pkl')
    # env = data['exploration/env']
    # policy = data['exploration/policy']
    # collect_demos_fixed(env, policy, "data/local/demos/walker_action_noise_10.npy", N=10, horizon=1000, threshold=4500, render=False)
    
    # data = load_local_or_remote_file('/home/murtaza/research/railrl/data/doodads3/03-09-sac-mujoco-envs-unnormalized-run-longer/03-09-sac_mujoco_envs_unnormalized_run_longer_2020_03_10_00_01_32_id000--s80895/params.pkl')
    # env = data['exploration/env']
    # policy = data['exploration/policy']
    # collect_demos_fixed(env, policy, "data/local/demos/walker_action_noise_15.npy", N=15, horizon=1000, threshold=4500, render=False)

    # data = load_local_or_remote_file('/home/murtaza/research/railrl/data/doodads3/03-09-sac-mujoco-envs-unnormalized-run-longer/03-09-sac_mujoco_envs_unnormalized_run_longer_2020_03_10_00_01_32_id000--s80895/params.pkl')
    # env = data['exploration/env']
    # policy = data['exploration/policy']
    # collect_demos_fixed(env, policy, "data/local/demos/walker_action_noise_25.npy", N=25, horizon=1000, threshold=4500, render=False)

    # data = load_local_or_remote_file(
        # '/home/murtaza/research/railrl/data/doodads3/03-08-bc-walker-gym-v1/03-08-bc_walker_gym_v1_2020_03_08_19_22_00_id000--s52831/bc.pkl')
    # policy = data.cpu()
    # collect_demos_fixed(env, policy, "data/local/demos/walker_off_policy_10_demos_100.npy", N=100, horizon=1000, threshold=-1,
                        # render=False)

    # data = load_local_or_remote_file(
        # '/home/murtaza/research/railrl/data/doodads3/03-08-bc-walker-gym-v1/03-08-bc_walker_gym_v1_2020_03_08_19_22_10_id000--s29670/bc.pkl')
    # policy = data.cpu()
    # collect_demos_fixed(env, policy, "data/local/demos/walker_off_policy_15_demos_100.npy", N=100, horizon=1000, threshold=-1,
                        # render=False)

    # data = load_local_or_remote_file(
        # '/home/murtaza/research/railrl/data/doodads3/03-08-bc-walker-gym-v1/03-08-bc_walker_gym_v1_2020_03_08_19_22_18_id000--s69927/bc.pkl')
    # policy = data.cpu()
    # collect_demos_fixed(env, policy, "data/local/demos/walker_off_policy_25_demos_100.npy", N=100, horizon=1000, threshold=-1,
                        # render=False)

