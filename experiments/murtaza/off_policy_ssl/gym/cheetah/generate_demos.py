from railrl.demos.collect_demo import collect_demos_fixed
from railrl.misc.asset_loader import load_local_or_remote_file
import gym

if __name__ == '__main__':
    data = load_local_or_remote_file('01-12-sac-mujoco-envs/01-12-sac_mujoco_envs_2020_01_12_22_34_03_id000--s52204/params.pkl')
    data = load_local_or_remote_file('02-20-sac-mujoco-envs-unnormalized-run-longer/02-20-sac_mujoco_envs_unnormalized_run_longer_2020_02_20_23_55_13_id000--s39214/params.pkl')
    env = data['exploration/env']
    policy = data['exploration/policy']
    collect_demos_fixed(env, policy, "data/local/demos/hc_action_noise_5.npy", N=5, horizon=1000, threshold=9000, render=False)

    # data = load_local_or_remote_file(
        # '/home/murtaza/research/railrl/data/local/03-04-bc-hc-v2/03-04-bc_hc_v2_2020_03_04_17_57_54_id000--s90897/bc.pkl')
    # env = gym.make('HalfCheetah-v2')
    # policy = data.cpu()
    # collect_demos_fixed(env, policy, "data/local/demos/hc_off_policy_100.npy", N=100, horizon=1000, threshold=8000,
                        # render=False)
