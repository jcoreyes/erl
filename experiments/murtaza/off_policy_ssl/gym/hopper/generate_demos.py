from rlkit.demos.collect_demo import collect_demos_fixed
from rlkit.misc.asset_loader import load_local_or_remote_file
import gym

if __name__ == '__main__':
    data = load_local_or_remote_file('02-20-sac-mujoco-envs-unnormalized-run-longer/02-20-sac_mujoco_envs_unnormalized_run_longer_2020_02_20_23_55_13_id000--s39214/params.pkl')
    env = data['exploration/env']
    policy = data['exploration/policy']
    collect_demos_fixed(env, policy, "data/local/demos/hc_action_noise_25.npy", N=25, horizon=1000, threshold=9000, render=False)

    # data = load_local_or_remote_file(
        # '/home/murtaza/research/rlkit/data/local/03-04-bc-hc-v2/03-04-bc_hc_v2_2020_03_04_17_57_54_id000--s90897/bc.pkl')
    # env = gym.make('HalfCheetah-v2')
    # policy = data.cpu()
    # collect_demos_fixed(env, policy, "data/local/demos/hc_off_policy_100.npy", N=100, horizon=1000, threshold=8000,
                        # render=False)
    # data = load_local_or_remote_file(
        # '/home/murtaza/research/rlkit/data/doodads3/03-05-bc-hc-gym-v5/03-05-bc_hc_gym_v5_2020_03_06_06_55_43_id000--s42378/bc.pkl')
    # env = gym.make('HalfCheetah-v2')
    # policy = data.cpu()
    # collect_demos_fixed(env, policy, "data/local/demos/hc_off_policy_10_demos_100.npy", N=100, horizon=1000, threshold=-1,
                        # render=False)
    # data = load_local_or_remote_file(
        # '/home/murtaza/research/rlkit/data/doodads3/03-05-bc-hc-gym-v5/03-05-bc_hc_gym_v5_2020_03_06_06_55_41_id000--s9333/bc.pkl')
    # env = gym.make('HalfCheetah-v2')
    # policy = data.cpu()
    # collect_demos_fixed(env, policy, "data/local/demos/hc_off_policy_15_demos_100.npy", N=100, horizon=1000, threshold=-1,
                        # render=False)
    # data = load_local_or_remote_file(
        # '/home/murtaza/research/rlkit/data/doodads3/03-05-bc-hc-gym-v5/03-05-bc_hc_gym_v5_2020_03_06_06_55_40_id000--s26034/bc.pkl')
    # env = gym.make('HalfCheetah-v2')
    # policy = data.cpu()
    # collect_demos_fixed(env, policy, "data/local/demos/hc_off_policy_25_demos_100.npy", N=100, horizon=1000, threshold=-1,
                        # render=False)

