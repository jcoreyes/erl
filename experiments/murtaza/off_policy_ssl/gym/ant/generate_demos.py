from rlkit.demos.collect_demo import collect_demos_fixed
from rlkit.misc.asset_loader import load_local_or_remote_file
import gym
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
if __name__ == '__main__':
    # data = load_local_or_remote_file('02-17-sac-mujoco-envs-unnormalized/02-17-sac_mujoco_envs_unnormalized_2020_02_18_01_07_32_id003--s29410/params.pkl')
    # env = data['exploration/env']
    # import ipdb; ipdb.set_trace()
    # policy = data['exploration/policy']
    # collect_demos_fixed(env, policy, "data/local/demos/ant_action_noise_10.npy", N=10, horizon=1000, threshold=5000, render=False)

    # data = load_local_or_remote_file('02-17-sac-mujoco-envs-unnormalized/02-17-sac_mujoco_envs_unnormalized_2020_02_18_01_07_32_id003--s29410/params.pkl')
    # env = data['exploration/env']
    # policy = data['exploration/policy']
    # collect_demos_fixed(env, policy, "data/local/demos/ant_action_noise_15.npy", N=15, horizon=1000, threshold=5000, render=False)

    # data = load_local_or_remote_file('02-17-sac-mujoco-envs-unnormalized/02-17-sac_mujoco_envs_unnormalized_2020_02_18_01_07_32_id003--s29410/params.pkl')
    # env = data['exploration/env']
    # policy = data['exploration/policy']
    # collect_demos_fixed(env, policy, "data/local/demos/ant_action_noise_25.npy", N=25, horizon=1000, threshold=5000, render=False)

    # data = load_local_or_remote_file(
        # '/home/murtaza/research/rlkit/data/local/03-04-bc-hc-v2/03-04-bc_hc_v2_2020_03_04_17_57_54_id000--s90897/bc.pkl')
    # env = gym.make('Ant-v2')
    # policy = data.cpu()
    # collect_demos_fixed(env, policy, "data/local/demos/ant_off_policy_100.npy", N=100, horizon=1000, threshold=8000,
                        # render=False)
    # data = load_local_or_remote_file(
        # '/home/murtaza/research/rlkit/data/doodads3/03-08-bc-ant-gym-v1/03-08-bc_ant_gym_v1_2020_03_08_19_22_00_id000--s39483/bc.pkl')
    # # env = gym.make('Ant-v2')
    # policy = MakeDeterministic(data.cpu())
    # collect_demos_fixed(env, policy, "data/local/demos/ant_off_policy_10_demos_100.npy", N=100, horizon=1000, threshold=-1,
                        # render=False)

    data = load_local_or_remote_file(
        '/home/murtaza/research/railrl/data/local/03-09-bc-ant-frac-trajs-sweep/03-09-bc_ant_frac_trajs_sweep_2020_03_09_17_58_01_id000--s71624/bc.pkl')
    env = gym.make('Ant-v2')
    policy = data.cpu()
    collect_demos_fixed(env, policy, "data/local/demos/ant_off_policy_10_demos_100.npy", N=100, horizon=1000, threshold=-1,
                        render=False)

    data = load_local_or_remote_file(
        '/home/murtaza/research/railrl/data/local/03-09-bc-ant-frac-trajs-sweep/03-09-bc_ant_frac_trajs_sweep_2020_03_09_17_58_02_id000--s47768/bc.pkl')
    env = gym.make('Ant-v2')
    policy = data.cpu()
    collect_demos_fixed(env, policy, "data/local/demos/ant_off_policy_15_demos_100.npy", N=100, horizon=1000,
                        threshold=-1,
                        render=False)

    data = load_local_or_remote_file(
        '/home/murtaza/research/railrl/data/local/03-09-bc-ant-frac-trajs-sweep/03-09-bc_ant_frac_trajs_sweep_2020_03_09_17_58_03_id000--s66729/bc.pkl')
    env = gym.make('Ant-v2')
    policy = data.cpu()
    collect_demos_fixed(env, policy, "data/local/demos/ant_off_policy_25_demos_100.npy", N=100, horizon=1000,
                        threshold=-1,
                        render=False)

