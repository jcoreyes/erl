from railrl.demos.collect_demo import collect_demos
import pickle
if __name__ == '__main__':
    data_file = '/home/murtaza/research/railrl/data/doodads3/11-06-door-reset-free-state-td3-confirm/11-06-door_reset_free_state_td3_confirm_2019_11_06_17_34_12_id000--s50047/params.pkl'
    data = pickle.load(open(data_file, 'rb'))
    env = data['evaluation/env']
    policy = data['trainer/trained_policy']
    # presampled_goals_path = osp.join(
                # osp.dirname(mwmj.__file__),
                # "goals",
                # "door_goals.npy",
            # )
    # presampled_goals = load_local_or_remote_file(
                    # presampled_goals_path
                # ).item()
    # image_env = ImageEnv(
                # env,
                # 48,
                # init_camera=sawyer_door_env_camera_v0,
                # transpose=True,
                # normalize=True,
                # presampled_goals=presampled_goals,
    # )
    collect_demos(env, policy, "data/local/demos/door_demos_1000.npy", N=1000, horizon=100, threshold=.1, add_action_noise=False)
