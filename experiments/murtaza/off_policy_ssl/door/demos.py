from railrl.demos.collect_demo import collect_demos
import pickle
if __name__ == '__main__':
    data_file = '/home/murtaza/research/railrl/data/local/10-16-dev-experiments-murtaza-off-policy-ssl-door-state/10-16-dev-experiments-murtaza-off-policy-ssl-door-state_2019_10_16_17_24_54_id000--s53075/params.pkl'
    f = open(data_file, 'rb')
    data = pickle.load(f)
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
    collect_demos(env, policy, "door_demos_100.npy", 100)
