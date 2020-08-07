import argparse

import joblib

from rlkit.policies.simple import RandomPolicy
from rlkit.state_distance.rollout_util import multitask_rollout

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--pause', action='store_true')
    args = parser.parse_args()
    if args.pause:
        import ipdb; ipdb.set_trace()

    data = joblib.load(args.file)
    env = data['env']
    qf = data['qf']
    policy = data['policy']
    tdm_policy = data['trained_policy']
    random_policy = RandomPolicy(env.action_space)
    vf = data['vf']
    path = multitask_rollout(
        env,
        random_policy,
        init_tau=0,
        max_path_length=100,
        animated=True,
    )
    goal = env.sample_goal_for_rollout()

    import ipdb; ipdb.set_trace()
    agent_infos = path['agent_infos']
