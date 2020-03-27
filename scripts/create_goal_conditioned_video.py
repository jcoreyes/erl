import argparse
import pickle
import uuid

import railrl.samplers.rollout_functions as rf
from railrl.visualization.video import dump_video
import railrl.torch.pytorch_util as ptu
from railrl.envs.remote import RemoteRolloutEnv
from railrl.envs.vae_wrappers import VAEWrappedEnv


def make_video(args):
    if args.pause:
        import ipdb; ipdb.set_trace()
    data = pickle.load(open(args.file, "rb")) # joblib.load(args.file)
    if 'policy' in data:
        policy = data['policy']
    elif 'evaluation/policy' in data:
        policy = data['evaluation/policy']
    else:
        raise AttributeError

    if 'env' in data:
        env = data['env']
    elif 'evaluation/env' in data:
        env = data['evaluation/env']
    else:
        raise AttributeError

    if isinstance(env, RemoteRolloutEnv):
        env = env._wrapped_env
    print("Policy loaded")
    if args.gpu:
        ptu.set_gpu_mode(True)
        policy.to(ptu.device)
    else:
        ptu.set_gpu_mode(False)
        policy.to(ptu.device)
    if isinstance(env, VAEWrappedEnv):
        env.mode(args.mode)

    max_path_length = 100
    observation_key = 'latent_observation'
    desired_goal_key = 'latent_desired_goal'
    rollout_function = rf.create_rollout_function(
        rf.multitask_rollout,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    env.mode(env._mode_map['video_env'])
    random_id = str(uuid.uuid4()).split('-')[0]
    dump_video(
        env,
        policy,
        'rollouts_{}.mp4'.format(random_id),
        rollout_function,
        rows=3,
        columns=6,
        pad_length=0,
        pad_color=255,
        do_timer=True,
        horizon=max_path_length,
        dirname_to_save_images=None,
        subdirname="rollouts",
        imsize=48,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--mode', default='video_env', type=str,
                        help='env mode')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--multitaskpause', action='store_true')
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    make_video(args)
