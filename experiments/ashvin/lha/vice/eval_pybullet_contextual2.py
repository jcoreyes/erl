import argparse
import pickle

from rlkit.core import logger
from rlkit.envs.remote import RemoteRolloutEnv
from rlkit.envs.vae_wrappers import VAEWrappedEnv
from rlkit.samplers.rollout_functions import (
    multitask_rollout,
    contextual_rollout,
)
from rlkit.torch.core import PyTorchModule
import rlkit.torch.pytorch_util as ptu
from rlkit.misc.asset_loader import local_path_from_s3_or_local_path
import torch
from rlkit.launchers.arglauncher import run_variants
import rlkit.misc.hyperparameter as hyp
from multiworld.envs.pygame import PickAndPlaceEnv
from rlkit.launchers.experiments.goal_distribution.irl_launcher import \
    representation_learning_with_goal_distribution_launcher

from rlkit.launchers.launcher_util import run_experiment
# from rlkit.torch.sets.launcher import test_offline_set_vae
# from rlkit.launchers.masking_launcher import default_masked_reward_fn
from rlkit.envs.contextual.mask_conditioned import default_masked_reward_fn
from rlkit.launchers.exp_launcher import rl_context_experiment

from roboverse.envs.goal_conditioned.sawyer_lift_gc import SawyerLiftEnvGC

from roboverse.envs.goal_conditioned.sawyer_lift_gc import SawyerLiftEnvGC
from rlkit.launchers.contextual.rig.rig_launcher import get_gym_env

def simulate_policy(args):
    if args.pause:
        import ipdb; ipdb.set_trace()
    # data = pickle.load(open(args.file, "rb")) # joblib.load(args.file)
    path = local_path_from_s3_or_local_path(args.file)
    data = torch.load(path)
    if 'policy' in data:
        policy = data['policy']
    elif 'evaluation/policy' in data:
        policy = data['evaluation/policy']

    env_kwargs = {
        'action_scale': .06,
        'action_repeat': 10,
        'timestep': 1./120,
        'solver_iterations': 500,
        'max_force': 1000,

        'gui': False,
        'pos_init': [.75, -.3, 0],
        'pos_high': [.75, .4, .3],
        'pos_low': [.75, -.4, -.36],
        'reset_obj_in_hand_rate': 0.0,
        'bowl_bounds': [-0.40, 0.40],

        'use_rotated_gripper': True,
        'use_wide_gripper': True,
        'soft_clip': True,
        'obj_urdf': 'spam',
        'max_joint_velocity': None,

        'hand_reward': True,
        'gripper_reward': True,
        'bowl_reward': True,

        'goal_sampling_mode': 'ground',
        'random_init_bowl_pos': True,
        'bowl_type': 'heavy',
        'num_obj': 4,
        'obj_success_threshold': 0.10,

        'objs_to_reset_outside_bowl': [0],
    }
    env = get_gym_env("", env_class=SawyerLiftEnvGC, env_kwargs=env_kwargs)

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
    if args.enable_render or hasattr(env, 'enable_render'):
        # some environments need to be reconfigured for visualization
        env.enable_render()
    if args.multitaskpause:
        env.pause_on_goal = True
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    paths = []
    def check(net):
        for name, param in net.named_parameters():
            if torch.isnan(param).any():
                print(name)
    qf = data['trainer/qf1']

    while True:
        paths.append(contextual_rollout(
            env,
            policy,
            max_path_length=args.H,
            render=not args.hide,
            observation_key=data.get('evaluation/observation_key', 'observation'),
            context_keys_for_policy=data.get('evaluation/context_keys_for_policy', ['context']),
            # context_keys_for_policy=['state_desired_goal'],
        ))
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
        if hasattr(env, "get_diagnostics"):
            for k, v in env.get_diagnostics(paths).items():
                logger.record_tabular(k, v)
        logger.dump_tabular()

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

    simulate_policy(args)
