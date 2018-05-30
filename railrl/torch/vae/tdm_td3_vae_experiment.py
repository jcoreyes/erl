from railrl.envs.multitask.multitask_env import MultitaskEnvToSilentMultitaskEnv
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.state_distance.tdm_networks import TdmQf, TdmPolicy
from railrl.state_distance.tdm_td3 import TdmTd3
import railrl.torch.pytorch_util as ptu
from railrl.misc.ml_util import IntPiecewiseLinearSchedule

from railrl.envs.wrappers import ImageMujocoEnv
from railrl.envs.vae_wrappers import VAEWrappedEnv
from railrl.data_management.her_replay_buffer import RelabelingReplayBuffer
import pickle

from railrl.core import logger
import os.path as osp

from railrl.torch.vae.sim_vae_policy import create_multitask_rollout_function


def tdm_td3_vae_experiment(variant):
    env = variant["env"](**variant['env_kwargs'])

    do_state_based_exp = variant.get("do_state_based_exp", False)
    render = variant["render"]

    if not do_state_based_exp:
        rdim = variant["rdim"]
        use_env_goals = variant["use_env_goals"]
        vae_path = variant["vae_paths"][str(rdim)]
        wrap_mujoco_env = variant.get("wrap_mujoco_env", False)
        reward_params = variant.get("reward_params", dict())

        init_camera = variant.get("init_camera", None)
        if init_camera is None:
            camera_name = "topview"
        else:
            camera_name = None

        if wrap_mujoco_env:
            env = ImageMujocoEnv(
                env,
                84,
                init_camera=init_camera,
                camera_name=camera_name,
                transpose=True,
                normalize=True,
            )

        use_vae_goals = not use_env_goals
        env = VAEWrappedEnv(env, vae_path, use_vae_obs=True,
            use_vae_reward=True, use_vae_goals=use_vae_goals,
            decode_goals=render,
            render_goals=render, render_rollouts=render,
            reward_params=reward_params,
            **variant.get('vae_wrapped_env_kwargs', {})
        )

    if do_state_based_exp:
        env = MultitaskEnvToSilentMultitaskEnv(env)
    if do_state_based_exp and render:
        env.pause_on_goal = True

    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    exploration_type = variant['exploration_type']
    exploration_noise = variant.get('exploration_noise', 0.1)
    if exploration_type == 'ou':
        es = OUStrategy(
            action_space=env.action_space,
            **variant['es_kwargs']
        )
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=exploration_noise,
            min_sigma=exploration_noise,  # Constant sigma
            **variant['es_kwargs']
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=exploration_noise,
            **variant['es_kwargs']
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    qf1 = TdmQf(
        env=env,
        vectorized=True,
        **variant['qf_kwargs']
    )
    qf2 = TdmQf(
        env=env,
        vectorized=True,
        **variant['qf_kwargs']
    )
    policy = TdmPolicy(
        env=env,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    if do_state_based_exp:
        testing_env = env
        training_env = env
        relabeling_env = pickle.loads(pickle.dumps(env))
    else:
        training_mode = variant.get("training_mode", "train")
        testing_mode = variant.get("testing_mode", "test")

        testing_env = pickle.loads(pickle.dumps(env))
        testing_env.mode(testing_mode)
        training_env = pickle.loads(pickle.dumps(env))
        training_env.mode(training_mode)

        relabeling_env = pickle.loads(pickle.dumps(env))
        relabeling_env.mode(training_mode)
        # save time by not resetting relabel env
        relabeling_env.reset_on_sample_goal_for_rollout = False
        relabeling_env.disable_render()

        video_vae_env = pickle.loads(pickle.dumps(env))
        video_vae_env.mode("video_vae")
        video_goal_env = pickle.loads(pickle.dumps(env))
        video_goal_env.mode("video_env")

    replay_buffer = RelabelingReplayBuffer(
        env=relabeling_env,
        **variant['replay_kwargs']
    )
    qf_criterion = variant['qf_criterion_class']()
    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['td3_kwargs']['qf_criterion'] = qf_criterion
    algo_kwargs['tdm_kwargs']['env_samples_goal_on_reset'] = True
    algo_kwargs['td3_kwargs']['training_env'] = training_env
    tau_schedule = IntPiecewiseLinearSchedule(**variant['tau_schedule_kwargs'])
    algo_kwargs['tdm_kwargs']['epoch_max_tau_schedule'] = tau_schedule
    algorithm = TdmTd3(
        testing_env,
        qf1=qf1,
        qf2=qf2,
        replay_buffer=replay_buffer,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )

    # print("use_gpu", variant["use_gpu"], bool(variant["use_gpu"]))
    # if variant["use_gpu"]: # change this to standardized format
    if ptu.gpu_enabled():
        print("using GPU")
        # gpu_id = variant["gpu_id"]
        # ptu.set_gpu_mode(True)
        # ptu.set_device(gpu_id)
        algorithm.cuda()
        if not do_state_based_exp:
            for e in [testing_env, training_env, video_vae_env, video_goal_env]:
                e.vae.cuda()

    save_video = variant.get("save_video", True)
    max_tau = algorithm.max_tau
    rollout_function = create_multitask_rollout_function(max_tau)
    if not do_state_based_exp and save_video:
        from railrl.torch.vae.sim_vae_policy import dump_video
        logdir = logger.get_snapshot_dir()
        filename = osp.join(logdir, 'video_0_env.mp4')
        dump_video(video_goal_env, policy, filename,
                   rollout_function=rollout_function)
        filename = osp.join(logdir, 'video_0_vae.mp4')
        dump_video(video_vae_env, policy, filename,
                   rollout_function=rollout_function)

    algorithm.train()

    if not do_state_based_exp and save_video:
        filename = osp.join(logdir, 'video_final_env.mp4')
        dump_video(video_goal_env, policy, filename,
                   rollout_function=rollout_function)
        filename = osp.join(logdir, 'video_final_vae.mp4')
        dump_video(video_vae_env, policy, filename,
                   rollout_function=rollout_function)
