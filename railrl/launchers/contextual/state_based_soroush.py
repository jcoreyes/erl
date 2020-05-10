from functools import partial

import numpy as np

import railrl.samplers.rollout_functions as rf
import railrl.torch.pytorch_util as ptu
from railrl.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
    RemapKeyFn,
)
from railrl.envs.contextual import ContextualEnv, delete_info

from railrl.envs.contextual.goal_conditioned import (
    GoalDictDistributionFromMultitaskEnv,
    ContextualRewardFnFromMultitaskEnv,
    AddImageDistribution,
    GoalConditionedDiagnosticsToContextualDiagnostics,
    IndexIntoAchievedGoal,
)
from railrl.envs.images import Renderer, InsertImageEnv
from railrl.launchers.contextual.util import (
    get_save_video_function,
    get_gym_env,
)
from railrl.launchers.rl_exp_launcher_util import create_exploration_policy
from railrl.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)
from railrl.torch.networks import FlattenMlp
from railrl.torch.sac.policies import MakeDeterministic
from railrl.torch.sac.policies import TanhGaussianPolicy
from railrl.torch.sac.sac import SACTrainer
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


import os.path as osp

from railrl.policies.action_repeat import ActionRepeatPolicy
from railrl.samplers.data_collector import VAEWrappedEnvPathCollector
from railrl.samplers.data_collector.path_collector import GoalConditionedPathCollector
from railrl.visualization.video import VideoSaveFunction
from railrl.torch.her.her import HERTrainer
from railrl.envs.reward_mask_wrapper import DiscreteDistribution, RewardMaskWrapper


def td3_experiment(variant):
    import railrl.samplers.rollout_functions as rf
    import railrl.torch.pytorch_util as ptu
    from railrl.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    from railrl.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )

    from railrl.torch.td3.td3 import TD3 as TD3Trainer
    from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

    from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
    preprocess_rl_variant(variant)

    max_path_length = variant['max_path_length']

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = variant.get('achieved_goal_key', 'latent_achieved_goal')

    context_key = desired_goal_key
    sample_context_from_obs_dict_fn = RemapKeyFn({context_key: observation_key})

    def contextual_env_distrib_and_reward(goal_sampling_mode):
        env = get_envs(variant) #get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)
        env.goal_sampling_mode = goal_sampling_mode
        goal_distribution = GoalDictDistributionFromMultitaskEnv(
            env,
            desired_goal_keys=[desired_goal_key],
        )
        reward_fn = ContextualRewardFnFromMultitaskEnv(
            env=env,
            achieved_goal_from_observation=IndexIntoAchievedGoal(observation_key),
            desired_goal_key=desired_goal_key,
            achieved_goal_key=achieved_goal_key,
            additional_keys=variant['replay_buffer_kwargs'].get('observation_keys', None),
        )
        diag_fn = GoalConditionedDiagnosticsToContextualDiagnostics(
            env.goal_conditioned_diagnostics,
            desired_goal_key=desired_goal_key,
            observation_key=observation_key,
        )
        env = ContextualEnv(
            env,
            context_distribution=goal_distribution,
            reward_fn=reward_fn,
            observation_key=observation_key,
            contextual_diagnostics_fns=[diag_fn],
            update_env_info_fn=delete_info,
        )
        return env, goal_distribution, reward_fn

    expl_env, expl_context_distrib, expl_reward = contextual_env_distrib_and_reward(
        variant.get("exploration_goal_sampling_mode", None)
    )
    eval_env, eval_context_distrib, eval_reward = contextual_env_distrib_and_reward(
        variant.get("evaluation_goal_sampling_mode", None)
    )

    obs_dim = (
            expl_env.observation_space.spaces[observation_key].low.size
            + expl_env.observation_space.spaces[context_key].low.size
    )

    action_dim = expl_env.action_space.low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )


    es = get_exploration_strategy(variant, expl_env)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    def concat_context_to_obs(batch):
        obs = batch['observations']
        next_obs = batch['next_observations']
        context = batch[context_key]
        batch['observations'] = np.concatenate([obs, context], axis=1)
        batch['next_observations'] = np.concatenate([next_obs, context], axis=1)
        return batch

    if 'observation_keys' not in variant['replay_buffer_kwargs']:
        variant['replay_buffer_kwargs']['observation_keys'] = []
    observation_keys = variant['replay_buffer_kwargs']['observation_keys']
    if observation_key not in observation_keys:
        observation_keys.append(observation_key)

    replay_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_keys=[context_key],
        # observation_keys=observation_keys,
        context_distribution=eval_context_distrib,
        sample_context_from_obs_dict_fn=sample_context_from_obs_dict_fn,
        reward_fn=eval_reward,
        post_process_batch_fn=concat_context_to_obs,
        **variant['replay_buffer_kwargs']
    )

    trainer = TD3Trainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['td3_trainer_kwargs']
    )

    eval_path_collector = ContextualPathCollector(
        eval_env,
        policy,
        observation_key=observation_key,
        context_keys_for_policy=[context_key],
    )
    expl_path_collector = ContextualPathCollector(
        expl_env,
        expl_policy,
        observation_key=observation_key,
        context_keys_for_policy=[context_key],
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **variant['algo_kwargs']
    )

    algorithm.to(ptu.device)
    # if not variant.get("do_state_exp", False):
    #     env.vae.to(ptu.device)
    algorithm.train()



def goal_conditioned_sac_experiment(
        max_path_length,
        qf_kwargs,
        sac_trainer_kwargs,
        replay_buffer_kwargs,
        policy_kwargs,
        algo_kwargs,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        achieved_goal_key='state_achieved_goal',
        exploration_policy_kwargs=None,
        evaluation_goal_sampling_mode=None,
        exploration_goal_sampling_mode=None,
        # Video parameters
        save_video=True,
        save_video_kwargs=None,
        renderer_kwargs=None,
):
    if exploration_policy_kwargs is None:
        exploration_policy_kwargs = {}
    if not save_video_kwargs:
        save_video_kwargs = {}
    if not renderer_kwargs:
        renderer_kwargs = {}
    context_key = desired_goal_key
    sample_context_from_obs_dict_fn = RemapKeyFn({context_key: observation_key})

    def contextual_env_distrib_and_reward(
            env_id, env_class, env_kwargs, goal_sampling_mode
    ):
        env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)
        env.goal_sampling_mode = goal_sampling_mode
        goal_distribution = GoalDictDistributionFromMultitaskEnv(
            env,
            desired_goal_keys=[desired_goal_key],
        )
        reward_fn = ContextualRewardFnFromMultitaskEnv(
            env=env,
            achieved_goal_from_observation=IndexIntoAchievedGoal(observation_key),
            desired_goal_key=desired_goal_key,
            achieved_goal_key=achieved_goal_key,
        )
        diag_fn = GoalConditionedDiagnosticsToContextualDiagnostics(
            env.goal_conditioned_diagnostics,
            desired_goal_key=desired_goal_key,
            observation_key=observation_key,
        )
        env = ContextualEnv(
            env,
            context_distribution=goal_distribution,
            reward_fn=reward_fn,
            observation_key=observation_key,
            contextual_diagnostics_fns=[diag_fn],
            update_env_info_fn=delete_info,
        )
        return env, goal_distribution, reward_fn


    expl_env, expl_context_distrib, expl_reward = contextual_env_distrib_and_reward(
        env_id, env_class, env_kwargs, exploration_goal_sampling_mode
    )
    eval_env, eval_context_distrib, eval_reward = contextual_env_distrib_and_reward(
        env_id, env_class, env_kwargs, evaluation_goal_sampling_mode
    )

    obs_dim = (
            expl_env.observation_space.spaces[observation_key].low.size
            + expl_env.observation_space.spaces[context_key].low.size
    )
    action_dim = expl_env.action_space.low.size

    def create_qf():
        return FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            **qf_kwargs
        )
    qf1 = create_qf()
    qf2 = create_qf()
    target_qf1 = create_qf()
    target_qf2 = create_qf()

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **policy_kwargs
    )

    def concat_context_to_obs(batch):
        obs = batch['observations']
        next_obs = batch['next_observations']
        context = batch[context_key]
        batch['observations'] = np.concatenate([obs, context], axis=1)
        batch['next_observations'] = np.concatenate([next_obs, context], axis=1)
        return batch
    replay_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_keys=[context_key],
        observation_keys=[observation_key],
        context_distribution=eval_context_distrib,
        sample_context_from_obs_dict_fn=sample_context_from_obs_dict_fn,
        reward_fn=eval_reward,
        post_process_batch_fn=concat_context_to_obs,
        **replay_buffer_kwargs
    )
    trainer = SACTrainer(
        env=expl_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **sac_trainer_kwargs
    )

    eval_path_collector = ContextualPathCollector(
        eval_env,
        MakeDeterministic(policy),
        observation_key=observation_key,
        context_keys_for_policy=[context_key],
    )
    exploration_policy = create_exploration_policy(
        policy, **exploration_policy_kwargs)
    expl_path_collector = ContextualPathCollector(
        expl_env,
        exploration_policy,
        observation_key=observation_key,
        context_keys_for_policy=[context_key],
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **algo_kwargs
    )
    algorithm.to(ptu.device)

    if save_video:
        rollout_function = partial(
            rf.contextual_rollout,
            max_path_length=max_path_length,
            observation_key=observation_key,
            context_keys_for_policy=[context_key],
        )
        renderer = Renderer(**renderer_kwargs)

        def add_images(env, state_distribution):
            state_env = env.env
            image_goal_distribution = AddImageDistribution(
                env=state_env,
                base_distribution=state_distribution,
                image_goal_key='image_desired_goal',
                renderer=renderer,
            )
            img_env = InsertImageEnv(state_env, renderer=renderer)
            return ContextualEnv(
                img_env,
                context_distribution=image_goal_distribution,
                reward_fn=eval_reward,
                observation_key=observation_key,
                update_env_info_fn=delete_info,
            )
        img_eval_env = add_images(eval_env, eval_context_distrib)
        img_expl_env = add_images(expl_env, expl_context_distrib)
        eval_video_func = get_save_video_function(
            rollout_function,
            img_eval_env,
            MakeDeterministic(policy),
            tag="eval",
            imsize=renderer.image_shape[0],
            image_format='CWH',
            **save_video_kwargs
        )
        expl_video_func = get_save_video_function(
            rollout_function,
            img_expl_env,
            exploration_policy,
            tag="train",
            imsize=renderer.image_shape[0],
            image_format='CWH',
            **save_video_kwargs
        )

        algorithm.post_train_funcs.append(eval_video_func)
        algorithm.post_train_funcs.append(expl_video_func)

    algorithm.train()


def process_args(variant):
    if variant.get("debug", False):
        pass

def preprocess_rl_variant(variant):
    if variant.get("do_state_exp", False):
        variant['observation_key'] = 'state_observation'
        variant['desired_goal_key'] = 'state_desired_goal'
        variant['achieved_goal_key'] = 'state_achieved_goal'


def get_presampled_goals_path(path=''):
    """
    :param path: if relative, this will rpe
    :param config: One of a few options:
        - string: the path to the
        - tuple of two strings: the first string specifies the 'mode' and the
            second string specifies extra parameters to that mode
        - None: return None
    :return:  Path to the presampled goals, or None.
    """
    if not path:
        return path
    if path[0] == '/':
        return path
    else:
        import multiworld.envs.mujoco as mwmj
        return osp.join(osp.dirname(mwmj.__file__), path)

def get_envs(variant):
    from multiworld.core.image_env import ImageEnv
    from railrl.envs.vae_wrappers import VAEWrappedEnv, ConditionalVAEWrappedEnv
    from railrl.misc.asset_loader import load_local_or_remote_file
    from railrl.torch.vae.conditional_conv_vae import CVAE, CDVAE, ACE, CADVAE, DeltaCVAE

    render = variant.get('render', False)
    vae_path = variant.get("vae_path", None)
    reward_params = variant.get("reward_params", dict())
    init_camera = variant.get("init_camera", None)
    do_state_exp = variant.get("do_state_exp", False)
    presample_goals = variant.get('presample_goals', False)
    presample_image_goals_only = variant.get('presample_image_goals_only', False)
    presampled_goals_path = get_presampled_goals_path(
        variant.get('presampled_goals_path', None))
    vae = load_local_or_remote_file(vae_path) if type(vae_path) is str else vae_path
    if 'env_id' in variant:
        import gym
        import multiworld
        multiworld.register_all_envs()
        env = gym.make(variant['env_id'])
    else:
        env = variant["env_class"](**variant['env_kwargs'])
    if not do_state_exp:
        if isinstance(env, ImageEnv):
            image_env = env
        else:
            image_env = ImageEnv(
                env,
                variant.get('imsize'),
                init_camera=init_camera,
                transpose=True,
                normalize=True,
            )
        if presample_goals:
            """
            This will fail for online-parallel as presampled_goals will not be
            serialized. Also don't use this for online-vae.
            """
            if presampled_goals_path is None:
                image_env.non_presampled_goal_img_is_garbage = True
                vae_env = VAEWrappedEnv(
                    image_env,
                    vae,
                    imsize=image_env.imsize,
                    decode_goals=render,
                    render_goals=render,
                    render_rollouts=render,
                    reward_params=reward_params,
                    **variant.get('vae_wrapped_env_kwargs', {})
                )
                presampled_goals = variant['generate_goal_dataset_fctn'](
                    env=vae_env,
                    env_id=variant.get('env_id', None),
                    **variant['goal_generation_kwargs']
                )
                del vae_env
            else:
                presampled_goals = load_local_or_remote_file(
                    presampled_goals_path
                ).item()
            del image_env
            image_env = ImageEnv(
                env,
                variant.get('imsize'),
                init_camera=init_camera,
                transpose=True,
                normalize=True,
                presampled_goals=presampled_goals,
                **variant.get('image_env_kwargs', {})
            )
            vae_env = VAEWrappedEnv(
                image_env,
                vae,
                imsize=image_env.imsize,
                decode_goals=render,
                render_goals=render,
                render_rollouts=render,
                reward_params=reward_params,
                presampled_goals = presampled_goals,
                **variant.get('vae_wrapped_env_kwargs', {})
            )
            print("Presampling all goals only")
        else:
            if type(vae) is CVAE or type(vae) is CDVAE or type(vae) is ACE or type(vae) is CADVAE or type(vae) is DeltaCVAE:
                vae_env = ConditionalVAEWrappedEnv(
                    image_env,
                    vae,
                    imsize=image_env.imsize,
                    decode_goals=render,
                    render_goals=render,
                    render_rollouts=render,
                    reward_params=reward_params,
                    **variant.get('vae_wrapped_env_kwargs', {})
                )
            else:
                vae_env = VAEWrappedEnv(
                    image_env,
                    vae,
                    imsize=image_env.imsize,
                    decode_goals=render,
                    render_goals=render,
                    render_rollouts=render,
                    reward_params=reward_params,
                    **variant.get('vae_wrapped_env_kwargs', {})
                )
            if presample_image_goals_only:
                presampled_goals = variant['generate_goal_dataset_fctn'](
                    image_env=vae_env.wrapped_env,
                    **variant['goal_generation_kwargs']
                )
                image_env.set_presampled_goals(presampled_goals)
                print("Presampling image goals only")
            else:
                print("Not using presampled goals")

        env = vae_env

    return env


def get_exploration_strategy(variant, env):
    from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
    from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
    from railrl.exploration_strategies.gaussian_and_epislon import \
        GaussianAndEpislonStrategy
    from railrl.exploration_strategies.ou_strategy import OUStrategy
    from railrl.exploration_strategies.noop import NoopStrategy

    exploration_type = variant['exploration_type']
    # exploration_noise = variant.get('exploration_noise', 0.1)
    es_kwargs = variant.get('es_kwargs', {})
    if exploration_type == 'ou':
        es = OUStrategy(
            action_space=env.action_space,
            # max_sigma=exploration_noise,
            # min_sigma=exploration_noise,  # Constant sigma
            **es_kwargs
        )
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            # max_sigma=exploration_noise,
            # min_sigma=exploration_noise,  # Constant sigma
            **es_kwargs
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            # prob_random_action=exploration_noise,
            **es_kwargs
        )
    elif exploration_type == 'gaussian_and_epsilon':
        es = GaussianAndEpislonStrategy(
            action_space=env.action_space,
            # max_sigma=exploration_noise,
            # min_sigma=exploration_noise,  # Constant sigma
            # epsilon=exploration_noise,
            **es_kwargs
        )
    elif exploration_type == 'noop':
        es = NoopStrategy(
            action_space=env.action_space
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    return es
