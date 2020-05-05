from functools import partial

import numpy as np
from multiworld.core.multitask_env import MultitaskEnv
from torch import nn

import railrl.samplers.rollout_functions as rf
import railrl.torch.pytorch_util as ptu
from railrl.core.distribution import DictDistribution
from railrl.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
    SampleContextFromObsDictFn,
    RemapKeyFn,
)
from railrl.envs.contextual import (
    ContextualEnv, ContextualRewardFn,
    delete_info,
)
from railrl.envs.contextual.goal_conditioned import (
    AddImageDistribution,
    GoalConditionedDiagnosticsToContextualDiagnostics,
)
from railrl.envs.images import Renderer, InsertImageEnv
from railrl.launchers.contextual.util import (
    get_save_video_function,
    get_gym_env,
)
from railrl.launchers.experiments.disentanglement.debug import (
    JointTrainer,
    DebugTrainer,
    DebugRenderer,
    InsertDebugImagesEnv,
    create_visualize_representation,
)
from railrl.policies.action_repeat import ActionRepeatPolicy
from railrl.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)
from railrl.torch.disentanglement.encoder_wrapped_env import (
    Encoder,
    EncoderFromNetwork,
)
from railrl.torch.disentanglement.networks import (
    DisentangledMlpQf,
    EncodeObsAndGoal,
)
from railrl.torch.modules import Concat
from railrl.torch.networks import FlattenMlp, Flatten
from railrl.torch.networks.mlp import MultiHeadedMlp
from railrl.torch.networks.stochastic.distribution_generator import TanhGaussian
from railrl.torch.sac.policies import (
    MakeDeterministic,
    PolicyFromDistributionGenerator,
)
from railrl.torch.sac.sac import SACTrainer
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


def create_exploration_policy(policy, exploration_version='identity', **kwargs):
    if exploration_version == 'identity':
        return policy
    elif exploration_version == 'occasionally_repeat':
        return ActionRepeatPolicy(policy, **kwargs)
    else:
        raise ValueError(exploration_version)


class EncoderRewardFnFromMultitaskEnv(ContextualRewardFn):
    def __init__(
            self,
            encoder: Encoder,
            next_state_encoder_input_key,
            context_key,
            vectorize=False,
            reward_scale=1.,
    ):
        self._encoder = encoder
        self._vectorize = vectorize
        self._next_state_key = next_state_encoder_input_key
        self._context_key = context_key
        self._reward_scale = reward_scale

    def __call__(self, states, actions, next_states, contexts):
        if self._context_key not in contexts:
            import ipdb
            ipdb.set_trace()
        z_s = self._encoder.encode(next_states[self._next_state_key])
        z_g = contexts[self._context_key]
        if self._vectorize:
            rewards = - np.abs(z_s - z_g)
        else:
            rewards = - np.linalg.norm(z_s - z_g, axis=1, ord=1)
        return self._reward_scale * rewards


class EncodedGoalDictDistributionFromMultitaskEnv(DictDistribution):
    def __init__(
            self,
            env: MultitaskEnv,
            encoder: Encoder,
            encoder_input_key,
            encoder_output_key,
            keys_to_keep=('desired_goal',),
    ):
        self._env = env
        self._goal_keys_to_keep = keys_to_keep
        self._encoder = encoder
        self._encoder_input_key = encoder_input_key
        self._encoder_output_key = encoder_output_key
        env_spaces = self._env.observation_space.spaces
        self._spaces = {
            k: env_spaces[k]
            for k in self._goal_keys_to_keep
        }
        self._spaces[encoder_output_key] = encoder.space

    def sample(self, batch_size: int):
        goals = {
            k: self._env.sample_goals(batch_size)[k]
            for k in self._goal_keys_to_keep
        }
        goals[self._encoder_output_key] = self._encoder.encode(
            goals[self._encoder_input_key]
        )
        return goals

    @property
    def spaces(self):
        return self._spaces


class ReEncoderAchievedStateFn(SampleContextFromObsDictFn):
    def __init__(self, encoder, encoder_input_key, encoder_output_key):
        self._encoder = encoder
        self._encoder_input_key = encoder_input_key
        self._encoder_output_key = encoder_output_key
        self._keys_to_keep = [self._encoder_input_key]

    def __call__(self, obs: dict):
        context = {k: obs[k] for k in self._keys_to_keep if k in obs}
        context[self._encoder_output_key] = self._encoder.encode(
            obs[self._encoder_input_key])
        return context


def encoder_goal_conditioned_sac_experiment(
        max_path_length,
        qf_kwargs,
        sac_trainer_kwargs,
        replay_buffer_kwargs,
        algo_kwargs,
        policy_kwargs,
        # Encoder parameters
        disentangled_qf_kwargs,
        encoder_kwargs=None,
        vectorized_reward=None,
        use_target_encoder_for_reward=False,
        encoder_reward_scale=1.,
        # Policy params
        policy_using_encoder_settings=None,
        # Env settings
        env_id=None,
        env_class=None,
        env_kwargs=None,
        contextual_env_kwargs=None,
        exploration_policy_kwargs=None,
        evaluation_goal_sampling_mode=None,
        exploration_goal_sampling_mode=None,
        # Video parameters
        save_video=True,
        save_video_kwargs=None,
        renderer_kwargs=None,
        debug_renderer_kwargs=None,
        debug_visualization_kwargs=None,
):
    if policy_using_encoder_settings is None:
        policy_using_encoder_settings = {}
    if debug_visualization_kwargs is None:
        debug_visualization_kwargs = {}
    if exploration_policy_kwargs is None:
        exploration_policy_kwargs = {}
    if contextual_env_kwargs is None:
        contextual_env_kwargs = {}
    if encoder_kwargs is None:
        encoder_kwargs = {}
    if save_video_kwargs is None:
        save_video_kwargs = {}
    if renderer_kwargs is None:
        renderer_kwargs = {}
    if debug_renderer_kwargs is None:
        debug_renderer_kwargs = {}

    state_observation_key = 'state_observation'
    latent_observation_key = 'latent_observation'
    latent_desired_goal_key = 'latent_desired_goal'
    state_desired_goal_key = 'state_desired_goal'

    def setup_env(state_env, encoder, reward_fn):
        goal_distribution = EncodedGoalDictDistributionFromMultitaskEnv(
            state_env,
            encoder=encoder,
            keys_to_keep=[state_desired_goal_key],
            encoder_input_key=state_desired_goal_key,
            encoder_output_key=latent_desired_goal_key,
        )
        state_diag_fn = GoalConditionedDiagnosticsToContextualDiagnostics(
            state_env.goal_conditioned_diagnostics,
            desired_goal_key=state_desired_goal_key,
            observation_key=state_observation_key,
        )
        env = ContextualEnv(
            state_env,
            context_distribution=goal_distribution,
            reward_fn=reward_fn,
            contextual_diagnostics_fns=[state_diag_fn],
            update_env_info_fn=delete_info,
            **contextual_env_kwargs,
        )
        return env, goal_distribution

    expl_env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)
    expl_env.goal_sampling_mode = exploration_goal_sampling_mode
    eval_env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)
    eval_env.goal_sampling_mode = evaluation_goal_sampling_mode

    state_dim = (
        expl_env.observation_space.spaces['state_observation'].low.size
    )
    encoder_net = FlattenMlp(input_size=state_dim, **encoder_kwargs)
    encoder_output_dim = encoder_net.output_size
    target_encoder_net = FlattenMlp(input_size=state_dim, **encoder_kwargs)
    encoder = EncoderFromNetwork(encoder_net)
    if use_target_encoder_for_reward:
        target_encoder = EncoderFromNetwork(target_encoder_net)
        reward_fn = EncoderRewardFnFromMultitaskEnv(
            encoder=target_encoder,
            next_state_encoder_input_key=state_observation_key,
            context_key=latent_desired_goal_key,
            reward_scale=encoder_reward_scale,
            vectorize=vectorized_reward,
        )
    else:
        reward_fn = EncoderRewardFnFromMultitaskEnv(
            encoder=encoder,
            next_state_encoder_input_key=state_observation_key,
            context_key=latent_desired_goal_key,
            vectorize=vectorized_reward,
        )
    expl_env, expl_context_distrib = setup_env(expl_env, encoder, reward_fn)
    eval_env, eval_context_distrib = setup_env(eval_env, encoder, reward_fn)

    action_dim = expl_env.action_space.low.size

    def make_qf(enc):
        return DisentangledMlpQf(
            encoder=enc,
            preprocess_obs_dim=state_dim,
            action_dim=action_dim,
            qf_kwargs=qf_kwargs,
            vectorized=vectorized_reward,
            **disentangled_qf_kwargs
        )

    qf1 = make_qf(encoder_net)
    qf2 = make_qf(encoder_net)
    target_qf1 = make_qf(target_encoder_net)
    target_qf2 = make_qf(target_encoder_net)

    context_key_for_rl = state_desired_goal_key
    observation_key_for_rl = state_observation_key
    state_dim = (
        expl_env.observation_space.spaces[context_key_for_rl].low.size)
    policy_encoder_net = EncodeObsAndGoal(
        encoder_net,
        state_dim,
        **policy_using_encoder_settings
    )
    obs_processor = nn.Sequential(
        policy_encoder_net,
        Concat(),
        MultiHeadedMlp(
            input_size=policy_encoder_net.output_size,
            output_sizes=[action_dim, action_dim],
            **policy_kwargs
        )
    )
    policy = PolicyFromDistributionGenerator(
        TanhGaussian(obs_processor)
    )

    def concat_context_to_obs(batch):
        obs = batch['observations']
        next_obs = batch['next_observations']
        context = batch[context_key_for_rl]
        batch['observations'] = np.concatenate([obs, context], axis=1)
        batch['next_observations'] = np.concatenate([next_obs, context], axis=1)
        return batch

    sample_context_from_observation = compose(
        # first map 'state_observation' --> 'state_desired_goal'
        RemapKeyFn({
            state_desired_goal_key: state_observation_key
        }),
        # them map `state_desired_goal` -> `latent_desired_goal`
        ReEncoderAchievedStateFn(
            encoder=encoder,
            encoder_input_key=state_desired_goal_key,
            encoder_output_key=latent_desired_goal_key,
        ),
    )

    replay_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_keys=[state_desired_goal_key, latent_desired_goal_key],
        context_distribution=eval_context_distrib,
        sample_context_from_obs_dict_fn=sample_context_from_observation,
        observation_keys=[state_observation_key],
        observation_key=observation_key_for_rl,
        reward_fn=reward_fn,
        post_process_batch_fn=concat_context_to_obs,
        reward_dim=encoder_output_dim if vectorized_reward else 1,
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
    debug_trainer = DebugTrainer(
        observation_space=expl_env.observation_space.spaces[
            state_observation_key
        ],
        encoder=encoder_net,
        encoder_output_dim=encoder_output_dim,
    )
    trainer = JointTrainer([trainer, debug_trainer])

    eval_path_collector = ContextualPathCollector(
        eval_env,
        MakeDeterministic(policy),
        observation_key=observation_key_for_rl,
        context_keys_for_policy=[context_key_for_rl],
    )
    exploration_policy = create_exploration_policy(
        policy, **exploration_policy_kwargs)
    expl_path_collector = ContextualPathCollector(
        expl_env,
        exploration_policy,
        observation_key=observation_key_for_rl,
        context_keys_for_policy=[context_key_for_rl],
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

    renderer = Renderer(**renderer_kwargs)
    if save_video:
        rollout_function = partial(
            rf.contextual_rollout,
            max_path_length=max_path_length,
            observation_key=observation_key_for_rl,
            context_keys_for_policy=[context_key_for_rl],
        )

        obj1_sweep_renderers = {
            'sweep_obj1_%d' % i: DebugRenderer(
                encoder, i, **debug_renderer_kwargs)
            for i in range(encoder_output_dim)
        }
        obj0_sweep_renderers = {
            'sweep_obj0_%d' % i: DebugRenderer(
                encoder, i, **debug_renderer_kwargs)
            for i in range(encoder_output_dim)

        }

        debugger_one = DebugRenderer(encoder, 0, **debug_renderer_kwargs)

        def create_shared_data_creator(obj_index):
            def compute_shared_data(raw_obs, env):
                state = raw_obs['state_observation']
                obs = state[:2]
                goal = state[2:]
                low = env.env.observation_space['state_observation'].low.min()
                high = env.env.observation_space['state_observation'].high.max()
                y = np.linspace(low, high, num=debugger_one.image_shape[0])
                x = np.linspace(low, high, num=debugger_one.image_shape[1])
                cross = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
                if obj_index == 0:
                    new_states = np.concatenate(
                        [
                            np.repeat(obs[None, :], cross.shape[0], axis=0),
                            cross,
                        ],
                        axis=1,
                    )
                elif obj_index == 1:
                    new_states = np.concatenate(
                        [
                            cross,
                            np.repeat(goal[None, :], cross.shape[0], axis=0),
                        ],
                        axis=1,
                    )
                else:
                    raise ValueError(obj_index)
                return encoder.encode(new_states)
            return compute_shared_data
        obj0_sweeper = create_shared_data_creator(0)
        obj1_sweeper = create_shared_data_creator(1)

        def add_images(env, base_distribution):
            state_env = env.env
            image_goal_distribution = AddImageDistribution(
                env=state_env,
                base_distribution=base_distribution,
                image_goal_key='image_desired_goal',
                renderer=renderer,
            )
            img_env = InsertImageEnv(state_env, renderer=renderer)
            img_env = InsertDebugImagesEnv(
                img_env,
                obj1_sweep_renderers,
                compute_shared_data=obj1_sweeper,
            )
            img_env = InsertDebugImagesEnv(
                img_env,
                obj0_sweep_renderers,
                compute_shared_data=obj0_sweeper,
            )
            return ContextualEnv(
                img_env,
                context_distribution=image_goal_distribution,
                reward_fn=reward_fn,
                observation_key=observation_key_for_rl,
                update_env_info_fn=delete_info,
            )

        img_eval_env = add_images(eval_env, eval_context_distrib)
        img_expl_env = add_images(expl_env, expl_context_distrib)

        def get_extra_imgs(
                path,
                index_in_path,
                env,
        ):
            return [
                path['full_observations'][index_in_path][key]
                for key in obj1_sweep_renderers
            ] + [
                path['full_observations'][index_in_path][key]
                for key in obj0_sweep_renderers
            ]
        eval_video_func = get_save_video_function(
            rollout_function,
            img_eval_env,
            MakeDeterministic(policy),
            tag="eval",
            imsize=renderer.image_shape[0],
            image_format='CWH',
            get_extra_imgs=get_extra_imgs,
            **save_video_kwargs
        )
        expl_video_func = get_save_video_function(
            rollout_function,
            img_expl_env,
            exploration_policy,
            tag="train",
            imsize=renderer.image_shape[0],
            image_format='CWH',
            get_extra_imgs=get_extra_imgs,
            **save_video_kwargs
        )


        algorithm.post_train_funcs.append(eval_video_func)
        algorithm.post_train_funcs.append(expl_video_func)
    visualize_representation = create_visualize_representation(
        encoder, True, eval_env, renderer,
        **debug_visualization_kwargs
    )
    algorithm.post_train_funcs.append(visualize_representation)
    visualize_representation = create_visualize_representation(
        encoder, False, eval_env, renderer,
        **debug_visualization_kwargs
    )
    algorithm.post_train_funcs.append(visualize_representation)

    algorithm.train()


def compose(*functions):
    def composite_function(x):
        for f in functions:
            x = f(x)
        return x

    return composite_function
