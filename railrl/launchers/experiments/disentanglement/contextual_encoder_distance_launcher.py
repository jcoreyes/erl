from functools import partial

import numpy as np
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
    PresampledDistribution,
    GoalDictDistributionFromMultitaskEnv,
    ContextualRewardFnFromMultitaskEnv,
    IndexIntoAchievedGoal,
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
from railrl.torch.networks import FlattenMlp, BasicCNN, Flatten, ConcatTuple
from railrl.torch.networks.mlp import MultiHeadedMlp, Mlp
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


class EncodedGoalDictDistribution(DictDistribution):
    def __init__(
            self,
            dict_distribution: DictDistribution,
            encoder: Encoder,
            encoder_input_key,
            encoder_output_key,
            keys_to_keep=('desired_goal',),
    ):
        self._dict_distribution = dict_distribution
        self._goal_keys_to_keep = keys_to_keep
        self._encoder = encoder
        self._encoder_input_key = encoder_input_key
        self._encoder_output_key = encoder_output_key
        self._spaces = dict_distribution.spaces
        self._spaces[encoder_output_key] = encoder.space

    def sample(self, batch_size: int):
        sampled_goals = self._dict_distribution.sample(batch_size)
        goals = {k: sampled_goals[k] for k in self._goal_keys_to_keep}
        goals[self._encoder_output_key] = self._encoder.encode(
            goals[self._encoder_input_key]
        )
        return goals

    @property
    def spaces(self):
        return self._spaces


class ReEncoderAchievedStateFn(SampleContextFromObsDictFn):
    def __init__(
            self, encoder, encoder_input_key, encoder_output_key,
            keys_to_keep=None,
    ):
        self._encoder = encoder
        self._encoder_input_key = encoder_input_key
        self._encoder_output_key = encoder_output_key
        self._keys_to_keep = keys_to_keep or [self._encoder_input_key]

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
        encoder_cnn_kwargs=None,
        vectorized_reward=None,
        encoder_reward_scale=1.,
        skip_encoder_mlp=False,
        # Policy params
        policy_using_encoder_settings=None,
        use_separate_encoder_for_policy=True,
        # Env settings
        env_id=None,
        env_class=None,
        env_kwargs=None,
        contextual_env_kwargs=None,
        exploration_policy_kwargs=None,
        evaluation_goal_sampling_mode=None,
        exploration_goal_sampling_mode=None,
        num_presampled_goals=5000,
        reward_type='encoder_distance',
        # Image env parameters
        use_image_observations=False,
        env_renderer_kwargs=None,
        # Video parameters
        save_video=True,
        save_debug_video=True,
        save_video_kwargs=None,
        video_renderer_kwargs=None,
        # Debugging parameters
        distance_scatterplot_save_period=0,
        distance_scatterplot_initial_save_period=0,
        debug_renderer_kwargs=None,
        debug_visualization_kwargs=None,
):
    if encoder_cnn_kwargs is None:
        encoder_cnn_kwargs = {}
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
    if video_renderer_kwargs is None:
        video_renderer_kwargs = {}
    if debug_renderer_kwargs is None:
        debug_renderer_kwargs = {}

    img_observation_key = 'image_observation'
    state_observation_key = 'state_observation'
    latent_observation_key = 'latent_observation'
    latent_desired_goal_key = 'latent_desired_goal'
    state_desired_goal_key = 'state_desired_goal'
    img_desired_goal_key = 'image_desired_goal'

    if use_image_observations:
        env_renderer = Renderer(**env_renderer_kwargs)

    def setup_env(state_env, encoder, reward_fn):
        goal_distribution = GoalDictDistributionFromMultitaskEnv(
            state_env,
            desired_goal_keys=[state_desired_goal_key],
        )
        if use_image_observations:
            goal_distribution = AddImageDistribution(
                env=state_env,
                base_distribution=goal_distribution,
                image_goal_key=img_desired_goal_key,
                renderer=env_renderer,
            )
            base_env = InsertImageEnv(state_env, renderer=env_renderer)
            goal_distribution = EncodedGoalDictDistribution(
                goal_distribution,
                encoder=encoder,
                keys_to_keep=[state_desired_goal_key, img_desired_goal_key],
                encoder_input_key=img_desired_goal_key,
                encoder_output_key=latent_desired_goal_key,
            )
            goal_distribution = PresampledDistribution(
                    goal_distribution, num_presampled_goals)
        else:
            base_env = state_env
            goal_distribution = EncodedGoalDictDistribution(
                goal_distribution,
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
            base_env,
            context_distribution=goal_distribution,
            reward_fn=reward_fn,
            contextual_diagnostics_fns=[state_diag_fn],
            update_env_info_fn=delete_info,
            **contextual_env_kwargs,
        )
        return env, goal_distribution

    state_expl_env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)
    state_expl_env.goal_sampling_mode = exploration_goal_sampling_mode
    state_eval_env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)
    state_eval_env.goal_sampling_mode = evaluation_goal_sampling_mode

    if use_image_observations:
        context_keys_to_save = [
            state_desired_goal_key,
            img_desired_goal_key,
            latent_desired_goal_key,
        ]
        context_key_for_rl = img_desired_goal_key
        observation_key_for_rl = img_observation_key

        def create_encoder():
            img_num_channels, img_height, img_width = env_renderer.image_chw
            cnn = BasicCNN(
                input_width=img_width,
                input_height=img_height,
                input_channels=img_num_channels,
                **encoder_cnn_kwargs
            )
            cnn_output_size = np.prod(cnn.output_shape)
            if skip_encoder_mlp:
                enc = nn.Sequential(cnn, Flatten())
                enc.output_size = cnn_output_size
            else:
                mlp = Mlp(input_size=cnn_output_size, **encoder_kwargs)
                enc = nn.Sequential(cnn, Flatten(), mlp)
                enc.output_size = mlp.output_size
            enc.input_size = img_width * img_height * img_num_channels
            return enc
    else:
        context_keys_to_save = [state_desired_goal_key, latent_desired_goal_key]
        context_key_for_rl = state_desired_goal_key
        observation_key_for_rl = state_observation_key

        def create_encoder():
            in_dim = (
                state_expl_env.observation_space.spaces[state_observation_key].low.size
            )
            return FlattenMlp(input_size=in_dim, **encoder_kwargs)

    encoder_net = create_encoder()
    target_encoder_net = create_encoder()
    encoder_input_dim = encoder_net.input_size

    encoder = EncoderFromNetwork(encoder_net)
    encoder.to(ptu.device)
    if reward_type == 'encoder_distance':
        reward_fn = EncoderRewardFnFromMultitaskEnv(
            encoder=encoder,
            next_state_encoder_input_key=observation_key_for_rl,
            context_key=latent_desired_goal_key,
            vectorize=vectorized_reward,
        )
    elif reward_type == 'target_encoder_distance':
        target_encoder = EncoderFromNetwork(target_encoder_net)
        reward_fn = EncoderRewardFnFromMultitaskEnv(
            encoder=target_encoder,
            next_state_encoder_input_key=observation_key_for_rl,
            context_key=latent_desired_goal_key,
            reward_scale=encoder_reward_scale,
            vectorize=vectorized_reward,
        )
    elif reward_type == 'state_distance':
        reward_fn = ContextualRewardFnFromMultitaskEnv(
            env=state_expl_env,
            achieved_goal_from_observation=IndexIntoAchievedGoal(
                'state_observation'
            ),
            desired_goal_key=state_desired_goal_key,
            achieved_goal_key='state_achieved_goal',
        )
    else:
        raise ValueError("invalid reward type {}".format(reward_type))
    expl_env, expl_context_distrib = setup_env(state_expl_env, encoder, reward_fn)
    eval_env, eval_context_distrib = setup_env(state_eval_env, encoder, reward_fn)

    action_dim = expl_env.action_space.low.size

    def make_qf(enc):
        return DisentangledMlpQf(
            encoder=enc,
            preprocess_obs_dim=encoder_input_dim,
            action_dim=action_dim,
            qf_kwargs=qf_kwargs,
            vectorized=vectorized_reward,
            **disentangled_qf_kwargs
        )
    qf1 = make_qf(encoder_net)
    qf2 = make_qf(encoder_net)
    target_qf1 = make_qf(target_encoder_net)
    target_qf2 = make_qf(target_encoder_net)
    if use_image_observations and not qf1.encode_state:
        raise ValueError("That doesn't make sense.")

    if use_separate_encoder_for_policy:
        policy_encoder = create_encoder()
        policy_encoder_net = EncodeObsAndGoal(
            policy_encoder,
            encoder_input_dim,
            encode_state=True,
            encode_goal=True,
            detach_encoder_via_goal=False,
            detach_encoder_via_state=False,
        )
    else:
        policy_encoder_net = EncodeObsAndGoal(
            encoder_net,
            encoder_input_dim,
            **policy_using_encoder_settings
        )
    obs_processor = nn.Sequential(
        policy_encoder_net,
        ConcatTuple(),
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

    if use_image_observations:
        # Do this so that the context has all two/three: the state, image, and
        # encoded goal
        sample_context_from_observation = compose(
            RemapKeyFn({
                state_desired_goal_key: state_observation_key,
                img_desired_goal_key: img_observation_key,
            }),
            ReEncoderAchievedStateFn(
                encoder=encoder,
                encoder_input_key=context_key_for_rl,
                encoder_output_key=latent_desired_goal_key,
                keys_to_keep=[state_desired_goal_key, img_desired_goal_key],
            ),
        )
        ob_keys_to_save_in_buffer = [state_observation_key, img_observation_key]
    else:
        sample_context_from_observation = compose(
            RemapKeyFn({
                state_desired_goal_key: state_observation_key,
            }),
            ReEncoderAchievedStateFn(
                encoder=encoder,
                encoder_input_key=context_key_for_rl,
                encoder_output_key=latent_desired_goal_key,
                keys_to_keep=[state_desired_goal_key],
            ),
        )
        ob_keys_to_save_in_buffer = [state_observation_key]

    encoder_output_dim = encoder_net.output_size
    replay_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_keys=context_keys_to_save,
        context_distribution=expl_context_distrib,
        sample_context_from_obs_dict_fn=sample_context_from_observation,
        observation_keys=ob_keys_to_save_in_buffer,
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
    if not use_image_observations:
        # TODO: implement this for images
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

    video_renderer = Renderer(**video_renderer_kwargs)
    if save_video:
        rollout_function = partial(
            rf.contextual_rollout,
            max_path_length=max_path_length,
            observation_key=observation_key_for_rl,
            context_keys_for_policy=[context_key_for_rl],
        )
        if save_debug_video and not use_image_observations:
            # TODO: add visualization for image-based envs
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

            low = eval_env.env.observation_space[state_observation_key].low.min()
            high = eval_env.env.observation_space[state_observation_key].high.max()
            y = np.linspace(low, high, num=debugger_one.image_shape[0])
            x = np.linspace(low, high, num=debugger_one.image_shape[1])
            cross = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

            def create_shared_data_creator(obj_index):
                def compute_shared_data(raw_obs, env):
                    state = raw_obs[observation_key_for_rl]
                    obs = state[:2]
                    goal = state[2:]
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
                if use_image_observations:
                    img_env = env
                    image_goal_distribution = base_distribution
                else:
                    state_env = env.env
                    image_goal_distribution = AddImageDistribution(
                        env=state_env,
                        base_distribution=base_distribution,
                        image_goal_key='image_desired_goal',
                        renderer=video_renderer,
                    )
                    img_env = InsertImageEnv(state_env, renderer=video_renderer)
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
            img_formats = [video_renderer.output_image_format]
            for r in obj1_sweep_renderers.values():
                img_formats.append(r.output_image_format)
            for r in obj0_sweep_renderers.values():
                img_formats.append(r.output_image_format)
            eval_video_func = get_save_video_function(
                rollout_function,
                img_eval_env,
                MakeDeterministic(policy),
                tag="eval",
                imsize=video_renderer.image_chw[1],
                image_formats=img_formats,
                get_extra_imgs=get_extra_imgs,
                **save_video_kwargs
            )
            expl_video_func = get_save_video_function(
                rollout_function,
                img_expl_env,
                exploration_policy,
                tag="train",
                imsize=video_renderer.image_chw[1],
                image_formats=img_formats,
                get_extra_imgs=get_extra_imgs,
                **save_video_kwargs
            )
        else:
            video_renderer = Renderer(**video_renderer_kwargs)

            def add_images(env, base_distribution):
                video_env = InsertImageEnv(
                    env, renderer=video_renderer, image_key='image_observation')
                if use_image_observations:
                    image_goal_distribution = base_distribution
                else:
                    state_env = env.env
                    image_goal_distribution = AddImageDistribution(
                        env=state_env,
                        base_distribution=base_distribution,
                        image_goal_key='image_desired_goal',
                        renderer=video_renderer,
                    )
                return ContextualEnv(
                    video_env,
                    context_distribution=image_goal_distribution,
                    reward_fn=reward_fn,
                    observation_key=observation_key_for_rl,
                    update_env_info_fn=delete_info,
                )

            img_eval_env = add_images(eval_env, eval_context_distrib)
            img_expl_env = add_images(expl_env, expl_context_distrib)

            if use_image_observations:
                keys_to_show = [
                    'image_desired_goal',
                    'image_observation',
                    'video_observation',
                ]
                image_formats = [
                    env_renderer.output_image_format,
                    env_renderer.output_image_format,
                    video_renderer.output_image_format,
                ]
            else:
                keys_to_show = ['image_desired_goal', 'image_observation']
                image_formats = [
                    video_renderer.output_image_format,
                    video_renderer.output_image_format,
                ]
            eval_video_func = get_save_video_function(
                rollout_function,
                img_eval_env,
                MakeDeterministic(policy),
                tag="eval",
                imsize=video_renderer.image_chw[1],
                keys_to_show=keys_to_show,
                image_formats=image_formats,
                **save_video_kwargs
            )
            expl_video_func = get_save_video_function(
                rollout_function,
                img_expl_env,
                exploration_policy,
                tag="train",
                imsize=video_renderer.image_chw[1],
                keys_to_show=keys_to_show,
                image_formats=image_formats,
                **save_video_kwargs
            )

        algorithm.post_train_funcs.append(eval_video_func)
        algorithm.post_train_funcs.append(expl_video_func)
    if use_image_observations:
        def state_to_encoder_input(state):
            goal_dict = {
                'state_desired_goal': state,
            }
            env_state = state_eval_env.get_env_state()
            state_eval_env.set_to_goal(goal_dict)
            start_img = env_renderer.create_image(state_eval_env)
            state_eval_env.set_env_state(env_state)
            return start_img
        visualize_representation = create_visualize_representation(
            encoder, True, eval_env, video_renderer,
            state_to_encoder_input=state_to_encoder_input,
            env_renderer=env_renderer,
            **debug_visualization_kwargs
        )
        algorithm.post_train_funcs.append(visualize_representation)
        visualize_representation = create_visualize_representation(
            encoder, False, eval_env, video_renderer,
            state_to_encoder_input=state_to_encoder_input,
            env_renderer=env_renderer,
            **debug_visualization_kwargs
        )
        algorithm.post_train_funcs.append(visualize_representation)
    else:
        visualize_representation = create_visualize_representation(
            encoder, True, eval_env, video_renderer,
            **debug_visualization_kwargs
        )
        algorithm.post_train_funcs.append(visualize_representation)
        visualize_representation = create_visualize_representation(
            encoder, False, eval_env, video_renderer,
            **debug_visualization_kwargs
        )
        algorithm.post_train_funcs.append(visualize_representation)

    if distance_scatterplot_save_period > 0:
        algorithm.post_train_funcs.append(create_save_h_vs_state_distance_fn(
            distance_scatterplot_save_period,
            distance_scatterplot_initial_save_period,
            encoder,
            observation_key_for_rl,
        ))
    algorithm.train()


def create_save_h_vs_state_distance_fn(
        save_period, initial_save_period, encoder, encoder_input_key):
    import matplotlib.pyplot as plt
    from railrl.core import logger
    import os.path as osp

    logdir = logger.get_snapshot_dir()

    def save_h_vs_state_distance(algo, epoch):
        if (
                (epoch < save_period and epoch % initial_save_period == 0)
                or epoch % save_period == 0
                or epoch >= algo.num_epochs - 1
        ):
            filename = osp.join(
                logdir,
                'h_vs_distance_scatterplot_{epoch}.png'.format(epoch=epoch))
            replay_buffer = algo.replay_buffer
            size = min(1024, replay_buffer._size)
            idxs1 = replay_buffer._sample_indices(size)
            idxs2 = replay_buffer._sample_indices(size)
            encoder_obs = replay_buffer._obs[encoder_input_key]
            x1 = encoder_obs[idxs1]
            x2 = encoder_obs[idxs2]
            z1 = encoder.encode(x1)
            z2 = encoder.encode(x2)

            state_obs = replay_buffer._obs['state_observation']
            states1 = state_obs[idxs1]
            states2 = state_obs[idxs2]
            state_deltas = np.linalg.norm(states1 - states2, axis=1, ord=1)
            encoder_deltas = np.linalg.norm(z1 - z2, axis=1, ord=1)

            plt.clf()
            plt.scatter(state_deltas, encoder_deltas, alpha=0.2)
            plt.savefig(filename)

    return save_h_vs_state_distance


def compose(*functions):
    def composite_function(x):
        for f in functions:
            x = f(x)
        return x

    return composite_function
