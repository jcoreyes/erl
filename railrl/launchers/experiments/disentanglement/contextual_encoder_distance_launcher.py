from functools import partial

import numpy as np
from multiworld.core.multitask_env import MultitaskEnv

import railrl.samplers.rollout_functions as rf
import railrl.torch.pytorch_util as ptu
from railrl.core.distribution import DictDistribution
from railrl.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
    SampleContextFromObsDictFn,
    RemapKeyFn,
)
from railrl.envs.contextual import ContextualEnv, ContextualRewardFn
from railrl.envs.contextual.goal_conditioned import (
    GoalDictDistributionFromMultitaskEnv,
    AddImageDistribution,
)
from railrl.envs.images import Renderer, InsertImageEnv
from railrl.launchers.contextual_env_launcher_util import (
    DeleteOldEnvInfo,
    get_gym_env,
    get_save_video_function,
)
from railrl.policies.action_repeat import ActionRepeatPolicy
from railrl.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)
from railrl.torch.disentanglement.encoder_wrapped_env import (
    Encoder,
    EncoderFromNetwork,
)
from railrl.torch.disentanglement.networks import DisentangledMlpQf
from railrl.torch.modules import Detach
from railrl.torch.networks import FlattenMlp
from railrl.torch.sac.policies import MakeDeterministic
from railrl.torch.sac.policies import TanhGaussianPolicy
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
            import ipdb; ipdb.set_trace()
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
        policy_kwargs,
        algo_kwargs,
        # Encoder parameters
        disentangled_qf_kwargs,
        encoder_kwargs=None,
        vectorized_reward=None,
        use_target_encoder_for_reward=False,
        encoder_reward_scale=1.,
        detach_encoder=False,
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
):
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

    state_observation_key = 'state_observation'
    latent_desired_goal_key = 'latent_desired_goal'
    state_desired_goal_key = 'state_desired_goal'
    context_key_for_rl = 'state_desired_goal'

    def setup_env(env, encoder, reward_fn):
        goal_distribution = EncodedGoalDictDistributionFromMultitaskEnv(
            env,
            encoder=encoder,
            keys_to_keep=[state_desired_goal_key],
            encoder_input_key=state_desired_goal_key,
            encoder_output_key=latent_desired_goal_key,
        )
        env = ContextualEnv(
            env,
            context_distribution=goal_distribution,
            reward_fn=reward_fn,
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
    target_encoder_net = FlattenMlp(input_size=state_dim, **encoder_kwargs)
    encoder = EncoderFromNetwork(encoder_net)
    if use_target_encoder_for_reward:
        target_encoder = EncoderFromNetwork(target_encoder_net)
        reward_fn = EncoderRewardFnFromMultitaskEnv(
            encoder=target_encoder,
            next_state_encoder_input_key=state_observation_key,
            context_key=latent_desired_goal_key,
            reward_scale=encoder_reward_scale,
        )
    else:
        reward_fn = EncoderRewardFnFromMultitaskEnv(
            encoder=encoder,
            next_state_encoder_input_key=state_observation_key,
            context_key=latent_desired_goal_key,
        )
    expl_env, expl_context_distrib = setup_env(expl_env, encoder, reward_fn)
    eval_env, eval_context_distrib = setup_env(eval_env, encoder, reward_fn)

    action_dim = expl_env.action_space.low.size

    def make_qf(enc):
        if detach_encoder:
            enc = Detach(enc)
        return DisentangledMlpQf(
            goal_processor=enc,
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

    obs_dim = (
            expl_env.observation_space.spaces[state_observation_key].low.size
            + expl_env.observation_space.spaces[context_key_for_rl].low.size
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **policy_kwargs
    )

    def concat_context_to_obs(batch):
        obs = batch['observations']
        next_obs = batch['next_observations']
        context = batch[context_key_for_rl]
        batch['observations'] = np.concatenate([obs, context], axis=1)
        batch['next_observations'] = np.concatenate([next_obs, context], axis=1)
        return batch

    sample_context = compose(
        # first map 'state_observation' --> 'state_desired_goal'
        RemapKeyFn({
            context_key_for_rl: state_observation_key
        }),
        # them map `state_desired_goal` -> `latent_desired_goal`
        ReEncoderAchievedStateFn(
            encoder=encoder,
            encoder_input_key=context_key_for_rl,
            encoder_output_key=latent_desired_goal_key,
        ),
    )

    replay_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_keys=[state_desired_goal_key, latent_desired_goal_key],
        context_distribution=eval_context_distrib,
        sample_context_from_obs_dict_fn=sample_context,
        observation_keys=[state_observation_key],
        reward_fn=reward_fn,
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
        observation_key=state_observation_key,
        context_key=context_key_for_rl,
    )
    exploration_policy = create_exploration_policy(
        policy, **exploration_policy_kwargs)
    expl_path_collector = ContextualPathCollector(
        expl_env,
        exploration_policy,
        observation_key=state_observation_key,
        context_key=context_key_for_rl,
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
            observation_key=state_observation_key,
            context_key=context_key_for_rl,
        )
        renderer = Renderer(**renderer_kwargs)

        def add_images(env, base_distribution):
            state_env = env.env
            image_goal_distribution = AddImageDistribution(
                env=state_env,
                base_distribution=base_distribution,
                image_goal_key='image_desired_goal',
                renderer=renderer,
            )
            img_env = InsertImageEnv(state_env, renderer=renderer)
            return ContextualEnv(
                img_env,
                context_distribution=image_goal_distribution,
                reward_fn=reward_fn,
                observation_key=state_observation_key,
                update_env_info_fn=DeleteOldEnvInfo(),
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


def compose(*functions):
    def composite_function(x):
        for f in functions:
            x = f(x)
        return x
    return composite_function
