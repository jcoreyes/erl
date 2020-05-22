from functools import partial

import numpy as np

from railrl.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
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
)
from railrl.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)

from railrl.launchers.rl_exp_launcher_util import (
    preprocess_rl_variant,
    get_envs,
    get_exploration_strategy,
)

from gym.spaces import Box
from railrl.samplers.rollout_functions import contextual_rollout

class TaskGoalDictDistributionFromMultitaskEnv(
        GoalDictDistributionFromMultitaskEnv):
    def __init__(
            self,
            *args,
            task_key='task_id',
            task_ids=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.task_key = task_key
        self._spaces[task_key] = Box(
            low=np.zeros(1),
            high=np.ones(1))
        self.task_ids = np.array(task_ids)

    def sample(self, batch_size: int, use_env_goal=False):
        goals = super().sample(batch_size, use_env_goal)
        idxs = np.random.choice(len(self.task_ids), batch_size)
        goals[self.task_key] = self.task_ids[idxs].reshape(-1, 1)
        return goals

class MaskedGoalDictDistributionFromMultitaskEnv(
        GoalDictDistributionFromMultitaskEnv):
    def __init__(
            self,
            *args,
            mask_dim=1,
            mask_key='mask',
            mask_idxs=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mask_key = mask_key
        self._spaces[mask_key] = Box(
            low=np.zeros(mask_dim),
            high=np.ones(mask_dim))
        self.mask_dim = mask_dim
        self.mask_idxs = mask_idxs
        if mask_idxs is not None:
            self.masks = np.zeros((len(self.mask_idxs), self.mask_dim))
            for (i, idx_list) in enumerate(self.mask_idxs):
                self.masks[i][idx_list] = 1

    def sample(self, batch_size: int, use_env_goal=False):
        goals = super().sample(batch_size, use_env_goal)
        if self.mask_idxs is not None:
            idxs = np.random.choice(len(self.masks), batch_size)
            goals[self.mask_key] = self.masks[idxs]
        else:
            goals[self.mask_key] = np.zeros((batch_size, self.mask_dim))
        return goals

class SequentialTaskPathCollector(ContextualPathCollector):
    def __init__(
            self,
            *args,
            task_key=None,
            max_path_length=100,
            task_ids=None,
            rotate_freq=0.0,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.rotate_freq = rotate_freq
        self.rollout_tasks = []

        def obs_processor(o):
            if len(self.rollout_tasks) > 0:
                task = self.rollout_tasks[0]
                self.rollout_tasks = self.rollout_tasks[1:]
                o[task_key] = task
                self._env._rollout_context_batch[task_key] = task[None]

            combined_obs = [o[self._observation_key]]
            for k in self._context_keys_for_policy:
                combined_obs.append(o[k])
            return np.concatenate(combined_obs, axis=0)

        def reset_postprocess_func():
            rotate = (np.random.uniform() < self.rotate_freq)
            self.rollout_tasks = []
            if rotate:
                num_steps_per_task = max_path_length // len(task_ids)
                self.rollout_tasks = np.ones((max_path_length, 1)) * (len(task_ids) - 1)
                for (idx, id) in enumerate(task_ids):
                    start = idx * num_steps_per_task
                    end = start + num_steps_per_task
                    self.rollout_tasks[start:end] = id

        self._rollout_fn = partial(
            contextual_rollout,
            context_keys_for_policy=self._context_keys_for_policy,
            observation_key=self._observation_key,
            obs_processor=obs_processor,
            reset_postprocess_func=reset_postprocess_func,
        )

def td3_experiment(variant):
    import railrl.torch.pytorch_util as ptu
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

    task_variant = variant.get('task_variant', {})
    task_conditioned = task_variant.get('task_conditioned', False)
    task_key = 'task_id'

    mask_variant = variant.get('mask_variant', {})
    env = get_envs(variant)
    mask_conditioned = mask_variant.get('mask_conditioned', False)
    mask_dim = env.observation_space.spaces[context_key].low.size
    mask_key = 'mask'

    assert not (task_conditioned and mask_conditioned)

    if task_conditioned:
        context_keys = [context_key, task_key]
    elif mask_conditioned:
        context_keys = [context_key, mask_key]
    else:
        context_keys = [context_key]

    def contextual_env_distrib_and_reward(goal_sampling_mode):
        env = get_envs(variant)
        env.goal_sampling_mode = goal_sampling_mode
        if task_conditioned:
            goal_distribution = TaskGoalDictDistributionFromMultitaskEnv(
                env,
                desired_goal_keys=[desired_goal_key],
                task_key=task_key,
                task_ids=task_variant['task_ids']
            )
            reward_fn = ContextualRewardFnFromMultitaskEnv(
                env=env,
                achieved_goal_from_observation=IndexIntoAchievedGoal(observation_key), # achieved_goal_key
                desired_goal_key=desired_goal_key,
                achieved_goal_key=achieved_goal_key,
                additional_obs_keys=variant['contextual_replay_buffer_kwargs'].get('observation_keys', None),
                additional_context_keys=[task_key],
            )
        elif mask_conditioned:
            goal_distribution = MaskedGoalDictDistributionFromMultitaskEnv(
                env,
                desired_goal_keys=[desired_goal_key],
                mask_key=mask_key,
                mask_dim=mask_dim,
                mask_idxs=mask_variant.get('mask_idxs', None),
            )
            reward_fn = ContextualRewardFnFromMultitaskEnv(
                env=env,
                achieved_goal_from_observation=IndexIntoAchievedGoal(observation_key), # achieved_goal_key
                desired_goal_key=desired_goal_key,
                achieved_goal_key=achieved_goal_key,
                additional_obs_keys=variant['contextual_replay_buffer_kwargs'].get('observation_keys', None),
                additional_context_keys=[mask_key],
            )
        else:
            goal_distribution = GoalDictDistributionFromMultitaskEnv(
                env,
                desired_goal_keys=[desired_goal_key],
            )
            reward_fn = ContextualRewardFnFromMultitaskEnv(
                env=env,
                achieved_goal_from_observation=IndexIntoAchievedGoal(observation_key), # achieved_goal_key
                desired_goal_key=desired_goal_key,
                achieved_goal_key=achieved_goal_key,
                additional_obs_keys=variant['contextual_replay_buffer_kwargs'].get('observation_keys', None),
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
            # update_env_info_fn=delete_info,
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
    if task_conditioned:
        obs_dim += 1
    elif mask_conditioned:
        obs_dim += mask_dim
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

    def context_from_obs_dict_fn(obs_dict):
        context_dict = {
            context_key: obs_dict[observation_key],
        }
        if task_conditioned:
            context_dict[task_key] = obs_dict[task_key]
        elif mask_conditioned:
            context_dict[mask_key] = obs_dict[mask_key]
        return context_dict

    def concat_context_to_obs(batch):
        obs = batch['observations']
        next_obs = batch['next_observations']
        context = batch[context_key]
        if task_conditioned:
            task = batch[task_key]
            batch['observations'] = np.concatenate([obs, context, task], axis=1)
            batch['next_observations'] = np.concatenate([next_obs, context, task], axis=1)
        elif mask_conditioned:
            mask = batch[mask_key]
            batch['observations'] = np.concatenate([obs, context, mask], axis=1)
            batch['next_observations'] = np.concatenate([next_obs, context, mask], axis=1)
        else:
            batch['observations'] = np.concatenate([obs, context], axis=1)
            batch['next_observations'] = np.concatenate([next_obs, context], axis=1)
        return batch

    if 'observation_keys' not in variant['contextual_replay_buffer_kwargs']:
        variant['contextual_replay_buffer_kwargs']['observation_keys'] = []
    observation_keys = variant['contextual_replay_buffer_kwargs']['observation_keys']
    if observation_key not in observation_keys:
        observation_keys.append(observation_key)

    replay_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_keys=context_keys,
        context_distribution=eval_context_distrib,
        sample_context_from_obs_dict_fn=context_from_obs_dict_fn,
        reward_fn=eval_reward,
        post_process_batch_fn=concat_context_to_obs,
        **variant['contextual_replay_buffer_kwargs']
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

    def create_path_collector(env, policy, mode='expl'):
        assert mode in ['expl', 'eval']

        if task_conditioned:
            rotate_freq = task_variant['rotate_task_freq_for_expl'] if mode == 'expl' \
                else task_variant['rotate_task_freq_for_eval']
            return SequentialTaskPathCollector(
                env,
                policy,
                observation_key=observation_key,
                context_keys_for_policy=context_keys,
                task_key=task_key,
                max_path_length=max_path_length,
                task_ids=task_variant['task_ids'],
                rotate_freq=rotate_freq,
            )
        elif mask_conditioned:
            return ContextualPathCollector(
                env,
                policy,
                observation_key=observation_key,
                context_keys_for_policy=context_keys,
            )
        else:
            return ContextualPathCollector(
                env,
                policy,
                observation_key=observation_key,
                context_keys_for_policy=context_keys,
            )

    expl_path_collector = create_path_collector(expl_env, expl_policy, mode='expl')
    eval_path_collector = create_path_collector(eval_env, policy, mode='eval')

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

    if variant.get("save_video", True):
        save_period = variant.get('save_video_period', 50)
        dump_video_kwargs = variant.get("dump_video_kwargs", dict())
        dump_video_kwargs['horizon'] = max_path_length

        # rollout_function = partial(
        #     rf.contextual_rollout,
        #     max_path_length=max_path_length,
        #     observation_key=observation_key,
        #     context_keys_for_policy=context_keys,
        # )
        renderer = Renderer(**variant.get('renderer_kwargs', {}))

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

        video_path_collector = create_path_collector(img_eval_env, policy, mode='eval')
        rollout_function = video_path_collector._rollout_fn

        eval_video_func = get_save_video_function(
            rollout_function,
            img_eval_env,
            policy,
            tag="",
            imsize=renderer.image_shape[0],
            image_format='HWC',
            save_video_period=save_period,
            **dump_video_kwargs
        )

        algorithm.post_train_funcs.append(eval_video_func)


    algorithm.train()
