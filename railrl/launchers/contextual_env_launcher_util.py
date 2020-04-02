from functools import partial

import numpy as np

import railrl.samplers.rollout_functions as rf
import railrl.torch.pytorch_util as ptu
from railrl.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
    SelectKeyFn,
)
from railrl.envs.contextual import ContextualEnv
from railrl.envs.contextual.goal_conditioned import (
    GoalDistributionFromMultitaskEnv,
    ContextualRewardFnFromMultitaskEnv,
)
from railrl.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)
from railrl.visualization.video import get_save_video_function
from railrl.torch.networks import FlattenMlp
from railrl.torch.sac.policies import MakeDeterministic
from railrl.torch.sac.policies import TanhGaussianPolicy
from railrl.torch.sac.sac import SACTrainer
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


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
        contextual_env_kwargs=None,
        evaluation_goal_sampling_mode=None,
        exploration_goal_sampling_mode=None,
        # Video parameters
        save_video=True,
        save_video_kwargs=None,
):
    if contextual_env_kwargs is None:
        contextual_env_kwargs = {}
    if not save_video_kwargs:
        save_video_kwargs = {}

    def contextual_env_distrib_and_reward(
            env_id, env_class, env_kwargs, goal_sampling_mode
    ):
        env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)
        env.goal_sampling_mode = goal_sampling_mode
        goal_distribution = GoalDistributionFromMultitaskEnv(
            env,
            desired_goal_key=desired_goal_key,
        )
        reward_fn = ContextualRewardFnFromMultitaskEnv(
            env=env,
            desired_goal_key=desired_goal_key,
            achieved_goal_key=achieved_goal_key,
        )
        env = ContextualEnv(
            env,
            context_distribution=goal_distribution,
            reward_fn=reward_fn,
            observation_key=observation_key,
            **contextual_env_kwargs,
        )
        return env, goal_distribution, reward_fn
    expl_env, expl_context_distrib, expl_reward = contextual_env_distrib_and_reward(
        env_id, env_class, env_kwargs, exploration_goal_sampling_mode
    )
    eval_env, eval_context_distrib, eval_reward = contextual_env_distrib_and_reward(
        env_id, env_class, env_kwargs, evaluation_goal_sampling_mode
    )
    context_key = desired_goal_key

    obs_dim = (
            expl_env.observation_space.spaces[observation_key].low.size
            + expl_env.observation_space.spaces[desired_goal_key].low.size
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

    ob_keys_to_save = [
        observation_key,
        desired_goal_key,
        achieved_goal_key,
    ]

    def concat_context_to_obs(batch):
        obs = batch['observations']
        next_obs = batch['next_observations']
        context = batch['contexts']
        batch['observations'] = np.concatenate([obs, context], axis=1)
        batch['next_observations'] = np.concatenate([next_obs, context], axis=1)
        return batch
    replay_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_key=desired_goal_key,
        context_distribution=eval_context_distrib,
        sample_context_from_obs_dict_fn=SelectKeyFn(achieved_goal_key),
        ob_keys_to_save=ob_keys_to_save,
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
        context_key=context_key,
    )
    expl_path_collector = ContextualPathCollector(
        expl_env,
        policy,
        observation_key=observation_key,
        context_key=context_key,
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
            context_key=context_key,
        )
        eval_video_func = get_save_video_function(
            rollout_function,
            eval_env,
            MakeDeterministic(policy),
            tag="eval",
            **save_video_kwargs
        )
        train_video_func = get_save_video_function(
            rollout_function,
            expl_env,
            policy,
            tag="train",
            **save_video_kwargs
        )

        algorithm.post_train_funcs.append(eval_video_func)
        algorithm.post_train_funcs.append(train_video_func)

    algorithm.train()


def get_gym_env(env_id, env_class=None, env_kwargs=None):
    if env_kwargs is None:
        env_kwargs = {}

    assert env_id or env_class
    if env_id:
        import gym
        import multiworld
        multiworld.register_all_envs()
        env = gym.make(env_id)
    else:
        env = env_class(**env_kwargs)
    return env
