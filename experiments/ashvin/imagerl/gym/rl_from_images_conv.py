from rlkit.envs.multitask.multitask_env import MultitaskToFlatEnv
from rlkit.envs.multitask.point2d import MultitaskPoint2DEnv
from rlkit.envs.mujoco.pusher2d import Pusher2DEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import (
    ConcatMlp, TanhMlpPolicy, ImageStatePolicy, ImageStateQ,
    MergedCNN, CNNPolicy,
)
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.td3.td3 import TD3
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from multiworld.core.gym_to_multi_env import MujocoGymToMultiEnv
from multiworld.core.image_env import ImageEnv

import gym
import torch

def her_td3_experiment(variant):
    import multiworld.envs.mujoco
    import multiworld.envs.pygame
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
    from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
    from rlkit.exploration_strategies.ou_strategy import OUStrategy
    from rlkit.torch.grill.launcher import get_video_save_func
    from rlkit.torch.her.her_td3 import HerTd3
    from rlkit.data_management.obs_dict_replay_buffer import (
        ObsDictRelabelingBuffer
    )

    if 'env_id' in variant:
        env = gym.make(variant['env_id'])
    else:
        env = variant['env_class'](**variant['env_kwargs'])

    imsize = 84
    env = MujocoGymToMultiEnv(env.env) # unwrap TimeLimit
    env = ImageEnv(env, non_presampled_goal_img_is_garbage=True, recompute_reward=False)

    observation_key = variant['observation_key']
    desired_goal_key = variant['desired_goal_key']
    variant['algo_kwargs']['her_kwargs']['observation_key'] = observation_key
    variant['algo_kwargs']['her_kwargs']['desired_goal_key'] = desired_goal_key
    if variant.get('normalize', False):
        raise NotImplementedError()

    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    obs_dim = env.observation_space.spaces[observation_key].low.size
    action_dim = env.action_space.low.size
    goal_dim = env.observation_space.spaces[desired_goal_key].low.size
    exploration_type = variant['exploration_type']
    if exploration_type == 'ou':
        es = OUStrategy(
            action_space=env.action_space,
            **variant['es_kwargs']
        )
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            **variant['es_kwargs'],
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            **variant['es_kwargs'],
        )
    else:
        raise Exception("Invalid type: " + exploration_type)

    use_images_for_q = variant["use_images_for_q"]
    use_images_for_pi = variant["use_images_for_pi"]

    qs = []
    for i in range(2):
        if use_images_for_q:
            image_q = MergedCNN(input_width=imsize,
                   input_height=imsize,
                   output_size=1,
                   input_channels=3,
                   added_fc_input_size=action_dim,
                   **variant['cnn_params'])
            q = ImageStateQ(image_q, None)
        else:
            state_q = ConcatMlp(
                input_size=action_dim + goal_dim,
                output_size=1,
                **variant['qf_kwargs']
            )
            q = ImageStateQ(None, state_q)
        qs.append(q)
    qf1, qf2 = qs

    if use_images_for_pi:
        image_policy = CNNPolicy(input_width=imsize,
                           input_height=imsize,
                           output_size=action_dim,
                           input_channels=3,
                           **variant['cnn_params'],
                           output_activation=torch.tanh,
        )
        policy = ImageStatePolicy(image_policy, None)
    else:
        state_policy = TanhMlpPolicy(
            input_size=goal_dim,
            output_size=action_dim,
            **variant['policy_kwargs']
        )
        policy = ImageStatePolicy(None, state_policy)

    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = HerTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    if variant.get("save_video", False):
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            policy,
            variant,
        )
        algorithm.post_epoch_funcs.append(video_func)
    algorithm.to(ptu.device)
    algorithm.train()

def experiment(variant):

    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    exploration_type = variant['exploration_type']
    if exploration_type == 'ou':
        es = OUStrategy(action_space=env.action_space)
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=0.1,
            min_sigma=0.1,  # Constant sigma
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=0.1,
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = TD3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=1001,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=100,
                num_updates_per_env_step=4,
                batch_size=128,
                discount=0.99,
                min_num_steps_before_training=4000,
                reward_scale=100,
                render=False,
                collection_mode='online',
                parallel_env_params=dict(
                    num_workers=1,
                )
            ),
            her_kwargs=dict(
                observation_key='state_observation',
                desired_goal_key='state_desired_goal',
            ),
            td3_kwargs=dict(),
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=1.0,
            fraction_resampled_goals_are_env_goals=0.0,
            ob_keys_to_save=[],
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        algorithm='HER-TD3',
        version='normal',
        es_kwargs=dict(
            max_sigma=.8,
        ),
        exploration_type='ou',
        observation_key='image_observation',
        desired_goal_key='state_observation',
        do_state_exp=True,
        save_video=False,

        snapshot_mode='gap_and_last',
        snapshot_gap=100,

        cnn_params=dict(
            kernel_sizes=[3, 3],
            n_channels=[16, 16],
            strides=[2, 2],
            # pool_sizes=[1, 1],
            hidden_sizes=[400, 300],
            paddings=[0, 0],
            use_batch_norm=False,
        ),

        use_images_for_q=True,
        use_images_for_pi=True,
    )

    n_seeds = 1

    search_space = {
        'exploration_type': [
            'ou',
        ],
        'algo_kwargs.base_kwargs.reward_scale': [0.1, 1, 10],
        'seedid': range(n_seeds),
        'env_id': ['Humanoid-v2', 'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2'],
        'use_images_for_q': [True, False],
        'use_images_for_pi': [True, False],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    run_variants(her_td3_experiment, sweeper.iterate_hyperparameters(), run_id=0)
