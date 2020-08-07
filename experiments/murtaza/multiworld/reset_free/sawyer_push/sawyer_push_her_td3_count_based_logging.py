import pickle
import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import SawyerPushAndReachXYEnv
from rlkit.data_management.obs_dict_count_based_replay_buffer import ObsDictCountBasedRelabelingBuffer
from rlkit.exploration_strategies.count_based.count_based_goal_sampling_env import CountBasedGoalSamplingEnv
from rlkit.launchers.launcher_util import run_experiment
import rlkit.torch.pytorch_util as ptu
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.torch.grill.launcher import get_video_save_func
from rlkit.torch.her.her_td3 import HerTd3
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
import rlkit.samplers.rollout_functions as rf

def her_td3_experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    env = CountBasedGoalSamplingEnv(wrapped_env=env, **variant['count_based_sampler_kwargs'])
    observation_key = variant['observation_key']
    desired_goal_key = variant['desired_goal_key']
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    exploration_counter = pickle.loads(pickle.dumps(env.goal_sampler))
    replay_buffer = ObsDictCountBasedRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        exploration_counter=exploration_counter,
        **variant['replay_buffer_kwargs']
    )

    obs_dim = env.observation_space.spaces['observation'].low.size
    action_dim = env.action_space.low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
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
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = HerTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        training_env=env,
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
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=1001,
                num_steps_per_epoch=1000,
                num_steps_per_eval=5000,
                max_path_length=500,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=0.99,
                min_num_steps_before_training=128,
                reward_scale=100,
            ),
            her_kwargs=dict(
                observation_key='state_observation',
                desired_goal_key='state_desired_goal',
            ),
            td3_kwargs=dict(),
        ),
        env_class=SawyerPushAndReachXYEnv,
        env_kwargs=dict(
            reward_type='puck_distance',
            reset_free=True,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
            num_resets_before_puck_reset=int(1e6),
        ),
        replay_buffer_class=ObsDictCountBasedRelabelingBuffer,
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=0.5,
            fraction_resampled_goals_are_env_goals=0.5,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        normalize=False,
        algorithm='HER-TD3',
        version='normal',
        es_kwargs=dict(
            max_sigma=.8,
        ),
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        exploration_type='ou',
        save_video_period=500,
        do_state_exp=True,
        init_camera=sawyer_pusher_camera_upright_v2,
        save_video=True,
        count_based_sampler_kwargs=dict(
            num_samples=1000,
            goal_key='state_observation',
            use_count_based_goal=False,
            hash_dim=16,
        ),
    )
    search_space = {
        'count_based_sampler_kwargs.count_based_reward_scale':[.01, .1, 0],
        'env_kwargs.reward_type':['puck_distance', 'state_distance'],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds= 1
    # mode='local'
    # exp_prefix= 'test'

    n_seeds=1
    mode = 'ec2'
    exp_prefix = 'sawyer_push_env_her_td3_count_based_reset_hand_with_puck'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            run_experiment(
                her_td3_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=2,
            )
