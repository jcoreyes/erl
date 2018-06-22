import pickle

import railrl.torch.pytorch_util as ptu
from multiworld.core.flat_goal_env import FlatGoalEnv
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from railrl.envs.multitask.multitask_env import \
    MultitaskEnvToSilentMultitaskEnv, MultiTaskHistoryEnv
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
from railrl.torch.sac.policies import TanhGaussianPolicy
from railrl.state_distance.tdm_networks import TdmQf, TdmPolicy
# from railrl.torch.her.her_twin_sac import HerTwinSac
# from railrl.torch.her.her_sac import HerSac

from railrl.state_distance.tdm_td3 import TdmTd3
from railrl.misc.ml_util import IntPiecewiseLinearSchedule


def tdm_td3_experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])

    render = variant["render"]
    # env = MultitaskEnvToSilentMultitaskEnv(env)
    if render:
        env.pause_on_goal = True

    if variant['normalize']:
        env = NormalizedBoxEnv(env)

    exploration_type = variant['exploration_type']
    if exploration_type == 'ou':
        es = OUStrategy(
            action_space=env.action_space,
            max_sigma=0.1,
            **variant['es_kwargs']
        )
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=0.1,
            min_sigma=0.1,  # Constant sigma
            **variant['es_kwargs'],
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=0.1,
            **variant['es_kwargs'],
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    obs_dim = env.observation_space.spaces['observation'].low.size
    action_dim = env.action_space.low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    vectorized = variant['algo_kwargs']['tdm_kwargs'].get('vectorized', True)
    qf1 = TdmQf(
        env=env,
        observation_dim=obs_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        vectorized=vectorized,
        **variant['qf_kwargs']
    )
    qf2 = TdmQf(
        env=env,
        observation_dim=obs_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        vectorized=vectorized,
        **variant['qf_kwargs']
    )
    policy = TdmPolicy(
        env=env,
        observation_dim=obs_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    relabeling_env = pickle.loads(pickle.dumps(env))

    observation_key = variant.get('observation_key', 'observation')
    desired_goal_key = variant.get('desired_goal_key', 'desired_goal')
    replay_buffer = ObsDictRelabelingBuffer(
        env=relabeling_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['replay_buffer_kwargs']
    )

    # qf_criterion = variant['qf_criterion_class']()
    algo_kwargs = variant['algo_kwargs']
    # algo_kwargs['td3_kwargs']['qf_criterion'] = qf_criterion
    algo_kwargs['tdm_kwargs']['env_samples_goal_on_reset'] = True
    algo_kwargs['td3_kwargs']['training_env'] = env
    if 'tau_schedule_kwargs' in variant:
        tau_schedule = IntPiecewiseLinearSchedule(**variant['tau_schedule_kwargs'])
    else:
        tau_schedule = None
    algo_kwargs['tdm_kwargs']['epoch_max_tau_schedule'] = tau_schedule

    algo_kwargs['tdm_kwargs']['observation_key'] = observation_key
    algo_kwargs['tdm_kwargs']['desired_goal_key'] = desired_goal_key

    algorithm = TdmTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        qf1.cuda()
        qf2.cuda()
        policy.cuda()
        algorithm.cuda()
    algorithm.train()

def tdm_twin_sac_experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    observation_key = variant.get('observation_key', 'observation')
    desired_goal_key = variant.get('desired_goal_key', 'desired_goal')
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['replay_buffer_kwargs']
    )
    obs_dim = env.observation_space.spaces['observation'].low.size
    action_dim = env.action_space.low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    vf = FlattenMlp(
        input_size=obs_dim + goal_dim,
        output_size=1,
        **variant['vf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    algorithm = HerTwinSac(
        env,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        policy=policy,
        replay_buffer=replay_buffer,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        qf1.cuda()
        qf2.cuda()
        vf.cuda()
        policy.cuda()
        algorithm.cuda()
    algorithm.train()

def tdm_sac_experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    observation_key = variant.get('observation_key', 'observation')
    desired_goal_key = variant.get('desired_goal_key', 'desired_goal')
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['replay_buffer_kwargs']
    )
    obs_dim = env.observation_space.spaces['observation'].low.size
    action_dim = env.action_space.low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    qf = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    vf = FlattenMlp(
        input_size=obs_dim + goal_dim,
        output_size=1,
        **variant['vf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    algorithm = HerSac(
        env,
        qf=qf,
        vf=vf,
        policy=policy,
        replay_buffer=replay_buffer,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        qf.cuda()
        vf.cuda()
        policy.cuda()
        algorithm.cuda()
    algorithm.train()