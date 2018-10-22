from railrl.data_management.tau_replay_buffer import TauReplayBuffer
from railrl.envs.mujoco.sawyer_gripper_env import SawyerXYEnv
from railrl.envs.multitask.point2d import MultitaskPoint2DEnv
from railrl.envs.multitask.pusher2d import FullPusher2DEnv
from railrl.envs.multitask.reacher_7dof import Reacher7DofMultitaskEnv, Reacher7DofFullGoal
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
from railrl.exploration_strategies.ou_strategy import OUStrategy
import railrl.torch.pytorch_util as ptu
from railrl.state_distance.tdm_networks import TdmPolicy
import railrl.misc.hyperparameter as hyp
from railrl.launchers.launcher_util import run_experiment
import numpy as np

from railrl.torch.inverse_model.inverse_model import Inverse_Model
from railrl.torch.networks import TanhMlpPolicy


def experiment(variant):
    # env = NormalizedBoxEnv(MultitaskPoint2DEnv())
    # env = Reacher7DofFullGoal()
    env = FullPusher2DEnv(include_puck=False, arm_range=.5)
    env = SawyerXYEnv()
    prob_random_action = variant['prob_random_action']
    # es = OUStrategy(action_space=env.action_space)
    es = EpsilonGreedy(action_space=env.action_space, prob_random_action=.2)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    policy = TanhMlpPolicy(
        input_size=obs_dim+env.goal_dim,
        output_size=action_dim,
        hidden_sizes=variant['policy_kwargs']['hidden_sizes']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer_size = variant['algo_params']['base_kwargs']['replay_buffer_size']
    replay_buffer = TauReplayBuffer(replay_buffer_size, env)
    algorithm = Inverse_Model(
        env,
        policy=policy,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        policy_kwargs=dict(
            hidden_sizes=[500, 400, 300],
        ),
        algo_params=dict(
            base_kwargs=dict(
                num_epochs=10,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=100,
                num_updates_per_env_step=15,
                batch_size=64,
                replay_buffer_size=10000,
                render=True,
            ),
        ),
    )
    search_space = {
        'algo_params.policy_criterion':[
            'MSE',
        ],
        'algo_params.base_kwargs.num_updates_per_env_step':[
            1,
            # 5,
            # 10,
            # 15,
        ],
        'algo_params.time_horizon':[
            # 0,
            5,
            10,
            15,
            None,
        ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        run_experiment(
            experiment,
            seed=np.random.randint(1, 10004),
            variant=variant,
            exp_id=exp_id,
            exp_prefix='test',
            mode='local',
        )

#things to try:
'''
reduce the horizon - 10/15
velocity control? 
sawyer - reacher3d 
reacher2d 
pull from goal conditioned images, test on pusher2d - fullpusher2denv
'''