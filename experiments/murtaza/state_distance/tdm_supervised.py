from rlkit.data_management.her_replay_buffer import HerReplayBuffer
# from rlkit.data_management.tau_replay_buffer import TauReplayBuffer
from rlkit.data_management.tau_replay_buffer import TauReplayBuffer
from rlkit.envs.multitask.point2d import MultitaskPoint2DEnv
from rlkit.envs.multitask.reacher_7dof import Reacher7DofXyzGoalState, Reacher7DofMultitaskEnv, Reacher7DofFullGoal
from rlkit.envs.multitask.reacher_env import MultitaskReacherEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.state_distance.tdm_supervised import TdmSupervised
import rlkit.torch.pytorch_util as ptu
from rlkit.state_distance.tdm_networks import TdmPolicy
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment
import numpy as np

def experiment(variant):
    env = NormalizedBoxEnv(MultitaskPoint2DEnv()) #try full state reacher
    # env = Reacher7DofMultitaskEnv()
    es = OUStrategy(action_space=env.action_space)
    policy = TdmPolicy(
        env=env,
        **variant['policy_kwargs']
    )
    replay_buffer_size = variant['algo_params']['base_kwargs']['replay_buffer_size']
    replay_buffer = TauReplayBuffer(replay_buffer_size, env)
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = TdmSupervised(
        env,
        replay_buffer=replay_buffer,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        policy_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        algo_params=dict(
            base_kwargs=dict(
                num_epochs=100,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=100,
                num_updates_per_env_step=1,
                batch_size=128,
                replay_buffer_size=1000000,
                render=True,
            ),
            tdm_kwargs=dict(
                max_tau=10,
                norm_order=1,
                square_distance=False,
            ),
        ),
    )
    search_space = {
        'algo_params.tdm_kwargs.max_tau': [
            0,
            # 5,
            # 7,
            # 10,
            # 10,
            # 15,
            # 20,
        ],
        'algo_params.policy_criterion':[
            'MSE',
            # 'Huber',
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
            exp_prefix='supervised_point_mass',
            mode='local',
        )
