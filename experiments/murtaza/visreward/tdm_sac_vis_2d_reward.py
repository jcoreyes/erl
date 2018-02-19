import random

import gym
import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.launchers.launcher_util import run_experiment
from railrl.sac.policies import TanhGaussianPolicy
from railrl.sac.sac import SoftActorCritic
from railrl.torch.networks import FlattenMlp
from railrl.state_distance.tdm_sac import TdmSac
import sys
sys.path.append('/home/murtaza/Documents/objectattention/')
from singleobj_visreward import SingleObjVisRewardEnv

env = SingleObjVisRewardEnv()

def experiment(variant):
    obs_dim = int(np.prod(env.observation_space.low.shape))
    action_dim = int(np.prod(env.action_space.low.shape))
    vectorized = variant['algo_params']['tdm_kwargs']['vectorized']
    qf_class = variant['qf_class']
    vf_class = variant['vf_class']
    policy_class = variant['policy_class']
    qf = qf_class(
        observation_dim=obs_dim,
        action_dim=action_dim,
        goal_dim=env.goal_dim,
        output_size=env.goal_dim if vectorized else 1,
        **variant['qf_params']
    )
    vf = vf_class(
        observation_dim=obs_dim,
        goal_dim=env.goal_dim,
        output_size=env.goal_dim if vectorized else 1,
        **variant['qf_params']
    )
    policy = policy_class(
        obs_dim=obs_dim,
        action_dim=action_dim,
        goal_dim=env.goal_dim,
        **variant['policy_params']
    )
    replay_buffer = HerReplayBuffer(
        env=env,
        **variant['her_replay_buffer_params']
    )
    algorithm = TdmSac(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        replay_buffer=replay_buffer,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker

    num_epochs = 200
    num_steps_per_epoch = 1000
    num_steps_per_eval = 1000
    batch_size = 512
    max_path_length = 100
    max_tau = max_path_length-1
    use_gpu = True
    variant = dict(
        algo_params = dict(
            base_kwargs=dict(
                num_epochs=num_epochs,
                num_steps_per_epoch=num_steps_per_epoch,
                num_steps_per_eval=num_steps_per_eval,
                max_path_length=max_path_length,
                num_updates_per_env_step=10,
                batch_size=batch_size,
                discount=1,
            ),
            tdm_kwargs=dict(
                sample_rollout_goals_from='environment',
                sample_train_goals_from='her',
                vectorized=True,
                cycle_taus_for_rollout=True,
                max_tau=10,
            ),
            sac_kwargs=dict(
                soft_target_tau=0,
                policy_lr=3E-4,
                qf_lr=3E-4,
                vf_lr=3E-4,
                qf_target_update_interval=5,
            ),
        ),
        her_replay_buffer_params = dict(
            max_size=int(2E5),
            num_goals_to_sample=4,
        ),
        qf_params = dict(
            max_tau=max_tau,
            hidden_sizes=[100, 100],
        ),
        vf_params = dict(
            max_tau=max_tau,
            hidden_sizes=[100, 100],
        ),
        policy_params = dict(
            max_tau=max_tau,
            hidden_sizes=[100, 100],
        ),
    )
    search_space = {
        'algo_params.reward_scale': [
            10,
            100,
        ],
        'algo_params.num_updates_per_env_step': [
            10,
            # 15,
            20,
            # 25,
        ],
        'algo_params.batch_size': [
            512,
            1024,
        ]

    }
    exp_prefix = 'TEST'
    mode='local'
    run_experiment(
        experiment,
        seed = random.randint(0, 10000),

        variant=variant,
        exp_prefix=exp_prefix,
        mode=mode,
    )

