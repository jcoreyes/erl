from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import ConcatMlp
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from sawyer_control.sawyer_reaching import SawyerXYZReachingEnv
import numpy as np
import rlkit.misc.hyperparameter as hyp
import ray
ray.init()
def experiment(variant):
    env_params = variant['env_params']
    env = SawyerXYZReachingEnv(**env_params)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']
    qf = ConcatMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = ConcatMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    algorithm = SoftActorCritic(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    num_epochs = 50
    num_steps_per_epoch=1000
    num_steps_per_eval=1000
    max_path_length=100
    variant = dict(
        algo_params=dict(
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch,
            num_steps_per_eval=num_steps_per_eval,
            max_path_length=max_path_length,
            batch_size=64,
            discount=0.99,
            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            normalize_env=False,
            render=False,
            collection_mode='online-parallel'

        ),
        net_size=100,
        env_params=dict(
            action_mode='torque',
            reward='norm',
        )
    )
    search_space = {
        'algo_params.reward_scale': [
            100,
        ],
        'algo_params.num_updates_per_env_step': [
            1,
            4,
        ],
        'algo_params.soft_target_tau': [
            .001,
        ],
        'algo_params.collection_mode':[
            'online-parallel',
            'online',
        ],
        'env_params.randomize_goal_on_reset': [
            False,
        ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    n_seeds = 1
    for variant in sweeper.iterate_hyperparameters():
        exp_prefix = 'sawyer_torque_ddpg_xyz_fixed_ee'
        mode = 'here_no_doodad'
        for i in range(n_seeds):
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                variant=variant,
            )
