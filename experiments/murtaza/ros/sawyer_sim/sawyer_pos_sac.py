from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import ConcatMlp
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from sawyer_control.sawyer_reaching import SawyerXYZReachingEnv
import numpy as np
import rlkit.misc.hyperparameter as hyp

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
    num_steps_per_epoch=100
    num_steps_per_eval=100
    max_path_length=20
    variant = dict(
        algo_params=dict(
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch,
            num_steps_per_eval=num_steps_per_eval,
            max_path_length=max_path_length,
            batch_size=64,
            discount=1,
            soft_target_tau=0.01,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
        ),
        net_size=300,
        env_params=dict(
            desired=[0.97711039, 0.56662792, 0.27901027],
            action_mode='position',
            reward='norm',
            safety_box=False,
        )
    )
    search_space = {
        'algo_params.reward_scale': [
            1,
            10,
            100,
            1000,
        ],
        'algo_params.num_updates_per_env_step': [
            5,
            10,
            15,
            20,
            35,
        ],
        'algo_params.soft_target_tau': [
            .01,
            .001,
        ],
        'env_params.randomize_goal_on_reset':[
            True,
            False,
        ],
        'net_size':[
            200,
            300,
            400,
        ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    n_seeds = 1
    for variant in sweeper.iterate_hyperparameters():
        exp_prefix = 'sawyer_simulated_sac_reaching_pos_cntrl'
        mode = 'here_no_doodad'
        for i in range(n_seeds):
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                variant=variant,
            )
