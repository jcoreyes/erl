import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.twin_sac import TwinSAC

from ss.remote.doodad.arglauncher import run_variants
from sawyer_control.envs.sawyer_insertion_refined_waterproof_dense_pureRL import SawyerHumanControlEnv

from rlkit.demos.td3_bc import TD3BC

from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy

def experiment(variant):
    env = SawyerHumanControlEnv(action_mode='joint_space_impd', position_action_scale=1, max_speed=0.015)
    # max_speed does not actually do anything, it is now included in the function request_angle_action of sawyer_env_base.

    training_env = env

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']

    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    es = GaussianStrategy(
            action_space=env.action_space,
            **variant['es_kwargs'],
        )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = TD3BC(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        exploration_policy=exploration_policy,
        **variant['algo_params']
    )

    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variants = []
    for seedid in range(1):
        variant = dict(
            algo_params=dict(               #original values from rklit sac example:
                num_epochs=400,             #1000
                num_steps_per_epoch=250,     #250
                num_steps_per_eval=250,      #250
                max_path_length=50,         #1000
                batch_size=128,             #128
                discount=0.98,              #0.99, formula: discount = 1-(1/n), where n = max_path_length

                soft_target_tau=0.001,      #0.001
                policy_lr=3E-4,             #3E-4
                qf_lr=3E-4,                 #3E-4
                vf_lr=3E-4,                 #3E-4

                eval_deterministic=False,   # added (default: True), Considering all possible ways makes more sense when considering inperfect target locations

                # entries from TD3 on Sawyer
                replay_buffer_size=int(1E6),
                save_environment=False,
                # min_num_steps_before_training=500,   # include this for better reproducebility
                save_algorithm = False,       # saves the algorithm in extra_data.pkl - creates error, debug this [Gerrit]
            ),
            net_size=100,            #300
            snapshot_mode = "gap_and_last",  # "gap_and_last", "gap", all"
            snapshot_gap = 1,
            seedid=seedid,
        )
        variants.append(variant)

    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)

    # this will run the experiment directly instead of through launcher/doodad
    # setup_logger('name-of-td3-experiment', variant=variant)
    # experiment(variant)

    # this runs the experiment through the launcher pipeline
    run_variants(experiment, variants, run_id=22)
    # run 8 was very good!
    # run 10 was very very good!
    # run 11 was with action penalty in reward
    # run 12 was without action penalty
    # run 13 was without force_threshold and 2-norm
    # run 14 was with fix of target position and change in controller
    # run 14 did not converge because the +-1mm target noise was too large
    # run 15 is in the making, SAC @ USB with very sparse reward + 1-norm -> too lage target noise +-1mm, slow convergence to generalized policy
    # run 16 was very sparse only and turned out to not train well
    # run 17 was less sparse (+1 all the time when inserted) and showed good training, the action space might be to large, will do another less sparse training
    # run 18 is less sparse (r = +1 when inserted), run 18 was amazing! very good convergence and 100x succes in a row
    # run 19 is less sparse and human controller with P = 0.3 and no insertion amplification
