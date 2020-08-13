"""
AWR + SAC from demo experiment
"""

from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.launchers.experiments.awac.awac_rl import experiment, process_args

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from rlkit.torch.sac.policies import GaussianPolicy
from rlkit.torch.networks import Clamp
from rlkit.torch.sac.bcq_trainer import BCQTrainer, BCQPolicy

if __name__ == "__main__":
    variant = dict(
        num_epochs=11,
        num_eval_steps_per_epoch=1000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=1000,
        batch_size=1024,
        replay_buffer_size=int(1E6),

        policy_class=BCQPolicy,
        policy_kwargs=dict(
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256, ],
            output_activation=Clamp(max=0), # rewards are <= 0
        ),

        algorithm="SAC",
        version="normal",
        collection_mode='batch',
        trainer_class=BCQTrainer,
        trainer_kwargs=dict(

        ),
        launcher_config=dict(
            num_exps_per_instance=1,
            region='us-west-2',
        ),

        path_loader_class=DictToMDPPathLoader,
        path_loader_kwargs=dict(
            obs_key="state_observation",
            demo_paths=[  # these can be loaded in awac_rl.py per env
                # dict(
                #     path="demos/icml2020/hand/pen_bc5.npy",
                #     obs_dict=False,
                #     is_demo=False,
                #     train_split=0.9,
                # ),
            ],
        ),
        add_env_demos=True,
        add_env_offpolicy_data=True,
        normalize_env=False,

        load_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,
    )

    search_space = {
        'env_id': ["pen-binary-v0", "door-binary-v0", "relocate-binary-v0", ],
        'seedid': range(3),
        'trainer_kwargs.beta': [0.5, ],
        'trainer_kwargs.clip_score': [0.5, ],
        'trainer_kwargs.awr_use_mle_for_vf': [True, ],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, process_args)
