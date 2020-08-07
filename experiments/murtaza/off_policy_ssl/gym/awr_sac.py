import rlkit.misc.hyperparameter as hyp
from rlkit.torch.sac.policies import GaussianPolicy
from rlkit.launchers.experiments.awac.awac_rl import experiment

from rlkit.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    variant = dict(
        num_epochs=3000,
        num_eval_steps_per_epoch=5000,
        num_trains_per_train_loop=1000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=1000,
        batch_size=1024,
        replay_buffer_size=int(1E6),
        layer_size=256,
        num_layers=2,
        algorithm="SAC AWR",
        version="normal",
        collection_mode='batch',
        sac_bc=True,
        load_demos=True,
        pretrain_rl=True,
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            beta=1,
            use_automatic_entropy_tuning=True,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=5000000,
            policy_weight_decay=1e-4,
            weight_loss=True,
            bc_num_pretrain_steps=100000,
            terminal_transform_kwargs=dict(m=0, b=0),
        ),
        policy_kwargs=dict(
            hidden_sizes=[256]*2,
            max_log_std=0,
            min_log_std=-6,
        ),
        path_loader_kwargs=dict(
            demo_path=None,
            frac_trajs=1,
        ),
        weight_update_period=10000,
    )

    search_space = {
        'use_weights':[True],
        'trainer_kwargs.use_automatic_entropy_tuning':[False],
        'trainer_kwargs.bc_weight':[0],
        'trainer_kwargs.alpha':[0],
        'trainer_kwargs.weight_loss':[True],
        'path_loader_kwargs.frac_trajs':[1],
        'trainer_kwargs.beta':[
            .0001,
            # .001,
            # .01,
            # 1,
            # 10,
            # 100,
            # 1000,
            # 1e4,
            # 1e5,
            # 1e6,
        ],
        'train_rl':[False],
        'pretrain_rl':[False],
        'load_demos':[True],
        'pretrain_policy':[False],
        'env': [
            'half-cheetah',
            # 'ant',
            # 'walker',
            # 'hopper',
        ],
        'policy_class':[
          # TanhGaussianPolicy,
          GaussianPolicy,
        ],
        'trainer_kwargs.bc_loss_type':[
            'mse',
        ],
        'trainer_kwargs.awr_loss_type':[
            # 'mse',
            'mle'
        ]

    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_name = 'awr_sac_offline_online_v1'
    

    # n_seeds = 2
    # mode = 'ec2'
    # exp_name = 'awr_sac_offline_train_longer_v1'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                num_exps_per_instance=1,
                use_gpu=True,
                gcp_kwargs=dict(
                    preemptible=False,
                ),
                # skip_wait=True,
            )
