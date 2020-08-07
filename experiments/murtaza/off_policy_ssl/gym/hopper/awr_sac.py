import rlkit.misc.hyperparameter as hyp
from rlkit.torch.sac.policies import GaussianPolicy
from rlkit.launchers.experiments.awac.awac_rl import experiment
from rlkit.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    varihopper = dict(
        num_epochs=500,
        num_eval_steps_per_epoch=3000,
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
            q_num_pretrain2_steps=500000,
            policy_weight_decay=1e-4,
            weight_loss=True,
            bc_num_pretrain_steps=100000,
            terminal_transform_kwargs=dict(m=0, b=0),
            pretraining_env_logging_period=100000,
            do_pretrain_rollouts=True,
        ),
        policy_kwargs=dict(
            hidden_sizes=[256]*4,
            max_log_std=0,
            min_log_std=-6,
            std_architecture="shared",
        ),
        path_loader_kwargs=dict(
        ),
        weight_update_period=10000,
    )

    search_space = {
        'use_weights':[True],
        'policy_kwargs.hidden_sizes':[[256]*4],
        'trainer_kwargs.use_automatic_entropy_tuning':[False],
        'trainer_kwargs.alpha':[0],
        'trainer_kwargs.weight_loss':[True],
        'path_loader_kwargs.demo_off_policy_path':[
            # 'demos/hopper_off_policy_10_demos_100.npy',
            'demos/hopper_off_policy_15_demos_100.npy',
            # 'demos/hopper_off_policy_25_demos_100.npy',
        ],
        'path_loader_kwargs.demo_path': [
            # 'demos/hopper_action_noise_10.npy',
            'demos/hopper_action_noise_15.npy',
            # 'demos/hopper_action_noise_25.npy',
        ],
        'trainer_kwargs.beta':[
            .01,
            .1,
            1,
            10,
            100,
        ],
        'train_rl':[True],
        'pretrain_rl':[True],
        'load_demos':[True],
        'pretrain_policy':[False],
        'env': [
            'hopper',
        ],
        'policy_class':[
          GaussianPolicy,
        ],
        'trainer_kwargs.awr_loss_type':[
            'mle'
        ],
        'trainer_kwargs.reparam_weight': [0.0],
        'trainer_kwargs.awr_weight': [1.0],
        'trainer_kwargs.bc_weight': [1.0, ],
        # 'policy_kwargs.std_architecture': ["values", "shared"],
        'trainer_kwargs.compute_bc': [True, ],
        'trainer_kwargs.awr_use_mle_for_vf': [True, ],
        'trainer_kwargs.awr_sample_actions': [False, ],
        'trainer_kwargs.awr_min_q': [True, ],
        'trainer_kwargs.q_weight_decay': [0],
        # 'trainer_kwargs.terminal_transform_kwargs': [dict(m=0, b=0), None],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=varihopper,
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_name = 'awr_sac_offline_hopper_v3'
    

    n_seeds = 4
    mode = 'ec2'
    exp_name = 'awr_sac_hopper_offline_online_v1'

    for exp_id, varihopper in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_name=exp_name,
                mode=mode,
                varihopper=varihopper,
                num_exps_per_instance=1,
                # use_gpu=True,
                gcp_kwargs=dict(
                    preemptible=False,
                ),
                # skip_wait=True,
            )
