from rlkit.launchers.launcher_util import run_experiment
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.experiments.ashvin.awr_rl import state_td3bc_experiment

from rlkit.launchers.arglauncher import run_variants

if __name__ == "__main__":
    variant = dict(
        env_id='pen-v0',
        algo_kwargs=dict(
            batch_size=1024,
            num_epochs=1000,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=10000,
            max_path_length=200,
        ),
        td3_trainer_kwargs=dict(
            discount=0.99,
        ),
        td3_bc_trainer_kwargs=dict(
            discount=0.99,
            demo_path=["demos/icml2020/hand/pen.npy"],
            demo_off_policy_path=None,
            bc_num_pretrain_steps=20000,
            q_num_pretrain_steps=10000,
            rl_weight=1.0,
            bc_weight=0,
            reward_scale=1.0,
            target_update_period=2,
            policy_update_period=2,
            obs_key='state_observation',
            # env_info_key='puck_distance',
            max_path_length=200,
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1e6),
            # fraction_goals_rollout_goals=0.2,
            # fraction_goals_env_goals=0.5,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        save_video=True,
        load_demos=True,
        dump_video_kwargs=dict(
            # imsize=(3, 84, 84),
            imwidth=84,
            imheight=84,
            num_imgs=1,
            dump_pickle=True,
            exploration_goal_image_key="image_observation",
            evaluation_goal_image_key="image_observation",
            rows=1,
            columns=5,
            # unnormalize=False,
            save_video_period=1,
        ),
        image_env_kwargs=dict(
            imsize=84,
            init_camera=None, # the environment initializes the camera already
            transpose=True,
            normalize=True,
            recompute_reward=False,
        ),
        exploration_noise=.1,
        td3_bc=True,

        num_exps_per_instance=1,
        region='us-west-2',

        logger_variant=dict(
            tensorboard=True,
        ),
    )

    search_space = {
        'td3_bc_trainer_kwargs.use_awr': [False],
        # 'td3_bc_trainer_kwargs.demo_beta':[1, 10],
        'td3_bc_trainer_kwargs.bc_weight': [1],
        'td3_bc_trainer_kwargs.rl_weight': [0],
        'algo_kwargs.num_epochs': [100],
        'algo_kwargs.num_eval_steps_per_epoch': [1000],
        'algo_kwargs.num_expl_steps_per_train_loop': [1000],
        'algo_kwargs.min_num_steps_before_training': [0],
        # 'td3_bc_trainer_kwargs.add_demos_to_replay_buffer':[True, False],
        # 'td3_bc_trainer_kwargs.num_trains_per_train_loop':[1000, 2000, 4000, 10000, 16000],
        # 'exploration_noise':[0.1, .3, .5],
        # 'pretrain_rl':[True],
        # 'pretrain_policy':[False],
        'pretrain_rl': [False],
        'pretrain_policy': [True],
        'seedid': range(5),
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(state_td3bc_experiment, variants, run_id=0)
