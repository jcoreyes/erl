from railrl.launchers.launcher_util import run_experiment
import railrl.misc.hyperparameter as hyp
from railrl.launchers.experiments.murtaza.rfeatures_rl import state_td3bc_experiment

if __name__ == "__main__":
    # demo_path = ["/home/anair/ros_ws/src/railrl-private/demos/door_demos_v3/processed_demos_%s_jitter2.pkl" % color for color in ["grey", "beige", "green", "brownhatch"]]
    # demo_off_policy_path = ["/home/anair/data/s3doodad/ashvin/rfeatures/sawyer/door2/bc-v3-varied1/run%s/id0/video_0_env.p" % str(i) for i in [0, 1]]
    # print(demo_off_policy_path)
    demo_path = None
    demo_off_policy_path = None
    variant = dict(
        env_id='SawyerDoorHookResetFreeEnv-v1',
        algo_kwargs=dict(
            num_epochs=200,
            max_path_length=100,
            batch_size=128,
            num_eval_steps_per_epoch=500,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=10000,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            demo_path=demo_path,
            demo_off_policy_path=demo_off_policy_path,
            add_demo_latents=False, # already done
            bc_num_pretrain_steps=10000,
            q_num_pretrain_steps=10000,
            rl_weight=1.0,
            bc_weight=0,
            # reward_scale=0.0001,
            # weight_decay=0.001,
        ),
        replay_buffer_kwargs=dict(
            max_size=1000000,
            fraction_goals_rollout_goals=0.2,
            fraction_goals_env_goals=0.5,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        save_video=False,
        exploration_noise=.2,
    )

    search_space = {
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    # n_seeds = 3
    # mode = 'gcp'
    # exp_prefix = 'skew-fit-door-reference-post-refactor'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                state_td3bc_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=3,
                gcp_kwargs=dict(
                    zone='us-east1-c',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-k80',
                        num_gpu=1,
                    )
                ),
            )
