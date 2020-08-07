import gym

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_and_epislon import \
    GaussianAndEpislonStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import GoalConditionedPathCollector
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
# from rlkit.torch.td3.td3 import TD3
from rlkit.demos.td3_bc import TD3BCTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
# from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_subset import SawyerMultiobjectEnv
# from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachXYZEnv

from multiworld.core.image_env import ImageEnv
from multiworld.envs.real_world.sawyer.sawyer_reaching import SawyerReachXYZEnv
# from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv

from rlkit.launchers.arglauncher import run_variants
import rlkit.misc.hyperparameter as hyp

# from rlkit.launchers.experiments.ashvin.rfeatures.rfeatures_trainer import TimePredictionTrainer
from rlkit.launchers.experiments.ashvin.rfeatures.rfeatures_rl import encoder_wrapped_td3bc_experiment

if __name__ == "__main__":
    colors = ["grey", "beige", "green", "brownhatch"]
    # colors = ["grey"]
    # demo_path = ["/home/anair/ros_ws/src/rlkit-private/demos/door_demos_v3/processed_demos_%s_latent_distance__use_goal_from_trajectory_jitter2.pkl" % color for color in colors]
    demo_path = ["/home/anair/ros_ws/src/railrl-private/demos/door_demos_v3/processed_demos_%s_latent_distance_use_initial_use_initial_from_trajectory_use_goal_from_trajectory_jitter2.pkl" % color for color in colors]
    variant = dict(
        env_class=SawyerReachXYZEnv,
        env_kwargs=dict(
            action_mode="position",
            max_speed = 0.05,
            camera="sawyer_head"
        ),
        # algo_kwargs=dict(
        #     num_epochs=3000,
        #     max_path_length=20,
        #     batch_size=128,
        #     num_eval_steps_per_epoch=1000,
        #     num_expl_steps_per_train_loop=1000,
        #     num_trains_per_train_loop=1000,
        #     min_num_steps_before_training=1000,
        # ),
        algo_kwargs=dict(
            num_epochs=500,
            max_path_length=100,
            batch_size=128,
            num_eval_steps_per_epoch=500,
            num_expl_steps_per_train_loop=500,
            num_trains_per_train_loop=500,
            min_num_steps_before_training=0,
        ),
        # config_params = dict(
        #     initial_type="",
        #     goal_type="",
        #     use_initial=False
        # ),
        config_params = dict(
            initial_type="use_initial_from_trajectory",
            goal_type="",
            use_initial=True
        ),
        reward_params_type="latent_distance",
        model_kwargs=dict(
            decoder_distribution='gaussian_identity_variance',
            input_channels=3,
            imsize=224,
            architecture=dict(
                hidden_sizes=[200, 200],
            ),
            delta_features=True,
            pretrained_features=False,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            demo_path=demo_path,
            # demo_path="/home/anair/ros_ws/src/rlkit-private/demos/door_demos_10_2/processed_demos_imagenet.pkl",
            add_demo_latents=False, # already done
            bc_num_pretrain_steps=10000,
            rl_weight=0.0,
            bc_weight=1.0,
            weight_decay=0.001,
        ),
        replay_buffer_kwargs=dict(
            max_size=1000000,
            fraction_goals_rollout_goals=1.0,
            fraction_goals_env_goals=0.0,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[128, 128],
        ),

        save_video=True,
        save_period=1,
        dump_video_kwargs=dict(
            imwidth=500,
            imheight=300,
            num_imgs=1,
            dump_pickle=True,
            exploration_goal_image_key="image_observation",
            evaluation_goal_image_key="image_observation",
            rows=1,
            columns=1,
        ),
        desired_trajectory="/home/anair/ros_ws/src/railrl-private/demos/door_demos_v3/demo_v3_grey_0.pkl",

        logger_variant=dict(
            tensorboard=True,
        ),
        model_path="/home/anair/data/s3doodad/facebook/models/rfeatures/multitask1/run2/id2/itr_4000.pt",
    )

    search_space = {
        'seedid': range(1),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(encoder_wrapped_td3bc_experiment, variants, run_id=29)
