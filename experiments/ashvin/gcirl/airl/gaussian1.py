from rlkit.launchers.arglauncher import run_variants
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.experiments.goal_distribution.irl_launcher import (
    irl_experiment,
    process_args
)
from rlkit.launchers.launcher_util import run_experiment

from multiworld.envs.pygame import PickAndPlaceEnv
from rlkit.torch.irl.irl_trainer import MahalanobisReward

if __name__ == '__main__':
    imsize = 200
    variant = dict(
        algo_kwargs=dict(
            num_epochs=1001,
            batch_size=128,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000, #4000,
            min_num_steps_before_training=1000,
            eval_epoch_freq=1,
        ),
        max_path_length=100,
        trainer_kwargs=dict(
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
            reward_scale=100,
            discount=0.99,
        ),
        contextual_replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_future_context=0.0,
            fraction_distribution_context=0.0,
            fraction_replay_buffer_context=0.0,
            # recompute_rewards=True,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        observation_key='observation',
        context_keys=[],
        save_env_in_snapshot=False,
        save_video=True,
        dump_video_kwargs=dict(
            rows=1,
            columns=8,
            pad_color=0,
            pad_length=0,
            subpad_length=1,
        ),
        save_video_period=50,
        renderer_kwargs=dict(
            width=imsize,
            height=imsize,
        ),
        env_id="HalfCheetah-v2",
        logger_config=dict(
            snapshot_gap=50,
            snapshot_mode='gap_and_last',
        ),
        launcher_config=dict(
            unpack_variant=True,
        ),
        reward_trainer_kwargs=dict(
            data_split=0.5,
            train_split=0.5,
        ),
        add_env_demos=True,
        path_loader_kwargs=dict(
            do_preprocess=False,
        ),
        score_fn_class=MahalanobisReward,
        score_fn_kwargs=dict(
        ),
    )

    search_space = {
        'seedid': range(3),
        'env_id': [
            'HalfCheetah-v2', 'Ant-v2',
            'Walker2d-v2',
            "pen-binary-v0",
            "door-binary-v0",
            "relocate-binary-v0",
        ],
        'trainer_kwargs.reward_scale': [1, 10, 100],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(irl_experiment, variants, process_args, run_id=0)
