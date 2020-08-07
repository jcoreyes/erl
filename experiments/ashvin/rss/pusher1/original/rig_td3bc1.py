import rlkit.misc.hyperparameter as hyp
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.grill.launcher import *
from rlkit.launchers.arglauncher import run_variants

from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj import SawyerMultiobjectEnv

from rlkit.misc.asset_loader import load_local_or_remote_file

import random
import numpy as np

# from torch import nn

def grill_her_td3_full_experiment(variant):
    full_experiment_variant_preprocess(variant)
    if not variant['grill_variant'].get('do_state_exp', False):
        train_vae_and_update_variant(variant)
    grill_her_td3_experiment(variant['grill_variant'])

def generate_vae_dataset_from_demos(variant):
    demo_path = variant["demo_path"]
    test_p = variant.get('test_p', 0.9)
    use_cached = variant.get('use_cached', True)
    imsize = variant.get('imsize', 84)
    num_channels = variant.get('num_channels', 3)
    show = variant.get('show', False)
    init_camera = variant.get('init_camera', None)

    def load_paths(paths):
        data = [load_path(path) for path in paths]
        data = np.concatenate(data, 0)
        return data

    def load_path(path):
        N = len(path["observations"])
        data = np.zeros((N, imsize * imsize * num_channels), dtype=np.uint8)
        i = 0
        for (
            ob,
            action,
            reward,
            next_ob,
            terminal,
            agent_info,
            env_info,
        ) in zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        ):
            img = ob["image_observation"]
            img = img.reshape(imsize, imsize, 3).transpose()
            data[i, :] = img.flatten()
            i += 1
        return data

    data = load_local_or_remote_file(demo_path)
    random.shuffle(data)
    N = int(len(data) * test_p)
    print("using", N, "paths for training")

    train_data = load_paths(data[:N])
    test_data = load_paths(data[N:])

    print("training data shape", train_data.shape)
    print("test data shape", test_data.shape)

    info = {}

    return train_data, test_data, info

def train_vae_and_update_variant(variant):
    from rlkit.core import logger
    grill_variant = variant['grill_variant']
    train_vae_variant = variant['train_vae_variant']
    if grill_variant.get('vae_path', None) is None:
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'vae_progress.csv', relative_to_snapshot_dir=True
        )
        vae, vae_train_data, vae_test_data = train_vae(train_vae_variant,
                                                       return_data=True)
        if grill_variant.get('save_vae_data', False):
            grill_variant['vae_train_data'] = vae_train_data
            grill_variant['vae_test_data'] = vae_test_data
        logger.save_extra_data(vae, 'vae.pkl', mode='pickle')
        logger.remove_tabular_output(
            'vae_progress.csv',
            relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'progress.csv',
            relative_to_snapshot_dir=True,
        )
        grill_variant['vae_path'] = vae  # just pass the VAE directly
    else:
        if grill_variant.get('save_vae_data', False):
            vae_train_data, vae_test_data, info = generate_vae_dataset(
                train_vae_variant['generate_vae_dataset_kwargs']
            )
            grill_variant['vae_train_data'] = vae_train_data
            grill_variant['vae_test_data'] = vae_test_data


def train_vae(variant, return_data=False):
    from rlkit.misc.ml_util import PiecewiseLinearSchedule
    from rlkit.torch.vae.conv_vae import (
        ConvVAE,
        SpatialAutoEncoder,
        AutoEncoder,
    )
    import rlkit.torch.vae.conv_vae as conv_vae
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    from rlkit.pythonplusplus import identity
    import torch
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    train_data, test_data, info = generate_vae_dataset_from_demos(
        variant['generate_vae_dataset_kwargs']
    )
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    if 'beta_schedule_kwargs' in variant:
        beta_schedule = PiecewiseLinearSchedule(
            **variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None
    if variant.get('decoder_activation', None) == 'sigmoid':
        decoder_activation = torch.nn.Sigmoid()
    else:
        decoder_activation = identity
    architecture = variant['vae_kwargs'].get('architecture', None)
    if not architecture and variant.get('imsize') == 84:
        architecture = conv_vae.imsize84_default_architecture
    elif not architecture and variant.get('imsize') == 48:
        architecture = conv_vae.imsize48_default_architecture
    variant['vae_kwargs']['architecture'] = architecture
    variant['vae_kwargs']['imsize'] = variant.get('imsize')

    if variant['algo_kwargs'].get('is_auto_encoder', False):
        m = AutoEncoder(representation_size, decoder_output_activation=decoder_activation,**variant['vae_kwargs'])
    elif variant.get('use_spatial_auto_encoder', False):
        raise NotImplementedError('This is currently broken, please update SpatialAutoEncoder then remove this line')
        m = SpatialAutoEncoder(representation_size, int(representation_size / 2))
    else:
        vae_class = variant.get('vae_class', ConvVAE)
        m = vae_class(representation_size, decoder_output_activation=decoder_activation,**variant['vae_kwargs'])
    m.to(ptu.device)
    t = ConvVAETrainer(train_data, test_data, m, beta=beta,
                       beta_schedule=beta_schedule, **variant['algo_kwargs'])
    save_period = variant['save_period']
    dump_skew_debug_plots = variant.get('dump_skew_debug_plots', False)
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % save_period == 0)
        t.train_epoch(epoch)
        t.test_epoch(
            epoch,
            save_reconstruction=should_save_imgs,
            save_scatterplot=should_save_imgs,
            # save_vae=False,
        )
        if should_save_imgs:
            t.dump_samples(epoch)
            if dump_skew_debug_plots:
                t.dump_best_reconstruction(epoch)
                t.dump_worst_reconstruction(epoch)
                t.dump_sampling_histogram(epoch)
        t.update_train_weights()
    logger.save_extra_data(m, 'vae.pkl', mode='pickle')
    if return_data:
        return m, train_data, test_data
    return m

def grill_her_td3_experiment(variant):
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from rlkit.demos.her_td3bc import HerTD3BC
    from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
    grill_preprocess_variant(variant)
    env = get_envs(variant)
    es = get_exploration_strategy(variant, env)

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    demo_train_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    demo_test_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['replay_buffer'] = replay_buffer
    base_kwargs = algo_kwargs['base_kwargs']
    base_kwargs['training_env'] = env
    base_kwargs['render'] = variant["render"]
    base_kwargs['render_during_eval'] = variant["render"]
    her_kwargs = algo_kwargs['her_kwargs']
    her_kwargs['observation_key'] = observation_key
    her_kwargs['desired_goal_key'] = desired_goal_key
    # algorithm = HerTd3(
    #     env,
    #     qf1=qf1,
    #     qf2=qf2,
    #     policy=policy,
    #     exploration_policy=exploration_policy,
    #     **variant['algo_kwargs']
    # )
    env.vae.to(ptu.device)

    algorithm = HerTD3BC(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        demo_train_buffer=demo_train_buffer,
        demo_test_buffer=demo_test_buffer,
        demo_path=variant["demo_path"],
        add_demo_latents=True,
        **variant['algo_kwargs']
    )

    if variant.get("save_video", True):
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            algorithm.eval_policy,
            variant,
        )
        algorithm.post_epoch_funcs.append(video_func)

    algorithm.to(ptu.device)
    if not variant.get("do_state_exp", False):
        env.vae.to(ptu.device)

    algorithm.train()

if __name__ == "__main__":

    x_low = -0.2
    x_high = 0.2
    y_low = 0.5
    y_high = 0.7
    t = 0.03

    variant = dict(
        imsize=84,
        init_camera=sawyer_pusher_camera_upright_v2,
        grill_variant=dict(
            save_video=True,
            save_video_period=100,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algo_kwargs=dict(
                base_kwargs=dict(
                    num_epochs=505,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=4000,
                    batch_size=128,
                    max_path_length=100,
                    discount=0.99,
                    num_updates_per_env_step=4,
                    collection_mode='online-parallel',
                    parallel_env_params=dict(
                        num_workers=1,
                    ),
                    reward_scale=1,
                ),
                her_kwargs=dict(),
                td3_kwargs=dict(
                    tau=1e-2,
                ),
            ),
            replay_buffer_kwargs=dict(
                max_size=int(1e6),
                fraction_goals_rollout_goals=0.1,
                fraction_goals_env_goals=0.5,
            ),
            algorithm='RIG-HER-TD3',
            normalize=False,
            render=False,
            exploration_noise=0.2,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            ),
            demo_path="demos/pusher_demos_100b.npy",
        ),
        train_vae_variant=dict(
            representation_size=4,
            beta=10.0 / 128,
            num_epochs=501,
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            generate_vae_dataset_kwargs=dict(
                demo_path="demos/pusher_demos_100b.npy",
                test_p=.9,
                N=10000,
                oracle_dataset_using_set_to_goal=False,
                random_rollout_data=True,
                use_cached=True,
                vae_dataset_specific_kwargs=dict(
                ),
                show=False,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                use_linear_dynamics=False,
                is_auto_encoder=False,
                batch_size=128,
                lr=1e-3,
            ),
            save_period=5,
        ),

        env_class=SawyerMultiobjectEnv,
        env_kwargs=dict(
            num_objects=1,
            preload_obj_dict=[
                dict(color2=(0.1, 0.1, 0.9)),
            ],
        ),
        demo_path="demos/pusher_demos_100b.npy",

        region="us-west-2",
    )

    search_space = {
        'seedid': range(5),
        'grill_variant.algo_kwargs.base_kwargs.num_updates_per_env_step': [4, 16, ],
        'grill_variant.replay_buffer_kwargs.fraction_goals_rollout_goals': [0.1, ],
        'grill_variant.replay_buffer_kwargs.fraction_goals_env_goals': [0.5, ],
        'grill_variant.algo_kwargs.base_kwargs.bc_weight': [1.0, 0],
        'grill_variant.algo_kwargs.base_kwargs.rl_weight': [1.0, 0],
        'grill_variant.algo_kwargs.td3_kwargs.weight_decay': [1e-4, 0],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        x = variant["grill_variant"]["algo_kwargs"]["base_kwargs"]["bc_weight"]
        y = variant["grill_variant"]["algo_kwargs"]["base_kwargs"]["rl_weight"]
        if x != 0 or y != 0:
            variants.append(variant)

    run_variants(grill_her_td3_full_experiment, variants, run_id=3)
