from functools import partial
import os.path as osp

import numpy as np

import railrl.samplers.rollout_functions as rf
import railrl.torch.pytorch_util as ptu
from railrl.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
    RemapKeyFn,
)
from railrl.envs.contextual import ContextualEnv
from railrl.envs.contextual.goal_conditioned import (
    GoalDictDistributionFromMultitaskEnv,
    ContextualRewardFnFromMultitaskEnv,
    AddImageDistribution,
    AddLatentDistribution,
    PriorDistribution,
)
from railrl.envs.images import Renderer, InsertImageEnv
from railrl.launchers.rl_exp_launcher_util import create_exploration_policy
from railrl.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector
)
from railrl.visualization.video import dump_video, VideoSaveFunction, RIGVideoSaveFunction
from railrl.torch.networks import FlattenMlp
from railrl.torch.sac.policies import MakeDeterministic
from railrl.torch.sac.policies import TanhGaussianPolicy
from railrl.torch.sac.sac import SACTrainer
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from railrl.core import logger
from railrl.envs.encoder_wrappers import EncoderWrappedEnv
from railrl.envs.vae_wrappers import VAEWrappedEnv
import time

from multiworld.core.image_env import ImageEnv, unormalize_image

# from railrl.torch.grill.common import (
#     train_vae,
#     full_experiment_variant_preprocess,
#     train_vae_and_update_variant,
# )


# def train_vae(variant):
#     variant['grill_variant']['save_vae_data'] = True
#     full_experiment_variant_preprocess(variant)
#     train_vae_and_update_variant(variant)


class DeleteOldEnvInfo(object):
    def __call__(self, contexutal_env, info, obs, reward, done):
        return {}


class DistanceRewardFn:
    def __init__(self, observation_key, desired_goal_key):
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key

    def __call__(self, states, actions, next_states, contexts):
        s = next_states[self.observation_key]
        c = contexts[self.desired_goal_key]
        return np.linalg.norm(s - c, axis=1)


def goal_conditioned_sac_experiment(
        max_path_length,
        qf_kwargs,
        sac_trainer_kwargs,
        replay_buffer_kwargs,
        policy_kwargs,
        algo_kwargs,
        train_vae_kwargs,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        achieved_goal_key='state_achieved_goal',
        exploration_policy_kwargs=None,
        evaluation_goal_sampling_mode=None,
        exploration_goal_sampling_mode=None,
        # Video parameters
        save_video=True,
        save_video_kwargs=None,
        renderer_kwargs=None,
        imsize=48,
        **kwargs
):
    print(kwargs)

    if exploration_policy_kwargs is None:
        exploration_policy_kwargs = {}
    if not save_video_kwargs:
        save_video_kwargs = {}
    if not renderer_kwargs:
        renderer_kwargs = {}

    renderer = Renderer(**renderer_kwargs)
    init_camera = renderer.init_camera

    def train_vae(variant, return_data=False):
        from railrl.misc.ml_util import PiecewiseLinearSchedule, ConstantSchedule
        from railrl.torch.vae.conv_vae import (
            ConvVAE,
            SpatialAutoEncoder,
            AutoEncoder,
        )
        import railrl.torch.vae.conv_vae as conv_vae
        from railrl.torch.vae.vae_trainer import ConvVAETrainer
        from railrl.core import logger
        import railrl.torch.pytorch_util as ptu
        from railrl.pythonplusplus import identity
        import torch
        beta = variant["beta"]
        representation_size = variant.get("representation_size", variant.get("latent_sizes", None))
        use_linear_dynamics = variant.get('use_linear_dynamics', False)
        generate_vae_dataset_fctn = variant.get('generate_vae_data_fctn',
                                                generate_vae_dataset)
        variant['generate_vae_dataset_kwargs']['use_linear_dynamics'] = use_linear_dynamics
        variant['generate_vae_dataset_kwargs']['batch_size'] = variant['algo_kwargs']['batch_size']
        train_dataset, test_dataset, info = generate_vae_dataset_fctn(
            variant['generate_vae_dataset_kwargs'])

        if use_linear_dynamics:
            action_dim = train_dataset.data['actions'].shape[2]

        logger.save_extra_data(info)
        logger.get_snapshot_dir()
        if 'beta_schedule_kwargs' in variant:
            beta_schedule = PiecewiseLinearSchedule(
                **variant['beta_schedule_kwargs'])
        else:
            beta_schedule = None
        if 'context_schedule' in variant:
            schedule = variant['context_schedule']
            if type(schedule) is dict:
                context_schedule = PiecewiseLinearSchedule(**schedule)
            else:
                context_schedule = ConstantSchedule(schedule)
            variant['algo_kwargs']['context_schedule'] = context_schedule
        if variant.get('decoder_activation', None) == 'sigmoid':
            decoder_activation = torch.nn.Sigmoid()
        else:
            decoder_activation = identity
        architecture = variant['vae_kwargs'].get('architecture', None)
        if not architecture and imsize == 84:
            architecture = conv_vae.imsize84_default_architecture
        elif not architecture and imsize == 48:
            architecture = conv_vae.imsize48_default_architecture
        variant['vae_kwargs']['architecture'] = architecture
        variant['vae_kwargs']['imsize'] = imsize

        if variant['algo_kwargs'].get('is_auto_encoder', False):
            model = AutoEncoder(representation_size, decoder_output_activation=decoder_activation,**variant['vae_kwargs'])
        elif variant.get('use_spatial_auto_encoder', False):
            model = SpatialAutoEncoder(representation_size, decoder_output_activation=decoder_activation,**variant['vae_kwargs'])
        else:
            vae_class = variant.get('vae_class', ConvVAE)
            if use_linear_dynamics:
                model = vae_class(representation_size, decoder_output_activation=decoder_activation, action_dim=action_dim,**variant['vae_kwargs'])
            else:
                model = vae_class(representation_size, decoder_output_activation=decoder_activation,**variant['vae_kwargs'])
        model.to(ptu.device)

        vae_trainer_class = variant.get('vae_trainer_class', ConvVAETrainer)
        trainer = vae_trainer_class(model, beta=beta,
                           beta_schedule=beta_schedule,
                           **variant['algo_kwargs'])
        save_period = variant['save_period']

        dump_skew_debug_plots = variant.get('dump_skew_debug_plots', False)
        for epoch in range(variant['num_epochs']):
            should_save_imgs = (epoch % save_period == 0)
            trainer.train_epoch(epoch, train_dataset)
            trainer.test_epoch(epoch, test_dataset)

            if should_save_imgs:
                trainer.dump_reconstructions(epoch)
                trainer.dump_samples(epoch)
                if dump_skew_debug_plots:
                    trainer.dump_best_reconstruction(epoch)
                    trainer.dump_worst_reconstruction(epoch)
                    trainer.dump_sampling_histogram(epoch)

            stats = trainer.get_diagnostics()
            for k, v in stats.items():
                logger.record_tabular(k, v)
            logger.dump_tabular()
            trainer.end_epoch(epoch)

            if epoch % 50 == 0:
                logger.save_itr_params(epoch, model)
        logger.save_extra_data(model, 'vae.pkl', mode='pickle')
        if return_data:
            return model, train_dataset, test_dataset
        return model


    def generate_vae_dataset(variant):
        print(variant)
        nonlocal env_kwargs, env_id, env_class, imsize, init_camera
        N = variant.get('N', 10000)
        batch_size = variant.get('batch_size', 128)
        test_p = variant.get('test_p', 0.9)
        use_cached = variant.get('use_cached', True)
        num_channels = variant.get('num_channels', 3)
        show = variant.get('show', False)
        dataset_path = variant.get('dataset_path', None)
        oracle_dataset_using_set_to_goal = variant.get('oracle_dataset_using_set_to_goal', False)
        random_rollout_data = variant.get('random_rollout_data', False)
        random_rollout_data_set_to_goal = variant.get('random_rollout_data_set_to_goal', True)
        random_and_oracle_policy_data=variant.get('random_and_oracle_policy_data', False)
        random_and_oracle_policy_data_split=variant.get('random_and_oracle_policy_data_split', 0)
        policy_file = variant.get('policy_file', None)
        n_random_steps = variant.get('n_random_steps', 100)
        vae_dataset_specific_env_kwargs = variant.get('vae_dataset_specific_env_kwargs', None)
        save_file_prefix = variant.get('save_file_prefix', None)
        non_presampled_goal_img_is_garbage = variant.get('non_presampled_goal_img_is_garbage', None)

        conditional_vae_dataset = variant.get('conditional_vae_dataset', False)
        use_env_labels = variant.get('use_env_labels', False)
        use_linear_dynamics = variant.get('use_linear_dynamics', False)
        enviorment_dataset = variant.get('enviorment_dataset', False)
        save_trajectories = variant.get('save_trajectories', False)
        save_trajectories = save_trajectories or use_linear_dynamics or conditional_vae_dataset

        tag = variant.get('tag', '')

        assert N % n_random_steps == 0, "Fix N/horizon or dataset generation will fail"

        from multiworld.core.image_env import ImageEnv, unormalize_image
        import railrl.torch.pytorch_util as ptu
        from railrl.misc.asset_loader import load_local_or_remote_file
        from railrl.data_management.dataset  import (
            TrajectoryDataset, ImageObservationDataset, EnvironmentDataset, ConditionalDynamicsDataset, InitialObservationNumpyDataset,
            InfiniteBatchLoader,
        )

        info = {}
        if dataset_path is not None:
            dataset = load_local_or_remote_file(dataset_path)
            dataset = dataset.item()
            N = dataset['observations'].shape[0] * dataset['observations'].shape[1]
            n_random_steps = dataset['observations'].shape[1]
        else:
            if env_kwargs is None:
                env_kwargs = {}
            if save_file_prefix is None:
                save_file_prefix = env_id
            if save_file_prefix is None:
                save_file_prefix = env_class.__name__
            filename = "/tmp/{}_N{}_{}_imsize{}_random_oracle_split_{}{}.npy".format(
                save_file_prefix,
                str(N),
                init_camera.__name__ if init_camera and hasattr(init_camera, '__name__') else '',
                imsize,
                random_and_oracle_policy_data_split,
                tag,
            )
            if use_cached and osp.isfile(filename):
                dataset = load_local_or_remote_file(filename)
                if conditional_vae_dataset:
                    dataset = dataset.item()
                print("loaded data from saved file", filename)
            else:
                now = time.time()

                if env_id is not None:
                    import gym
                    import multiworld
                    multiworld.register_all_envs()
                    env = gym.make(env_id)
                else:
                    if vae_dataset_specific_env_kwargs is None:
                        vae_dataset_specific_env_kwargs = {}
                    for key, val in env_kwargs.items():
                        if key not in vae_dataset_specific_env_kwargs:
                            vae_dataset_specific_env_kwargs[key] = val
                    env = env_class(**vae_dataset_specific_env_kwargs)
                if not isinstance(env, ImageEnv):
                    env = ImageEnv(
                        env,
                        imsize,
                        init_camera=init_camera,
                        transpose=True,
                        normalize=True,
                        non_presampled_goal_img_is_garbage=non_presampled_goal_img_is_garbage,
                    )
                else:
                    imsize = env.imsize
                    env.non_presampled_goal_img_is_garbage = non_presampled_goal_img_is_garbage
                env.reset()
                info['env'] = env
                if random_and_oracle_policy_data:
                    policy_file = load_local_or_remote_file(policy_file)
                    policy = policy_file['policy']
                    policy.to(ptu.device)
                if random_rollout_data:
                    from railrl.exploration_strategies.ou_strategy import OUStrategy
                    policy = OUStrategy(env.action_space)

                if save_trajectories:
                    dataset = {
                        'observations': np.zeros((N // n_random_steps, n_random_steps, imsize * imsize * num_channels), dtype=np.uint8),
                        'actions': np.zeros((N // n_random_steps, n_random_steps, env.action_space.shape[0]), dtype=np.float),
                        'env': np.zeros((N // n_random_steps, imsize * imsize * num_channels), dtype=np.uint8),
                        }
                else:
                    dataset = np.zeros((N, imsize * imsize * num_channels), dtype=np.uint8)
                labels = []
                for i in range(N):
                    if random_and_oracle_policy_data:
                        num_random_steps = int(N*random_and_oracle_policy_data_split)
                        if i < num_random_steps:
                            env.reset()
                            for _ in range(n_random_steps):
                                obs = env.step(env.action_space.sample())[0]
                        else:
                            obs = env.reset()
                            policy.reset()
                            for _ in range(n_random_steps):
                                policy_obs = np.hstack((
                                    obs['state_observation'],
                                    obs['state_desired_goal'],
                                ))
                                action, _ = policy.get_action(policy_obs)
                                obs, _, _, _ = env.step(action)
                    elif random_rollout_data: #ADD DATA WHERE JUST PUCK MOVES
                        if i % n_random_steps == 0:
                            env.reset()
                            policy.reset()
                            env_img = env._get_obs()['image_observation']
                            if random_rollout_data_set_to_goal:
                                env.set_to_goal(env.get_goal())
                        obs = env._get_obs()
                        u = policy.get_action_from_raw_action(env.action_space.sample())
                        env.step(u)
                    elif oracle_dataset_using_set_to_goal:
                        print(i)
                        goal = env.sample_goal()
                        env.set_to_goal(goal)
                        obs = env._get_obs()
                    else:
                        env.reset()
                        for _ in range(n_random_steps):
                            obs = env.step(env.action_space.sample())[0]

                    img = obs['image_observation']
                    if use_env_labels:
                        labels.append(obs['label'])
                    if save_trajectories:
                        dataset['observations'][i // n_random_steps, i % n_random_steps, :] = unormalize_image(img)
                        dataset['actions'][i // n_random_steps, i % n_random_steps, :] = u
                        dataset['env'][i // n_random_steps, :] = unormalize_image(env_img)
                    else:
                        dataset[i, :] = unormalize_image(img)

                    if show:
                        img = img.reshape(3, imsize, imsize).transpose()
                        img = img[::-1, :, ::-1]
                        cv2.imshow('img', img)
                        cv2.waitKey(1)
                        # radius = input('waiting...')
                print("done making training data", filename, time.time() - now)
                np.save(filename, dataset)
                #np.save(filename[:-4] + 'labels.npy', np.array(labels))

        info['train_labels'] = []
        info['test_labels'] = []

        if use_linear_dynamics and conditional_vae_dataset:
            num_trajectories = N // n_random_steps
            n = int(num_trajectories * test_p)
            train_dataset = ConditionalDynamicsDataset({
                'observations': dataset['observations'][:n, :, :],
                'actions': dataset['actions'][:n, :, :],
                'env': dataset['env'][:n, :]
            })
            test_dataset = ConditionalDynamicsDataset({
                'observations': dataset['observations'][n:, :, :],
                'actions': dataset['actions'][n:, :, :],
                'env': dataset['env'][n:, :]
            })

            num_trajectories = N // n_random_steps
            n = int(num_trajectories * test_p)
            indices = np.arange(num_trajectories)
            np.random.shuffle(indices)
            train_i, test_i = indices[:n], indices[n:]

            try:
                train_dataset = ConditionalDynamicsDataset({
                    'observations': dataset['observations'][train_i, :, :],
                    'actions': dataset['actions'][train_i, :, :],
                    'env': dataset['env'][train_i, :]
                })
                test_dataset = ConditionalDynamicsDataset({
                    'observations': dataset['observations'][test_i, :, :],
                    'actions': dataset['actions'][test_i, :, :],
                    'env': dataset['env'][test_i, :]
                })
            except:
                train_dataset = ConditionalDynamicsDataset({
                    'observations': dataset['observations'][train_i, :, :],
                    'actions': dataset['actions'][train_i, :, :],
                })
                test_dataset = ConditionalDynamicsDataset({
                    'observations': dataset['observations'][test_i, :, :],
                    'actions': dataset['actions'][test_i, :, :],
                })
        elif use_linear_dynamics:
            num_trajectories = N // n_random_steps
            n = int(num_trajectories * test_p)
            train_dataset = TrajectoryDataset({
                'observations': dataset['observations'][:n, :, :],
                'actions': dataset['actions'][:n, :, :]
            })
            test_dataset = TrajectoryDataset({
                'observations': dataset['observations'][n:, :, :],
                'actions': dataset['actions'][n:, :, :]
            })
        elif enviorment_dataset:
            n = int(n_random_steps * test_p)
            train_dataset = EnvironmentDataset({
                'observations': dataset['observations'][:, :n, :],
            })
            test_dataset = EnvironmentDataset({
                'observations': dataset['observations'][:, n:, :],
            })
        elif conditional_vae_dataset:
            num_trajectories = N // n_random_steps
            n = int(num_trajectories * test_p)
            indices = np.arange(num_trajectories)
            np.random.shuffle(indices)
            train_i, test_i = indices[:n], indices[n:]

            if 'env' in dataset:
                train_dataset = InitialObservationNumpyDataset({
                    'observations': dataset['observations'][train_i, :, :],
                    'env': dataset['env'][train_i, :]
                })
                test_dataset = InitialObservationNumpyDataset({
                    'observations': dataset['observations'][test_i, :, :],
                    'env': dataset['env'][test_i, :]
                })
            else:
                train_dataset = InitialObservationNumpyDataset({
                    'observations': dataset['observations'][train_i, :, :],
                })
                test_dataset = InitialObservationNumpyDataset({
                    'observations': dataset['observations'][test_i, :, :],
                })

            train_batch_loader_kwargs = variant.get(
                'train_batch_loader_kwargs',
                dict(batch_size=batch_size, num_workers=0, )
            )
            test_batch_loader_kwargs = variant.get(
                'test_batch_loader_kwargs',
                dict(batch_size=batch_size, num_workers=0, )
            )

            train_data_loader = data.DataLoader(train_dataset,
                shuffle=True, drop_last=True, **train_batch_loader_kwargs)
            test_data_loader = data.DataLoader(test_dataset,
                shuffle=True, drop_last=True, **test_batch_loader_kwargs)

            train_dataset = InfiniteBatchLoader(train_data_loader)
            test_dataset = InfiniteBatchLoader(test_data_loader)
        else:
            n = int(N * test_p)
            train_dataset = ImageObservationDataset(dataset[:n, :])
            test_dataset = ImageObservationDataset(dataset[n:, :])
        return train_dataset, test_dataset, info

    def contextual_env_distrib_and_reward(
            env_id, env_class, env_kwargs, goal_sampling_mode
    ):
        state_env = get_gym_env(env_id, env_class=env_class, env_kwargs=env_kwargs)
        state_goal_distribution = GoalDictDistributionFromMultitaskEnv(
            state_env,
            desired_goal_keys=["state_desired_goal"],
        )

        renderer = Renderer(**renderer_kwargs)
        img_env = InsertImageEnv(state_env, renderer=renderer)
        # img_env = ImageEnv(
        #     env,
        #     imsize,
        #     # init_camera=init_camera,
        #     transpose=True,
        #     normalize=True,
        #     # non_presampled_goal_img_is_garbage=non_presampled_goal_img_is_garbage,
        # )

        encoded_env = EncoderWrappedEnv(
            img_env,
            model,
            dict(image_observation="latent_observation", ),
            # dict(image_desired_goal="latent_desired_goal", ),
        )
        if goal_sampling_mode == "vae_prior":
            latent_goal_distribution = PriorDistribution(
                model,
                "latent_desired_goal",
            )
        elif goal_sampling_mode == "reset_of_env":
            image_goal_distribution = AddImageDistribution(
                env=state_env,
                base_distribution=state_goal_distribution,
                image_goal_key='image_desired_goal',
                renderer=renderer,
            )
            latent_goal_distribution = AddLatentDistribution(
                image_goal_distribution,
                "image_desired_goal",
                "latent_desired_goal",
                model,
            )
        else:
            error

        # env = VAEWrappedEnv(img_env, model, imsize=imsize)
        # env.goal_sampling_mode = goal_sampling_mode

        reward_fn = DistanceRewardFn(
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
        )
        env = ContextualEnv(
            encoded_env,
            context_distribution=latent_goal_distribution,
            reward_fn=reward_fn,
            observation_key=observation_key,
            # update_env_info_fn=DeleteOldEnvInfo(),
        )
        # env = ContextualEnv(
        #     env,
        #     context_distribution=goal_distribution,
        #     reward_fn=reward_fn,
        #     observation_key=observation_key,
        #     # update_env_info_fn=DeleteOldEnvInfo(),
        # )
        return env, latent_goal_distribution, reward_fn

    model = train_vae(train_vae_kwargs)
    if type(model) is str:
        model = load_local_or_remote_file(model)

    expl_env, expl_context_distrib, expl_reward = contextual_env_distrib_and_reward(
        env_id, env_class, env_kwargs, exploration_goal_sampling_mode
    )
    eval_env, eval_context_distrib, eval_reward = contextual_env_distrib_and_reward(
        env_id, env_class, env_kwargs, evaluation_goal_sampling_mode
    )
    context_key = desired_goal_key

    obs_dim = (
            expl_env.observation_space.spaces[observation_key].low.size
            + expl_env.observation_space.spaces[context_key].low.size
    )
    action_dim = expl_env.action_space.low.size

    def create_qf():
        return FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            **qf_kwargs
        )
    qf1 = create_qf()
    qf2 = create_qf()
    target_qf1 = create_qf()
    target_qf2 = create_qf()

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **policy_kwargs
    )

    def concat_context_to_obs(batch):
        obs = batch['observations']
        next_obs = batch['next_observations']
        context = batch[context_key]
        batch['observations'] = np.concatenate([obs, context], axis=1)
        batch['next_observations'] = np.concatenate([next_obs, context], axis=1)
        return batch
    replay_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_keys=[context_key],
        observation_keys=[observation_key],
        observation_key=observation_key,
        context_distribution=eval_context_distrib,
        sample_context_from_obs_dict_fn=RemapKeyFn({context_key: observation_key}),
        reward_fn=eval_reward,
        post_process_batch_fn=concat_context_to_obs,
        **replay_buffer_kwargs
    )
    trainer = SACTrainer(
        env=expl_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **sac_trainer_kwargs
    )

    eval_path_collector = ContextualPathCollector(
        eval_env,
        MakeDeterministic(policy),
        observation_key=observation_key,
        context_key=context_key,
    )
    exploration_policy = create_exploration_policy(
        policy, **exploration_policy_kwargs)
    expl_path_collector = ContextualPathCollector(
        expl_env,
        exploration_policy,
        observation_key=observation_key,
        context_key=context_key,
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **algo_kwargs
    )
    algorithm.to(ptu.device)

    if save_video:
        expl_video_func = RIGVideoSaveFunction(
            None,
            expl_path_collector,
            "train",
            "image_desired_goal",
            rows=2,
            columns=5,
            unnormalize=True,
            save_video_period=50,
            # max_path_length=200,
            imsize=48,
        )
        algorithm.post_train_funcs.append(expl_video_func)

        eval_video_func = RIGVideoSaveFunction(
            model,
            eval_path_collector,
            "eval",
            "image_desired_goal",
            rows=2,
            columns=5,
            unnormalize=True,
            save_video_period=50,
            # max_path_length=200,
            imsize=48,
        )
        algorithm.post_train_funcs.append(eval_video_func)

        # rollout_function = partial(
        #     rf.contextual_rollout,
        #     max_path_length=max_path_length,
        #     observation_key=observation_key,
        #     context_key=context_key,
        # )
        # renderer = Renderer(**renderer_kwargs)

        # def add_images(env, state_distribution):
        #     state_env = env.env
        #     image_goal_distribution = AddImageDistribution(
        #         env=state_env,
        #         base_distribution=state_distribution,
        #         image_goal_key='image_desired_goal',
        #         renderer=renderer,
        #     )
        #     img_env = InsertImageEnv(state_env, renderer=renderer)
        #     return ContextualEnv(
        #         img_env,
        #         context_distribution=image_goal_distribution,
        #         reward_fn=eval_reward,
        #         observation_key=observation_key,
        #         # update_env_info_fn=DeleteOldEnvInfo(),
        #     )
        # img_eval_env = eval_env # add_images(eval_env, eval_context_distrib)
        # img_expl_env = expl_env # add_images(expl_env, expl_context_distrib)
        # eval_video_func = get_save_video_function(
        #     rollout_function,
        #     img_eval_env,
        #     MakeDeterministic(policy),
        #     tag="eval",
        #     imsize=renderer.image_shape[0],
        #     image_format='CWH',
        #     **save_video_kwargs
        # )
        # expl_video_func = get_save_video_function(
        #     rollout_function,
        #     img_expl_env,
        #     exploration_policy,
        #     tag="train",
        #     imsize=renderer.image_shape[0],
        #     image_format='CWH',
        #     **save_video_kwargs
        # )

        # algorithm.post_train_funcs.append(eval_video_func)
        # algorithm.post_train_funcs.append(expl_video_func)

    algorithm.train()


def get_save_video_function(
        rollout_function,
        env,
        policy,
        save_video_period=10,
        imsize=48,
        tag="",
        **dump_video_kwargs
):
    logdir = logger.get_snapshot_dir()

    def save_video(algo, epoch):
        if epoch % save_video_period == 0 or epoch == algo.num_epochs:
            filename = osp.join(
                logdir,
                'video_{}_{epoch}_env.mp4'.format(tag, epoch=epoch),
            )
            dump_video(env, policy, filename, rollout_function,
                       imsize=imsize, **dump_video_kwargs)
    return save_video


def get_gym_env(env_id, env_class=None, env_kwargs=None):
    if env_kwargs is None:
        env_kwargs = {}

    assert env_id or env_class
    if env_id:
        import gym
        import multiworld
        multiworld.register_all_envs()
        env = gym.make(env_id)
    else:
        env = env_class(**env_kwargs)
    return env


def process_args(variant):
    if variant.get("debug", False):
        variant["train_vae_kwargs"]["num_epochs"] = 1
        variant["max_path_length"] = 50
        algo_kwargs = variant["algo_kwargs"]
        algo_kwargs["batch_size"] = 2
        algo_kwargs["num_eval_steps_per_epoch"] = 500
        algo_kwargs["num_expl_steps_per_train_loop"] = 500
        algo_kwargs["num_trains_per_train_loop"] = 50
        algo_kwargs["min_num_steps_before_training"] = 500
