import os.path as osp
import pickle
import time

import cv2
import numpy as np

import railrl.samplers.rollout_functions as rf
import railrl.torch.pytorch_util as ptu
from multiworld.core.image_env import ImageEnv, unormalize_image
from railrl.core import logger
from railrl.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from railrl.data_management.online_vae_replay_buffer import \
    OnlineVaeRelabelingBuffer
from railrl.envs.vae_wrappers import VAEWrappedEnv, temporary_mode
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.misc.asset_loader import local_path_from_s3_or_local_path
from railrl.misc.ml_util import PiecewiseLinearSchedule
from railrl.state_distance.tdm_networks import TdmQf, TdmVf, TdmPolicy, StochasticTdmPolicy
from railrl.state_distance.tdm_td3 import TdmTd3
from railrl.state_distance.tdm_twin_sac import TdmTwinSAC
from railrl.torch.grill.video_gen import dump_video
from railrl.torch.her.her_td3 import HerTd3
from railrl.torch.her.her_twin_sac import HerTwinSAC
from railrl.torch.her.online_vae_her_td3 import OnlineVaeHerTd3
from railrl.torch.her.online_vae_her_twin_sac import OnlineVaeHerTwinSac
from railrl.torch.her.online_vae_joint_algo import OnlineVaeHerJointAlgo
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
from railrl.torch.sac.policies import TanhGaussianPolicy
from railrl.torch.td3.td3 import TD3
from railrl.torch.vae.conv_vae import ConvVAE, ConvVAETrainer, SpatialVAE, AutoEncoder
from railrl.torch.sac.policies import TanhGaussianPolicy
from railrl.torch.online_vae.online_vae_tdm_td3 import OnlineVaeTdmTd3
from railrl.misc.asset_loader import sync_down


def grill_tdm_td3_full_experiment(variant):
    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)
    grill_tdm_td3_experiment(variant['grill_variant'])


def grill_tdm_twin_sac_full_experiment(variant):
    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)
    grill_tdm_twin_sac_experiment(variant['grill_variant'])


def grill_her_td3_full_experiment(variant):
    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)
    grill_her_td3_experiment(variant['grill_variant'])


def grill_her_td3_online_vae_full_experiment(variant):
    variant['grill_variant']['save_vae_data'] = True
    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)
    variant['grill_variant']['vae_trainer_kwargs'] = \
            variant['train_vae_variant']['algo_kwargs']
    if variant['double_algo']:
        grill_her_td3_experiment_online_vae_exploring(variant['grill_variant'])
    else:
        grill_her_td3_experiment_online_vae(variant['grill_variant'])

def grill_her_twin_sac_online_vae_full_experiment(variant):
    variant['grill_variant']['save_vae_data'] = True
    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)
    grill_her_twin_sac_experiment_online_vae(variant['grill_variant'])

def grill_tdm_td3_online_vae_full_experiment(variant):
    variant['grill_variant']['save_vae_data'] = True
    variant['grill_variant']['vae_trainer_kwargs'] = \
            variant['train_vae_variant']['algo_kwargs']

    full_experiment_variant_preprocess(variant)
    train_vae_and_update_variant(variant)
    grill_tdm_td3_experiment_online_vae(variant['grill_variant'])


def full_experiment_variant_preprocess(variant):
    train_vae_variant = variant['train_vae_variant']
    grill_variant = variant['grill_variant']
    env_class = variant['env_class']
    env_kwargs = variant['env_kwargs']
    init_camera = variant.get('init_camera', None)
    train_vae_variant['generate_vae_dataset_kwargs']['env_class'] = env_class
    train_vae_variant['generate_vae_dataset_kwargs']['env_kwargs'] = env_kwargs
    train_vae_variant['generate_vae_dataset_kwargs']['init_camera'] = init_camera
    grill_variant['env_class'] = env_class
    grill_variant['env_kwargs'] = env_kwargs
    grill_variant['init_camera'] = init_camera


def train_vae_and_update_variant(variant):
    grill_variant = variant['grill_variant']
    train_vae_variant = variant['train_vae_variant']
    if grill_variant.get('vae_path', None) is None:
        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'vae_progress.csv', relative_to_snapshot_dir=True
        )
        vae, vae_train_data, vae_test_data = train_vae(train_vae_variant, return_data=True)
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
                    **train_vae_variant['generate_vae_dataset_kwargs']
            )
            grill_variant['vae_train_data'] = vae_train_data
            grill_variant['vae_test_data'] = vae_test_data

def train_vae(variant, return_data=False):
    from railrl.core import logger
    import railrl.torch.pytorch_util as ptu
    beta = variant["beta"]
    representation_size = variant["representation_size"]
    generate_vae_dataset_fctn = variant.get('generate_vae_data_fctn', generate_vae_dataset)
    train_data, test_data, info = generate_vae_dataset_fctn(
        **variant['generate_vae_dataset_kwargs']
    )
    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    if 'beta_schedule_kwargs' in variant:
        beta_schedule = PiecewiseLinearSchedule(**variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None
    if variant['algo_kwargs'].get('is_auto_encoder', False):
        m = AutoEncoder(representation_size, input_channels=3)
    elif variant.get('use_spatial_auto_encoder', False):
        m = SpatialVAE(representation_size, int(representation_size/2), input_channels=3)
    else:
        m = ConvVAE(representation_size, input_channels=3)
    if ptu.gpu_enabled():
        m.cuda()
    t = ConvVAETrainer(train_data, test_data, m, beta=beta,
                       beta_schedule=beta_schedule, **variant['algo_kwargs'])
    save_period = variant['save_period']
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
    logger.save_extra_data(m, 'vae.pkl', mode='pickle')
    if return_data:
        return m, train_data, test_data
    return m


def generate_vae_dataset(
        env_class,
        N=10000,
        test_p=0.9,
        use_cached=True,
        imsize=84,
        num_channels=1,
        show=False,
        init_camera=None,
        dataset_path=None,
        env_kwargs=None,
        oracle_dataset=False,
        n_random_steps=100,
        vae_dataset_specific_env_kwargs=None,
):
    if env_kwargs is None:
        env_kwargs = {}
    filename = "/tmp/{}_{}_{}_oracle{}.npy".format(
        env_class.__name__,
        str(N),
        init_camera.__name__ if init_camera else '',
        oracle_dataset,
    )
    info = {}
    if dataset_path is not None:
        filename = local_path_from_s3_or_local_path(dataset_path)
        dataset = np.load(filename)
        N = dataset.shape[0]
    elif use_cached and osp.isfile(filename):
        dataset = np.load(filename)
        print("loaded data from saved file", filename)
    else:

        if vae_dataset_specific_env_kwargs is None:
            vae_dataset_specific_env_kwargs = {}
        for key, val in env_kwargs.items():
            if key not in vae_dataset_specific_env_kwargs:
                vae_dataset_specific_env_kwargs[key] = val
        now = time.time()
        env = env_class(**vae_dataset_specific_env_kwargs)
        env = ImageEnv(
            env,
            imsize,
            init_camera=init_camera,
            transpose=True,
            normalize=True,
        )
        env.reset()
        info['env'] = env

        dataset = np.zeros((N, imsize * imsize * num_channels), dtype=np.uint8)
        for i in range(N):
            if oracle_dataset:
                goal = env.sample_goal()
                env.set_to_goal(goal)
            else:
                env.reset()
                for _ in range(n_random_steps):
                    obs = env.step(env.action_space.sample())[0]
            obs = env.step(env.action_space.sample())[0]
            img = obs['image_observation']
            dataset[i, :] = unormalize_image(img)
            if show:
                img = img.reshape(3, 84, 84).transpose()
                img = img[::-1, :, ::-1]
                cv2.imshow('img', img)
                cv2.waitKey(1)
                # radius = input('waiting...')
        print("done making training data", filename, time.time() - now)
        np.save(filename, dataset)

    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset, info


def get_envs(variant):
    render = variant["render"]
    vae_path = variant.get("vae_path", None)
    reward_params = variant.get("reward_params", dict())
    init_camera = variant.get("init_camera", None)
    do_state_exp = variant.get("do_state_exp", False)
    from railrl.envs.vae_wrappers import load_vae
    vae = load_vae(vae_path) if type(vae_path) is str else vae_path
    presample_goals = variant.get('presample_goals', False)
    env = variant["env_class"](**variant['env_kwargs'])
    if not do_state_exp:
        env = ImageEnv(
            env,
            84,
            init_camera=init_camera,
            transpose=True,
            normalize=True,
        )
        env = VAEWrappedEnv(
            env,
            vae,
            decode_goals=render,
            render_goals=render,
            render_rollouts=render,
            reward_params=reward_params,
            **variant.get('vae_wrapped_env_kwargs', {})
        )
        if presample_goals:
            presampled_goals = variant['generate_goal_dataset_fn'](env=env, **variant['goal_generation_kwargs'])
            env.set_presampled_goals(presampled_goals)

    if not do_state_exp:
        training_mode = variant.get("training_mode", "train")
        testing_mode = variant.get("testing_mode", "test")
        env.add_mode('eval', testing_mode)
        env.add_mode('train', training_mode)
        env.add_mode('relabeling', training_mode)
        # relabeling_env.disable_render()
        env.add_mode("video_vae", 'video_vae')
        env.add_mode("video_env", 'video_env')
    return env


def get_exploration_strategy(variant, env):
    exploration_type = variant['exploration_type']
    exploration_noise = variant.get('exploration_noise', 0.1)
    if exploration_type == 'ou':
        es = OUStrategy(
            action_space=env.action_space,
            max_sigma=exploration_noise,
            min_sigma=exploration_noise,  # Constant sigma
        )
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=exploration_noise,
            min_sigma=exploration_noise,  # Constant sigma
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=exploration_noise,
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    return es


def grill_preprocess_variant(variant):
    if variant.get("do_state_exp", False):
        variant['observation_key'] = 'state_observation'
        variant['desired_goal_key'] = 'state_desired_goal'
        variant['achieved_goal_key'] = 'state_acheived_goal'


def grill_her_td3_experiment(variant):
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
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
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
        **variant['replay_kwargs']
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
    algorithm = HerTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
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

    if ptu.gpu_enabled():
        print("using GPU")
        algorithm.cuda()
        if not variant.get("do_state_exp", False):
            env.vae.cuda()

    algorithm.train()


def grill_her_twin_sac_experiment(variant):
    grill_preprocess_variant(variant)
    env = get_envs(variant)

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        env.observation_space.spaces[observation_key].low.size
        + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    vf = FlattenMlp(
        input_size=obs_dim,
        output_size=1,
        **variant['vf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_kwargs']
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
    algorithm = HerTwinSAC(
        env,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        policy=policy,
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

    if ptu.gpu_enabled():
        print("using GPU")
        algorithm.cuda()
        if not variant.get("do_state_exp", False):
            env.vae.cuda()

    algorithm.train()


def grill_tdm_td3_experiment(variant):
    grill_preprocess_variant(variant)
    env = get_envs(variant)
    es = get_exploration_strategy(variant, env)
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        env.observation_space.spaces[observation_key].low.size
    )
    goal_dim = (
        env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size

    vectorized = 'vectorized' in env.reward_type
    norm_order = env.norm_order
    variant['algo_kwargs']['tdm_kwargs']['vectorized'] = vectorized
    variant['qf_kwargs']['vectorized'] = vectorized
    variant['qf_kwargs']['norm_order'] = norm_order

    qf1 = TdmQf(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    qf2 = TdmQf(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    policy = TdmPolicy(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    variant['replay_kwargs']['vectorized'] = vectorized
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_kwargs']
    )
    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['replay_buffer'] = replay_buffer
    base_kwargs = algo_kwargs['base_kwargs']
    base_kwargs['training_env'] = env
    base_kwargs['render'] = variant["render"]
    base_kwargs['render_during_eval'] = variant["render"]
    tdm_kwargs = algo_kwargs['tdm_kwargs']
    tdm_kwargs['observation_key'] = observation_key
    tdm_kwargs['desired_goal_key'] = desired_goal_key
    algorithm = TdmTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )

    if ptu.gpu_enabled():
        print("using GPU")
        algorithm.cuda()
        if not variant.get("do_state_exp", False):
            env.vae.cuda()
    if variant.get("save_video", True):
        logdir = logger.get_snapshot_dir()
        policy.train(False)
        rollout_function = rf.create_rollout_function(
            rf.tdm_rollout,
            init_tau=algorithm.max_tau,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            rollout_function,
            env,
            policy,
            variant,
        )
        algorithm.post_epoch_funcs.append(video_func)
    algorithm.train()

def grill_her_twin_sac_experiment_online_vae(variant):
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
    hidden_sizes = variant.get('hidden_sizes', [400, 300])
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    vf = FlattenMlp(
        input_size=obs_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    )

    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    vae = env.vae

    replay_buffer = OnlineVaeRelabelingBuffer(
        vae=vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_kwargs']
    )
    variant["algo_kwargs"]["replay_buffer"] = replay_buffer

    t = ConvVAETrainer(variant['vae_train_data'],
                       variant['vae_test_data'],
                       vae,
                       beta=variant['online_vae_beta'])
    render = variant["render"]
    assert 'vae_training_schedule' not in variant, "Just put it in algo_kwargs"
    algorithm = OnlineVaeHerTwinSac(
        algo_kwargs=dict(
            env=env,
            training_env=env,
            qf1=qf1,
            qf2=qf2,
            vf=vf,
            policy=policy,
            render=render,
            render_during_eval=render,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
            **variant['algo_kwargs']
        ),
        online_vae_algo_kwargs=dict(
            vae=vae,
            vae_trainer=t,
            **variant['online_vae_algo_kwargs']
        )
    )


    if ptu.gpu_enabled():
        print("using GPU")
        algorithm.cuda()
        vae.cuda()
    if variant.get("save_video", True):
        logdir = logger.get_snapshot_dir()
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_video_save_func(
            algorithm,
            rollout_function,
            env,
            policy,
            variant,
        )
        algorithm.post_epoch_funcs.append(video_func)
    algorithm.train()

def grill_tdm_td3_experiment_online_vae(variant):
    grill_preprocess_variant(variant)
    env = get_envs(variant)
    es = get_exploration_strategy(variant, env)
    vae_trainer_kwargs = variant.get('vae_trainer_kwargs')
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        env.observation_space.spaces[observation_key].low.size
    )
    goal_dim = (
        env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size

    vectorized = 'vectorized' in env.reward_type
    variant['algo_kwargs']['tdm_td3_kwargs']['tdm_kwargs']['vectorized'] = vectorized

    norm_order = env.norm_order
    variant['algo_kwargs']['tdm_td3_kwargs']['tdm_kwargs']['norm_order'] = norm_order

    qf1 = TdmQf(
        env=env,
        vectorized=vectorized,
        norm_order=norm_order,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    qf2 = TdmQf(
        env=env,
        vectorized=vectorized,
        norm_order=norm_order,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    policy = TdmPolicy(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    vae = env.vae

    replay_buffer = OnlineVaeRelabelingBuffer(
        vae=vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_kwargs']
    )
    algo_kwargs = variant['algo_kwargs']['tdm_td3_kwargs']
    td3_kwargs = algo_kwargs['td3_kwargs']
    td3_kwargs['training_env'] = env
    tdm_kwargs = algo_kwargs['tdm_kwargs']
    tdm_kwargs['observation_key'] = observation_key
    tdm_kwargs['desired_goal_key'] = desired_goal_key
    algo_kwargs["replay_buffer"] = replay_buffer

    t = ConvVAETrainer(variant['vae_train_data'],
                       variant['vae_test_data'],
                       vae,
                       beta=variant['online_vae_beta'],
                       **vae_trainer_kwargs)
    render = variant["render"]
    assert 'vae_training_schedule' not in variant, "Just put it in algo_kwargs"
    algorithm = OnlineVaeTdmTd3(
        online_vae_kwargs=dict(
            vae=vae,
            vae_trainer=t,
            **variant['algo_kwargs']['online_vae_kwargs']
        ),
        tdm_td3_kwargs=dict(
            env=env,
            qf1=qf1,
            qf2=qf2,
            policy=policy,
            exploration_policy=exploration_policy,
            **variant['algo_kwargs']['tdm_td3_kwargs']
        ),
    )

    if ptu.gpu_enabled():
        print("using GPU")
        algorithm.cuda()
        vae.cuda()
    if variant.get("save_video", True):
        rollout_function = rf.create_rollout_function(
            rf.tdm_rollout,
            init_tau=algorithm._sample_max_tau_for_rollout(),
            decrement_tau=algorithm.cycle_taus_for_rollout,
            cycle_tau=algorithm.cycle_taus_for_rollout,
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

    if ptu.gpu_enabled():
        print("using GPU")
        algorithm.cuda()
        if not variant.get("do_state_exp", False):
            env.vae.cuda()

    algorithm.train()


def grill_tdm_twin_sac_experiment(variant):
    grill_preprocess_variant(variant)
    env = get_envs(variant)
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
        env.observation_space.spaces[observation_key].low.size
    )
    goal_dim = (
        env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size

    vectorized = 'vectorized' in env.reward_type
    norm_order = env.norm_order
    variant['algo_kwargs']['tdm_kwargs']['vectorized'] = vectorized
    variant['qf_kwargs']['vectorized'] = vectorized
    variant['vf_kwargs']['vectorized'] = vectorized
    variant['qf_kwargs']['norm_order'] = norm_order
    variant['vf_kwargs']['norm_order'] = norm_order

    qf1 = TdmQf(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    qf2 = TdmQf(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['qf_kwargs']
    )
    vf = TdmVf(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        **variant['vf_kwargs']
    )
    policy = StochasticTdmPolicy(
        env=env,
        observation_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    variant['replay_kwargs']['vectorized'] = vectorized
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_kwargs']
    )
    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['replay_buffer'] = replay_buffer
    base_kwargs = algo_kwargs['base_kwargs']
    base_kwargs['training_env'] = env
    base_kwargs['render'] = variant["render"]
    base_kwargs['render_during_eval'] = variant["render"]
    tdm_kwargs = algo_kwargs['tdm_kwargs']
    tdm_kwargs['observation_key'] = observation_key
    tdm_kwargs['desired_goal_key'] = desired_goal_key
    algorithm = TdmTwinSAC(
        env,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        policy=policy,
        **variant['algo_kwargs']
    )

    if variant.get("save_video", True):
        rollout_function = rf.create_rollout_function(
            rf.tdm_rollout,
            init_tau=algorithm._sample_max_tau_for_rollout(),
            decrement_tau=algorithm.cycle_taus_for_rollout,
            cycle_tau=algorithm.cycle_taus_for_rollout,
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

    if ptu.gpu_enabled():
        print("using GPU")
        algorithm.cuda()
        if not variant.get("do_state_exp", False):
            env.vae.cuda()

    algorithm.train()


def grill_her_td3_experiment_online_vae(variant):
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
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs'],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    vae = env.vae
    vae.action_dim = action_dim

    replay_buffer = OnlineVaeRelabelingBuffer(
        vae=vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_kwargs']
    )
    variant["algo_kwargs"]["base_kwargs"]["replay_buffer"] = replay_buffer
    if variant.get('use_replay_buffer_goals', False):
        env.replay_buffer = replay_buffer
        env.use_replay_buffer_goals = True


    vae_trainer_kwargs = variant.get('vae_trainer_kwargs')
    t = ConvVAETrainer(variant['vae_train_data'],
                       variant['vae_test_data'],
                       vae,
                       beta=variant['online_vae_beta'],
                       **vae_trainer_kwargs)
    render = variant["render"]
    assert 'vae_training_schedule' not in variant, "Just put it in algo_kwargs"
    algorithm = OnlineVaeHerTd3(
        online_vae_kwargs=dict(
            vae=vae,
            vae_trainer=t,
            **variant['algo_kwargs']['online_vae_kwargs']
        ),
        base_kwargs=dict(
            env=env,
            training_env=env,
            policy=policy,
            exploration_policy=exploration_policy,
            render=render,
            render_during_eval=render,
            **variant['algo_kwargs']['base_kwargs'],
        ),
        her_kwargs=dict(
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
        ),
        td3_kwargs=dict(
            **variant['algo_kwargs']['td3_kwargs'],
            qf1=qf1,
            qf2=qf2,
        )
    )

    if ptu.gpu_enabled():
        print("using GPU")
        algorithm.cuda()
        vae.cuda()
    if variant.get("save_video", True):
        logdir = logger.get_snapshot_dir()
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
    algorithm.train()

def grill_her_td3_experiment_online_vae_exploring(variant):
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
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs'],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    exploring_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    exploring_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs'],
    )
    exploring_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs'],
    )
    exploring_exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=exploring_policy,
    )

    vae = env.vae
    replay_buffer = OnlineVaeRelabelingBuffer(
        vae=vae,
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_kwargs']
    )
    variant["algo_kwargs"]["replay_buffer"] = replay_buffer
    if variant.get('use_replay_buffer_goals', False):
        env.replay_buffer = replay_buffer
        env.use_replay_buffer_goals = True

    vae_trainer_kwargs = variant.get('vae_trainer_kwargs')
    t = ConvVAETrainer(variant['vae_train_data'],
                       variant['vae_test_data'],
                       vae,
                       beta=variant['online_vae_beta'],
                       **vae_trainer_kwargs)

    control_algorithm = TD3(
        env=env,
        training_env=env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    exploring_algorithm = TD3(
        env=env,
        training_env=env,
        qf1=exploring_qf1,
        qf2=exploring_qf2,
        policy=exploring_policy,
        exploration_policy=exploring_exploration_policy,
        **variant['algo_kwargs']
    )

    assert 'vae_training_schedule' not in variant, "Just put it in joint_algo_kwargs"
    algorithm = OnlineVaeHerJointAlgo(
        vae=vae,
        vae_trainer=t,
        env=env,
        training_env=env,
        policy=policy,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        algo1=control_algorithm,
        algo2=exploring_algorithm,
        algo1_prefix="Control_",
        algo2_prefix="VAE_Exploration_",
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        **variant['joint_algo_kwargs']
    )

    if ptu.gpu_enabled():
        print("using GPU")
        algorithm.cuda()
        vae.cuda()
    if variant.get("save_video", True):
        logdir = logger.get_snapshot_dir()
        policy.train(False)
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
    algorithm.train()

def get_video_save_func(rollout_function, env, policy, variant):
    logdir = logger.get_snapshot_dir()
    save_period = variant.get('save_video_period', 50)
    do_state_exp = variant.get("do_state_exp", False)
    dump_video_kwargs = variant.get("dump_video_kwargs", dict())

    if do_state_exp:
        image_env = ImageEnv(
            env,
            84,
            init_camera=variant.get('init_camera', None),
            transpose=True,
            normalize=True,
        )
        def save_video(algo, epoch):
            if epoch % save_period == 0 or epoch == algo.num_epochs:
                filename = osp.join(logdir,
                                    'video_{epoch}_env.mp4'.format(epoch=epoch))
                dump_video(image_env, policy, filename, rollout_function, **dump_video_kwargs)
    else:
        image_env = env
        def save_video(algo, epoch):
            if epoch % save_period == 0 or epoch == algo.num_epochs:
                filename = osp.join(logdir,
                                    'video_{epoch}_env.mp4'.format(epoch=epoch))
                temporary_mode(
                    image_env,
                    mode='video_env',
                    func=dump_video,
                    args=(image_env, policy, filename, rollout_function),
                    kwargs=dump_video_kwargs
                )
                filename = osp.join(logdir,
                                    'video_{epoch}_vae.mp4'.format(epoch=epoch))
                temporary_mode(
                    image_env,
                    mode='video_vae',
                    func=dump_video,
                    args=(image_env, policy, filename, rollout_function),
                    kwargs=dump_video_kwargs
                )
    return save_video
