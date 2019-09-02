def her_td3_experiment(variant):
    import gym
    import multiworld.envs.mujoco
    import multiworld.envs.pygame
    import railrl.samplers.rollout_functions as rf
    import railrl.torch.pytorch_util as ptu
    from railrl.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
    from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
    from railrl.exploration_strategies.ou_strategy import OUStrategy
    from railrl.torch.grill.launcher import get_state_experiment_video_save_function
    from railrl.torch.her.her_td3 import HerTd3
    from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
    from railrl.data_management.obs_dict_replay_buffer import (
        ObsDictRelabelingBuffer
    )
    from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

    if 'env_id' in variant:
        env = gym.make(variant['env_id'])
    else:
        env = variant['env_class'](**variant['env_kwargs'])

    observation_key = variant['observation_key']
    desired_goal_key = variant['desired_goal_key']
    variant['algo_kwargs']['her_kwargs']['observation_key'] = observation_key
    variant['algo_kwargs']['her_kwargs']['desired_goal_key'] = desired_goal_key
    if variant.get('normalize', False):
        raise NotImplementedError()

    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )
    obs_dim = env.observation_space.spaces['observation'].low.size
    action_dim = env.action_space.low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    exploration_type = variant['exploration_type']
    if exploration_type == 'ou':
        es = OUStrategy(
            action_space=env.action_space,
            **variant['es_kwargs']
        )
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            **variant['es_kwargs'],
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            **variant['es_kwargs'],
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    trainer = TD3(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        # target_qf1=target_qf1,
        # target_qf2=target_qf2,
        # target_policy=target_policy,
        **variant['td3_trainer_kwargs']
    )
    # trainer = HERTrainer(trainer)

    # algorithm = HerTd3(
    #     env,
    #     qf1=qf1,
    #     qf2=qf2,
    #     policy=policy,
    #     exploration_policy=exploration_policy,
    #     replay_buffer=replay_buffer,
    #     **variant['algo_kwargs']
    # )
    algorithm = TorchBatchRLAlgorithm(
        trainer,
        exploration_env,
        evaluation_env,
        exploration_data_collector: DataCollector,
        evaluation_data_collector: DataCollector,
        replay_buffer: ReplayBuffer,
        **variant['algo_kwargs']
    )

    if variant.get("save_video", False):
        rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            max_path_length=algorithm.max_path_length,
            observation_key=algorithm.observation_key,
            desired_goal_key=algorithm.desired_goal_key,
        )
        video_func = get_state_experiment_video_save_function(
            rollout_function,
            env,
            policy,
            variant,
        )
        algorithm.post_epoch_funcs.append(video_func)
    algorithm.to(ptu.device)
    algorithm.train()

def her_td3_experiment(variant):
    import railrl.samplers.rollout_functions as rf
    import railrl.torch.pytorch_util as ptu
    from railrl.data_management.online_vae_replay_buffer import \
        OnlineVaeRelabelingBuffer
    from railrl.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )
    from railrl.torch.her.online_vae_joint_algo import OnlineVaeHerJointAlgo
    from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
    from railrl.torch.td3.td3 import TD3
    from railrl.torch.vae.vae_trainer import ConvVAETrainer
    from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

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
        **variant['replay_buffer_kwargs']
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

    assert 'vae_training_schedule' not in variant,\
        "Just put it in joint_algo_kwargs"
    algorithm = TorchBatchRLAlgorithm(
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


    algorithm.to(ptu.device)
    vae.to(ptu.device)
    if variant.get("save_video", True):
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
        algorithm.post_train_funcs.append(video_func)
    algorithm.train()
