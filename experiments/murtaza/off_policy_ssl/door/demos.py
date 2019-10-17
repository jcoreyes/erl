from railrl.demos.collect_demo import collect_demos
from railrl.torch.networks import TanhMlpPolicy

if __name__ == '__main__':
    policy_file = ''
    import gym
    import multiworld
    multiworld.register_all_envs()
    env_id = 'SawyerDoorHookResetFreeEnv-v1'
    env = gym.make('env_id')
    obs_dim = env.observation_space.spaces['observation'].low.size
    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    action_dim = env.action_space.low.size
    policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )
    #reload policy
    collect_demos(env, policy, "door_demos_100.npy", 100)
