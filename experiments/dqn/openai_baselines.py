from railrl.launchers.launcher_util import run_experiment


def experiment(_):
    import gym
    from baselines import deepq
    from rllab.misc import logger

    def callback(lcl, glb):
        # stop training if reward exceeds 199
        is_solved = lcl['t'] > 100 and sum(
            lcl['episode_rewards'][-101:-1]) / 100 >= 199
        return is_solved

    env = gym.make("CartPole-v0")
    model = deepq.models.mlp([64])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=1000000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=1000,
        callback=callback,
        logger=logger,
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")
    logger.save_itr_params(100, dict(
        action_function=act
    ))


def main():
    run_experiment(
        experiment,
        exp_prefix="openai-baselines-dqn-cartpole"
    )


if __name__ == '__main__':
    main()
