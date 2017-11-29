from railrl.launchers.launcher_util import run_experiment
import railrl.misc.hyperparameter as hyp


def experiment(variant):
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
        callback=callback,
        logger=logger,
        **variant['algo_kwargs']
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")
    logger.save_itr_params(100, dict(
        action_function=act
    ))


def main():
    variant = dict(
        algo_kwargs=dict(
            lr=1e-3,
            max_timesteps=1000000,
            buffer_size=50000,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            print_freq=1000,
        ),
    )
    search_space = {
        'algo_kwargs.prioritized_replay': [True, False],
        'algo_kwargs.gamma': [1, 0.99],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(3):
            run_experiment(
                experiment,
                exp_id=exp_id,
                variant=variant,
                exp_prefix="openai-baselines-dqn-cartpole-sweep",
                mode='ec2',
                # exp_prefix="dev-openai-baselines-dqn-cartpole",
                # mode='local',
            )


if __name__ == '__main__':
    main()
