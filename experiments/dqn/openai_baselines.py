from rlkit.launchers.launcher_util import run_experiment
import rlkit.misc.hyperparameter as hyp


def experiment(variant):
    import gym
    from baselines import deepq
    from rlkit.core import logger

    def callback(lcl, glb):
        # only stop training after number of time steps have been reached
        return False

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
    # logger.save_itr_params(100, dict(
    #     action_function=act
    # ))


def main():
    variant = dict(
        algo_kwargs=dict(
            lr=1e-3,
            max_timesteps=int(1E7),
            buffer_size=int(5E5),
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            print_freq=1000,
            gamma=0.99,
        ),
    )
    search_space = {
        'algo_kwargs.prioritized_replay': [True, False],
        'algo_kwargs.lr': [1e-2, 1e-3, 1e-4],
        'algo_kwargs.exploration_fraction': [0.1, 0.5],
        'algo_kwargs.exploration_final_eps': [0.2, 0.02],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(1):
            run_experiment(
                experiment,
                exp_id=exp_id,
                variant=variant,
                exp_prefix="openai-baselines-dqn-cartpole-sweep-2",
                mode='ec2',
                # exp_prefix="dev-openai-baselines-dqn-cartpole-2",
                # mode='local',
            )


if __name__ == '__main__':
    main()
