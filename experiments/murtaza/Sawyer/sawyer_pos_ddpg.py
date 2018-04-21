from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
from railrl.torch.ddpg.ddpg import DDPG
import railrl.torch.pytorch_util as ptu
from sawyer_control.sawyer_reaching import SawyerXYZReachingEnv
import railrl.misc.hyperparameter as hyp


class FakePolicy:
    def __init__(self, desired):
        self.desired = desired

    def get_action(self, obs):
        return self.desired - obs[21:24], {}

    def reset(self):
        pass

def experiment(variant):
    env_params = variant['env_params']
    env = SawyerXYZReachingEnv(**env_params)
    # es = EpsilonGreedy(action_space=env.action_space, prob_random_action=.2)
    es = OUStrategy(action_space=env.action_space)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )
    fake_policy = FakePolicy(env.desired)
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = DDPG(
        env,
        qf=qf,
        policy=policy,
        exploration_policy=exploration_policy,
        # eval_policy=fake_policy,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_params=dict(
            num_epochs=10,
            num_steps_per_epoch=50,
            num_steps_per_eval=50,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            max_path_length=10,
            discount=0.9,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            render=False,
            num_updates_per_env_step=1,
            #collection_mode='online-parallel'
        ),
        env_params=dict(
            desired=[0.97711039, 0.56662792, 0.67901027],
            action_mode='position',
            reward_magnitude=1,
        )
    )
    search_space = {
        'algo_params.reward_scale': [
            # 1,
            # 10,
            100,
            # 1000,
        ],
        'algo_params.num_updates_per_env_step': [
            1,
            # 5,
            # 10,
            # 15,
        ],
        'env_params.randomize_goal_on_reset': [
            # True,
            False,
        ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    for variant in sweeper.iterate_hyperparameters():
        n_seeds = 1
        exp_prefix = 'test'
        mode = 'here_no_doodad'
        for i in range(n_seeds):
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                variant=variant,
            )
