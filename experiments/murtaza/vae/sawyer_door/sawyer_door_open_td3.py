import rlkit.misc.hyperparameter as hyp
from rlkit.data_management.her_replay_buffer import RelabelingReplayBuffer
from rlkit.envs.mujoco.sawyer_door_env_flat import SawyerDoorPushOpenEnv, SawyerDoorPullOpenEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
from rlkit.torch.td3.td3 import TD3
import rlkit.torch.pytorch_util as ptu

def experiment(variant):
    env = NormalizedBoxEnv(variant['env_class']())
    es = OUStrategy(
        action_space=env.action_space,
        max_sigma=0.1,
        min_sigma=0.1,  # Constant sigma
    )
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = TD3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    env.set_goal(variant['goal'])
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()

if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            num_epochs=100,
            num_steps_per_epoch=100,
            num_steps_per_eval=100,
            max_path_length=100,
            num_updates_per_env_step=1,
            batch_size=128,
            discount=0.99,
            min_num_steps_before_training=128,
            render=True,
        ),
        goal=.25,
        env_class=SawyerDoorPushOpenEnv,
        env_kwargs=dict(
        ),
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'test'

    search_space = {
        'algo_kwargs.num_updates_per_env_step': [
            1,
        #     5,
        #     10,
        ],
        # 'algo_kwargs.max_path_length': [
        #     75,
        #     100,
        #     125,
        #     150,
        # ],
        # 'goal': [
        #     0.1,
        #     .25,
        #     .5,
        #     .75,
        # ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                exp_id=94439,
            )
