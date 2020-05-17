import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.data_management.her_replay_buffer import SimpleHerReplayBuffer
from railrl.envs.mujoco.sawyer_gripper_env import SawyerXYZEnv
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.her.her_sac import HerSac
from railrl.torch.networks import ConcatMlp
from railrl.torch.sac.policies import TanhGaussianPolicy


def experiment(variant):
    env = SawyerXYZEnv(**variant['env_kwargs'])
    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    goal_dim = env.goal_dim
    qf = ConcatMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    vf = ConcatMlp(
        input_size=obs_dim + goal_dim,
        output_size=1,
        **variant['vf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + goal_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )
    replay_buffer = SimpleHerReplayBuffer(
        env=env,
        **variant['replay_buffer_kwargs']
    )
    algorithm = HerSac(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=100,
            discount=0.99,
        ),
        env_kwargs=dict(
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            num_goals_to_sample=0,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        vf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        normalize=True,
        algorithm='HER-SAC',
        version='normal',
    )
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'sawyer-sim-xyz-state'

    search_space = {
        'replay_buffer_kwargs.num_goals_to_sample': [0, 4, 8],
        'algo_kwargs.num_updates_per_env_step': [1, 5],
        'algo_kwargs.reward_scale': [1, 10, 100],
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
            )
