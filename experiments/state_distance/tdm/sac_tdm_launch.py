from torch.nn.functional import tanh
import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.her_replay_buffer import HerReplayBuffer
from rlkit.envs.multitask.ant_env import GoalXYPosAnt
from rlkit.envs.multitask.point2d import MultitaskPoint2DEnv
from rlkit.envs.multitask.point2d_uwall import MultitaskPoint2dUWall
from rlkit.envs.multitask.point2d_wall import MultitaskPoint2dWall
from rlkit.envs.multitask.reacher_7dof import (
    Reacher7DofXyzGoalState,
    Reacher7DofFullGoal)
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.state_distance.tdm_networks import (
    TdmNormalizer,
    TdmQf,
    StochasticTdmPolicy,
    TdmVf,
)
from rlkit.state_distance.tdm_sac import TdmSac


def experiment(variant):
    vectorized = variant['sac_tdm_kwargs']['tdm_kwargs']['vectorized']
    # env = NormalizedBoxEnv(variant['env_class'](**variant['env_kwargs']))
    env = NormalizedBoxEnv(variant['env_class'](**variant['env_kwargs']))
    max_tau = variant['sac_tdm_kwargs']['tdm_kwargs']['max_tau']
    tdm_normalizer = TdmNormalizer(
        env,
        vectorized,
        max_tau=max_tau,
        **variant['tdm_normalizer_kwargs']
    )
    qf = TdmQf(
        env=env,
        vectorized=vectorized,
        tdm_normalizer=tdm_normalizer,
        **variant['qf_kwargs']
    )
    vf = TdmVf(
        env=env,
        vectorized=vectorized,
        tdm_normalizer=tdm_normalizer,
        **variant['vf_kwargs']
    )
    policy = StochasticTdmPolicy(
        env=env,
        tdm_normalizer=tdm_normalizer,
        **variant['policy_kwargs']
    )
    replay_buffer = HerReplayBuffer(
        env=env,
        **variant['her_replay_buffer_kwargs']
    )
    algorithm = TdmSac(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        replay_buffer=replay_buffer,
        **variant['sac_tdm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-sac-tdm-launch"

    n_seeds = 1
    # mode = "ec2"
    exp_prefix = "sac-tdm-point2d"

    num_epochs = 100
    num_steps_per_epoch = 100
    num_steps_per_eval = 100
    max_path_length = 50

    # noinspection PyTypeChecker
    variant = dict(
        sac_tdm_kwargs=dict(
            base_kwargs=dict(
                num_epochs=num_epochs,
                num_steps_per_epoch=num_steps_per_epoch,
                num_steps_per_eval=num_steps_per_eval,
                max_path_length=max_path_length,
                num_updates_per_env_step=25,
                batch_size=128,
                discount=1,
                save_replay_buffer=False,
                render=True,
            ),
            tdm_kwargs=dict(
                sample_rollout_goals_from='fixed',
                sample_train_goals_from='her',
                vectorized=True,
                norm_order=2,
                cycle_taus_for_rollout=True,
                max_tau=10,
                square_distance=True,
            ),
            sac_kwargs=dict(
                soft_target_tau=0.01,
                policy_lr=3E-4,
                qf_lr=3E-4,
                vf_lr=3E-4,
            ),
            give_terminal_reward=False,
        ),
        her_replay_buffer_kwargs=dict(
            max_size=int(1E6),
            num_goals_to_sample=4,
        ),
        qf_kwargs=dict(
            hidden_sizes=[32, 32],
            structure='squared_difference',
            # structure='norm_difference',
            # structure='none',
            learn_offset=True,
        ),
        vf_kwargs=dict(
            hidden_sizes=[32, 32],
        ),
        policy_kwargs=dict(
            hidden_sizes=[32, 32],
        ),
        tdm_normalizer_kwargs=dict(
            normalize_tau=False,
            log_tau=False,
        ),
        env_kwargs=dict(),
        version="SAC-TDM",
        algorithm="SAC-TDM",
    )
    search_space = {
        'env_class': [
            # GoalXVelHalfCheetah,
            # Reacher7DofXyzGoalState,
            # Reacher7DofFullGoal,
            # MultitaskPoint2DEnv,
            MultitaskPoint2dWall,
            # MultitaskPoint2dUWall,
            # GoalXYPosAnt,
            # Walker2DTargetXPos,
            # MultitaskPusher3DEnv,
            # CylinderXYPusher2DEnv,
        ],
        # 'env_kwargs': [
        #     dict(max_distance=6),
        # ],
        'sac_tdm_kwargs.base_kwargs.reward_scale': [
            # 1,
            # 10,
            100,
            # 1000,
            # 10000,
        ],
        'qf_kwargs.hidden_activation': [
            ptu.softplus,
            # tanh
        ],
        'sac_tdm_kwargs.tdm_kwargs.vectorized': [
            # False,
            True,
        ],
        # 'sac_tdm_kwargs.give_terminal_reward': [
        #     False,
        #     True,
        # ],
        'sac_tdm_kwargs.tdm_kwargs.terminate_when_goal_reached': [
            # True,
            False,
        ],
        'sac_tdm_kwargs.tdm_kwargs.sample_rollout_goals_from': [
            # 'fixed',
            'environment',
            # 'replay_buffer',
        ],
        'sac_tdm_kwargs.tdm_kwargs.max_tau': [
            0,
            # 10,
            # 25,
            # max_path_length-1,
            # 1,
            # 10,
            # 99,
            # 49,
            # 15,
        ],
        'sac_tdm_kwargs.base_kwargs.num_updates_per_env_step': [
            1,
            # 5,
            # 10,
            # 25,
        ],
        'sac_tdm_kwargs.base_kwargs.discount': [
            1,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if (
            not variant['sac_tdm_kwargs']['tdm_kwargs']['terminate_when_goal_reached']
            and variant['sac_tdm_kwargs']['give_terminal_reward']
        ):
            continue
        for i in range(n_seeds):
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                variant=variant,
                exp_id=exp_id,
            )
