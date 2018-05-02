import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.networks import CNNPolicy, MergedCNN
from railrl.torch.modules import HuberLoss
from railrl.torch.ddpg.ddpg import DDPG
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
import torch
from railrl.torch.td3.td3 import TD3
from sawyer_control.sawyer_reaching import SawyerXYZReachingEnv
from sawyer_control.sawyer_image import ImageSawyerEnv

def experiment(variant):

    imsize = variant['imsize']
    history = variant['history']
    env_params = variant['env_params']
    env = SawyerXYZReachingEnv(**env_params)
    env = NormalizedBoxEnv(ImageSawyerEnv(env,
                                    imsize=imsize,
                                    keep_prev=history-1,))

    es = OUStrategy(action_space=env.action_space)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    qf1 = MergedCNN(input_width=imsize,
                   input_height=imsize,
                   output_size=1,
                   input_channels= 3 * history,
                   added_fc_input_size=action_dim,
                   **variant['cnn_params'])

    qf2 = MergedCNN(input_width=imsize,
                   input_height=imsize,
                   output_size=1,
                   input_channels=3 * history,
                   added_fc_input_size=action_dim,
                   **variant['cnn_params'])


    policy = CNNPolicy(input_width=imsize,
                       input_height=imsize,
                       output_size=action_dim,
                       input_channels=3 * history,
                       **variant['cnn_params'],
                       output_activation=torch.tanh,
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
    if ptu.gpu_enabled():
        algorithm.cuda()

    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        imsize=84,
        history=1,
        algo_kwargs=dict(
            num_epochs=50,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=64,
            max_path_length=100,
            discount=0.99,
            train_on_eval_paths=True,
            replay_buffer_size=int(1E6),
            normalize_env=False,
        ),
        cnn_params=dict(
            kernel_sizes=[5, 5],
            n_channels=[16, 32],
            strides=[2, 2],
            pool_sizes=[1, 1],
            hidden_sizes=[100, 100],
            paddings=[0, 0],
            use_batch_norm=False,
        ),
        env_params=dict(
            action_mode='torque',
        ),

        algo_class=DDPG,
        qf_criterion_class=HuberLoss,
    )
    search_space = {
        'qf_criterion_class': [
            HuberLoss,
        ],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            run_experiment(
                experiment,
                variant=variant,
                exp_id=exp_id,
                exp_prefix="test",
                mode='here_no_doodad',
                use_gpu=True,
            )
