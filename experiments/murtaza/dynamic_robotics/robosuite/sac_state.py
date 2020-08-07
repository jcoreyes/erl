import argparse
from rlkit.envs.robosuite_wrapper import RobosuiteStateWrapperEnv
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
import rlkit.misc.hyperparameter as hyp
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import rlkit.torch.pytorch_util as ptu

def experiment(variant):
    env = RobosuiteStateWrapperEnv(wrapped_env_id=variant['env_id'], **variant['env_kwargs']) #

    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    hidden_sizes = variant['hidden_sizes']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes,
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
    )

    es = OUStrategy(action_space=env.action_space, max_sigma=variant['exploration_noise'],
                    min_sigma=variant['exploration_noise'])

    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        env,
        exploration_policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        env,
    )
    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=variant['max_path_length'],
        batch_size=variant['batch_size'],
        num_epochs=variant['num_epochs'],
        num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
        num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
        num_trains_per_train_loop=variant['num_trains_per_train_loop'],
        min_num_steps_before_training=variant['min_num_steps_before_training'],
    )
    algorithm.to(ptu.device)
    algorithm.train()


variant = dict(
    num_epochs=1000000,
    num_eval_steps_per_epoch=1000,
    num_trains_per_train_loop=1000,
    num_expl_steps_per_train_loop=1000,
    min_num_steps_before_training=128,
    max_path_length=200,
    batch_size=128,
    replay_buffer_size=int(1E6),
    hidden_sizes=[400, 300],
    algorithm="Twin-SAC",
    version="normal",

    trainer_kwargs=dict(
        discount=0.99,
        policy_lr=1e-3,
        qf_lr=1e-3,
        reward_scale=1,
        soft_target_tau=1e-3,  # 1e-2
        target_update_period=1,  # 1
        use_automatic_entropy_tuning=True,
    ),

    exploration_noise=0,
    env_kwargs=dict(
        has_renderer=False,  # no on-screen renderer
        has_offscreen_renderer=False,  # no off-screen renderer
        use_object_obs=True,  # use object-centric feature
        use_camera_obs=False,  # no camera observations
        reward_shaping=True,
    ),
    env_id="SawyerLift",
)

common_params = {
    'env_id':['SawyerLift', 'SawyerStack', 'SawyerPickPlace', 'SawyerNutAssembly'],
    'exploration_noise':[0, .1, .3, .5, .8]
}

env_params = {
    'robosuite': {
    },
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        default='robosuite')
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # noinspection PyTypeChecker
    args = parse_args()

    exp_prefix = "sac-" + args.env
    if len(args.label) > 0:
        exp_prefix = exp_prefix + "-" + args.label

    search_space = common_params
    search_space.update(env_params[args.env])
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    if args.mode == 'gcp' and args.gpu:
        num_exps_per_instance = args.num_seeds//2
        num_outer_loops = args.num_seeds//2+1
    else:
        num_exps_per_instance = 1
        num_outer_loops = args.num_seeds

    for _ in range(num_outer_loops):
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            if variant['env_id'] == 'SawyerLift':
                variant['max_path_length'] = 200
            elif variant['env_id']  == 'SawyerStack':
                variant['max_path_length'] = 500
            elif variant['env_id']  == 'SawyerPickPlace':
                variant['max_path_length'] = 2000
            elif variant['env_id']  == 'SawyerNutAssembly':
                variant['max_path_length'] = 2000
            else:
                raise EnvironmentError()
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=args.mode,
                exp_id=exp_id,
                variant=variant,
                use_gpu=args.gpu,
                num_exps_per_instance=num_exps_per_instance,
                snapshot_mode='last',
                gcp_kwargs=dict(
                    preemptible=False,
                    zone='us-east1-c',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-k80',
                        num_gpu=1,
                    )
                ),
            )
