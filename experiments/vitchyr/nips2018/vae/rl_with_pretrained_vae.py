import railrl.misc.hyperparameter as hyp
from railrl.torch.vae.sim_vae_policy import dump_video
from railrl.core import logger
import os.path as osp
import railrl.torch.pytorch_util as ptu
from railrl.envs.multitask.multitask_env import MultitaskToFlatEnv
from railrl.envs.multitask.point2d import MultitaskImagePoint2DEnv
from railrl.envs.mujoco.sawyer_gripper_env import SawyerXYEnv
from railrl.envs.vae_wrappers import VAEWrappedImageGoalEnv, VAEWrappedEnv
from railrl.envs.wrappers import ImageMujocoEnv
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
from railrl.torch.td3.td3 import TD3


def experiment(variant):
    from path import Path
    import joblib
    vae_dir = Path(variant['vae_dir'])
    vae_path = str(vae_dir / 'params.pkl')
    extra_data_path = str(vae_dir / 'extra_data.pkl')
    extra_data = joblib.load(extra_data_path)
    if 'env' in extra_data:
        env = extra_data['env']
    else:
        env = SawyerXYEnv()
        from railrl.images.camera import sawyer_init_camera
        env = ImageMujocoEnv(
            env, 84,
            transpose=True,
            init_camera=sawyer_init_camera,
            normalize=True,
        )
    use_env_goals = variant["use_env_goals"]
    render = variant["render"]

    if use_env_goals:
        env = VAEWrappedImageGoalEnv(
            env,
            vae_path,
            render_goals=render,
            render_rollouts=render,
            **variant['vae_wrapped_env_kwargs']
        )
    else:
        env = VAEWrappedEnv(
            env,
            vae_path,
            render_goals=render,
            render_rollouts=render,
            **variant['vae_wrapped_env_kwargs']
        )

    env = MultitaskToFlatEnv(env)
    if variant['normalize']:
        env = NormalizedBoxEnv(env)
    exploration_type = variant['exploration_type']
    if exploration_type == 'ou':
        es = OUStrategy(action_space=env.action_space)
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            max_sigma=0.1,
            min_sigma=0.1,  # Constant sigma
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            prob_random_action=0.1,
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )
    qf2 = FlattenMlp(
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
        training_env=env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
        env._wrapped_env.vae.cuda()
    else:
        env._wrapped_env.vae.cpu()
    logdir = logger.get_snapshot_dir()
    filename = osp.join(logdir, 'video_0.mp4')
    dump_video(env, policy, filename)

    algorithm.train()

    logdir = logger.get_snapshot_dir()
    filename = osp.join(logdir, 'video_final.mp4')
    dump_video(env, policy, filename)


if __name__ == "__main__":
    vae_latent_size_to_dir = {
        2: '/home/vitchyr/git/railrl/data/local/04-25-sawyer-vae-local-working/04-25-sawyer-vae-local-working_2018_04_25_18_06_29_0000--s-93337/',
        4: '/home/vitchyr/git/railrl/data/local/04-25-sawyer-vae-local-working/04-25-sawyer-vae-local-working_2018_04_25_18_20_26_0000--s-96538/',
        8: '/home/vitchyr/git/railrl/data/local/04-25-sawyer-vae-local-working/04-25-sawyer-vae-local-working_2018_04_25_18_24_44_0000--s-200/',
        16: '/home/vitchyr/git/railrl/data/local/04-25-sawyer-vae-local-working/04-25-sawyer-vae-local-working_2018_04_25_18_29_12_0000--s-81838/',
    }
    variant = dict(
        algo_kwargs=dict(
            num_epochs=500,
            num_steps_per_epoch=100,
            num_steps_per_eval=100,
            # num_steps_per_epoch=100,
            # num_steps_per_eval=100,
            tau=1e-2,
            batch_size=128,
            max_path_length=100,
            discount=0.99,
        ),
        env_kwargs=dict(
            # render_onscreen=False,
            # render_size=84,
            # ignore_multitask_goal=True,
            # ball_radius=1,
        ),
        vae_wrapped_env_kwargs=dict(
            track_qpos_goal=3,
            use_vae_obs=True,
            use_vae_reward=True,
            use_vae_goals=True,
        ),
        algorithm='TD3',
        normalize=False,
        rdim=4,
        render=False,
        # env=MultitaskImagePoint2DEnv,
        env=SawyerXYEnv,
        use_env_goals=True,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-rl-with-pretrained-vae'

    # n_seeds = 3
    # mode = 'ec2'
    exp_prefix = 'sawyer-rl-with-pretrained-vae-local-4'

    search_space = {
        'exploration_type': [
            'ou',
        ],
        'algo_kwargs.reward_scale': [1],
        'vae_latent_size': [4, 8, 16, 2],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant['vae_dir'] = vae_latent_size_to_dir[variant['vae_latent_size']]
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
            )
