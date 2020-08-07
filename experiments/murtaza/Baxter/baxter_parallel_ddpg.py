from rlkit.envs.remote import RemoteRolloutEnv
from rlkit.envs.wrappers import convert_gym_space
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import FeedForwardQFunction, FeedForwardPolicy
from rlkit.torch.algos.parallel_ddpg import ParallelDDPG
from rlkit.envs.ros.baxter_env import BaxterEnv
from rlkit.exploration_strategies.ou_strategy import OUStrategy
import sys
from rlkit.launchers.launcher_util import continue_experiment
from rlkit.launchers.launcher_util import resume_torch_algorithm
from rllab.envs.normalized_env import normalize

def example(variant):
    env_class = variant['env_class']
    env_params = variant['env_params']
    env = env_class(**env_params)
    obs_space = convert_gym_space(env.observation_space)
    action_space = convert_gym_space(env.action_space)
    es_class = variant['es_class']
    es_params = es_params = dict(
        action_space=action_space,
    )
    policy_class = variant['policy_class']
    use_gpu = variant['use_gpu']

    if variant['normalize_env']:
        env = normalize(env)

    es = es_class(**es_params)
    qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        100,
        100,
    )
    policy_params = dict(
        obs_dim=int(obs_space.flat_dim),
        action_dim=int(action_space.flat_dim),
        fc1_size=100,
        fc2_size=100,
    )
    policy = policy_class(**policy_params)
    remote_env = RemoteRolloutEnv(
        env_class,
        env_params,
        policy_class,
        policy_params,
        es_class,
        es_params,
        variant['max_path_length'],
        variant['normalize_env'],
    )

    algorithm =ParallelDDPG(
        remote_env,
        exploration_strategy=es,
        qf=qf,
        policy=policy,
        **variant['algo_params'],
    )
    if use_gpu:
        algorithm.cuda()
    algorithm.train()

experiments=[
    'joint_angle|fixed_angle',
    'joint_angle|varying_angle',
    'end_effector_position|fixed_ee',
    'end_effector_position|varying_ee',
    'end_effector_position_orientation|fixed_ee',
    'end_effector_position_orientation|varying_ee'
]
max_path_length = 1000
left_exp = dict(
        example=example,
        exp_prefix="ddpg-baxter-left-arm-increased-reward-magnitude",
        seed=0,
        mode='here',
        variant={
            'version': 'Original',
            'arm_name':'left',
            'safety_box':False,
            'loss':'huber',
            'huber_delta':10,
            'safety_force_magnitude':1,
            'temp':1.2,
            'remove_action':False,
            'experiment':experiments[0],
            'es_min_sigma':.1,
            'es_max_sigma':.1,
            'num_epochs':30,
            'batch_size':1024,
            'use_gpu':True,
            'include_torque_penalty': False,
            'number_of_gradient_steps': 1,
            'reward_magnitude':10,
        },
        use_gpu=True,
    )
right_exp = dict(
    example=example,
    exp_prefix="ddpg-parallel-baxter-right-arm",
    seed=0,
    mode='here',
    variant={
        'version': 'Original',
        'max_path_length':max_path_length,
        'use_gpu': True,
        'es_class':OUStrategy,
        'env_class':BaxterEnv,
        'policy_class':FeedForwardPolicy,
        'normalize_env':False,
        'env_params':{
            'arm_name': 'right',
            'safety_box': False,
            'loss': 'huber',
            'huber_delta': 10,
            'safety_force_magnitude': 1,
            'temp': 1.2,
            'remove_action': False,
            'experiment': experiments[0],
            'reward_magnitude': 1,
            'include_torque_penalty': False,
        },
        'algo_params':dict(
            batch_size=1024,
            num_epochs=30,
            number_of_gradient_steps=1,
        ),
    },
    use_gpu=True,
)

if __name__ == "__main__":
    try:
        exp_dir = sys.argv[1]
    except:
        exp_dir = None

    dictionary = right_exp
    if exp_dir == None:
        run_experiment(
            dictionary['example'],
            exp_prefix=dictionary['exp_prefix'],
            seed=dictionary['seed'],
            mode=dictionary['mode'],
            variant=dictionary['variant'],
            use_gpu=dictionary['use_gpu'],
        )
    else:
        continue_experiment(exp_dir, resume_torch_algorithm)

