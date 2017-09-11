from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG
from os.path import exists
from railrl.envs.ros.baxter_env import BaxterEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.torch import pytorch_util as ptu
import joblib
import sys
from railrl.launchers.launcher_util import continue_experiment
from railrl.launchers.launcher_util import resume_torch_algorithm
from rllab.envs.normalized_env import normalize

def example(variant):
    load_policy_file = variant.get('load_policy_file', None)
    if not load_policy_file == None and exists(load_policy_file):
        data = joblib.load(load_policy_file)
        algorithm = data['algorithm']
        epochs = data['epoch']
        use_gpu = variant['use_gpu']
        if use_gpu and ptu.gpu_enabled():
            algorithm.cuda()
        algorithm.train(start_epoch=epochs)
    else:
        arm_name = variant['arm_name']
        experiment = variant['experiment']
        loss = variant['loss']
        huber_delta= variant['huber_delta']
        safety_box = variant['safety_box']
        remove_action = variant['remove_action']
        safety_force_magnitude = variant['safety_force_magnitude']
        temp = variant['temp']
        es_min_sigma = variant['es_min_sigma']
        es_max_sigma = variant['es_max_sigma']
        num_epochs = variant['num_epochs']
        batch_size = variant['batch_size']
        use_gpu = variant['use_gpu']
        include_torque_penalty = variant['include_torque_penalty']
        number_of_gradient_steps = variant['number_of_gradient_steps']
        reward_magnitude = variant['reward_magnitude']
        env = BaxterEnv(
            experiment=experiment,
            arm_name=arm_name,
            loss=loss,
            safety_box=safety_box,
            remove_action=remove_action,
            safety_force_magnitude=safety_force_magnitude,
            temp=temp,
            huber_delta=huber_delta,
            include_torque_penalty=include_torque_penalty,
            reward_magnitude=reward_magnitude,
        )
        es = OUStrategy(
            max_sigma=es_max_sigma,
            min_sigma=es_min_sigma,
            action_space=env.action_space,
        )
        qf = FeedForwardQFunction(
            int(env.observation_space.flat_dim),
            int(env.action_space.flat_dim),
            100,
            100,
        )
        policy = FeedForwardPolicy(
            int(env.observation_space.flat_dim),
            int(env.action_space.flat_dim),
            100,
            100,
        )
        algorithm = DDPG(
            env,
            qf,
            policy,
            es,
            num_epochs=num_epochs,
            batch_size=batch_size,
            number_of_gradient_steps=number_of_gradient_steps,
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
    exp_prefix="ddpg-baxter-right-arm-merged-test",
    seed=0,
    mode='here',
    variant={
        'version': 'Original',
        'arm_name': 'right',
        'safety_box': False,
        'loss': 'huber',
        'huber_delta': 10,
        'safety_force_magnitude': 1,
        'temp': 1.2,
        'remove_action': False,
        'experiment': experiments[0],
        'es_min_sigma': .1,
        'es_max_sigma': .1,
        'num_epochs': 30,
        'batch_size': 1024,
        'use_gpu': True,
        'include_torque_penalty': False,
        'number_of_gradient_steps': 10,
        'reward_magnitude': 1,
    },
    use_gpu=True,
)

if __name__ == "__main__":
    try:
        exp_dir = sys.argv[1]
    except:
        exp_dir = None

    dictionary = left_exp
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

