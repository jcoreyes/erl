import argparse

import joblib
import tensorflow as tf

from controllers_solution import MPCcontroller
from dynamics_solution import NNDynamicsModel
from main_solution import sample
from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    args = parser.parse_args()

    with tf.Session() as sess:
        data = joblib.load(args.file)
        env = data['env']
        model_params = data['model_params']
        mpc_params = data['mpc_params']
        dyn_model = NNDynamicsModel(
            env=env,
            **model_params)
        dyn_model.sess = sess
        mpc_params = dict(
            horizon=15,
            cost_fn=env.cost_fn,
            num_simulated_paths=10000,
        )
        mpc_controller = MPCcontroller(env=env,
                                       dyn_model=dyn_model,
                                       **mpc_params)
        tf.global_variables_initializer().run()
        """
        Right now, the network parameters aren't saved, so this is just a 
        random network.
        """
        while True:
            new_paths = sample(env=env,
                               controller=mpc_controller,
                               num_paths=10,
                               horizon=50,
                               render=True,
                               verbose=True)
            if not query_yes_no('Continue simulation?'):
                break
