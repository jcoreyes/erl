import tensorflow as tf
import numpy as np

def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate
                 ):
        """ YOUR CODE HERE """

    def fit(self, data):
        """
        Write a function to take in a dataset of states, actions, next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """

    def predict(self, states, actions):
        """ Write a function to take in a batch of states and actions and return the next states as predicted by using the model """
        """ YOUR CODE HERE """
