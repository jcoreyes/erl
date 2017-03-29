"""
:author: Vitchyr Pong
"""
import tensorflow as tf

from railrl.algos.bptt_ddpg import BpttDDPG
from railrl.core.rnn.rnn import OutputStateRnn


class SumBpttDDPG(BpttDDPG):
    """
    BpttDDPT but sum the Q values as you unroll the policy.
    """

    def _init_policy_ops(self):
        self._rnn_inputs_ph = tf.placeholder(
            tf.float32,
            [None, self._num_bptt_unrolls, self._env_obs_dim],
            name='rnn_time_inputs',
        )
        all_env_obs = tf.unstack(self._rnn_inputs_ph, axis=1)
        self._rnn_init_state_ph = self.policy.get_init_state_placeholder()

        self._rnn_cell_scope.reuse_variables()
        wrapper_rnn_cell = OutputStateRnn(self._rnn_cell)
        with tf.variable_scope(self._rnn_cell_scope):
            self._rnn_outputs, self._rnn_final_state = (
                tf.contrib.rnn.static_rnn(
                    wrapper_rnn_cell,
                    all_env_obs,
                    initial_state=self._rnn_init_state_ph,
                    dtype=tf.float32,
                    scope=self._rnn_cell_scope,
                )
            )
        all_env_actions, all_write_actions = zip(*self._rnn_outputs)
        """
        Creating all_rnn_observations is a bit tricky. Let X be shorthand for
        the memories in all_rnn_observations. We want
        
            X[0] = batch element 0, T = 0 <-- from self._rnn_init_state_ph
            X[1] = batch element 0, T = 1 <-- from all_write_actions[0]
            X[2] = batch element 0, T = 2 <-- from all_write_actions[1]
            ...
            X[N-1] = batch element 0, T = N-1 <-- from self._rnn_init_state_ph
            X[N] = batch element 1, T = 0     <-- from all_write_actions[0]
            X[N+1] = batch element 1, T = 1   <-- from all_write_actions[0]
            ...
            
        Basically, for each element in the batch, the first element needs to 
        come from `self._rnn_init_state_ph`, while the next N-1 elements need to
        come from all_write_actions. 
        
        Let N be self._num_bptt_unrolls.
        all_write_actions is a list of N tensors, each w/ shape [None x memory].
        
        Here's how I do this:
          1. Convert the first N-1 all_write_actions into a tensor of shape
              [None x N-1 x memory]
          2. Convert `self._rnn_init_state_ph` into shape [None x 1 x memory]
          3. Combine the above two into a tensor of shape [None x N x memory]
          4. Convert this into a list of N tensors, each w/ shape [None x mem]
        """
        memory_dim = self.observation_dim - self._env_obs_dim

        # Step 1
        outputted_memories = tf.stack(all_write_actions[:-1], axis=1)
        outputted_memories.get_shape().assert_is_compatible_with(
            (0, self._num_bptt_unrolls - 1, memory_dim)
        )

        # Step 2
        expanded_init = tf.expand_dims(input=self._rnn_init_state_ph, axis=1)
        expanded_init.get_shape().assert_is_compatible_with(
            (0, 1, memory_dim)
        )

        # Step 3
        all_memory_obs = tf.concat(
            axis=1,
            values=[
                expanded_init,
                outputted_memories,
            ],
        )
        all_memory_obs.get_shape().assert_is_compatible_with(
            (0, self._num_bptt_unrolls, memory_dim)
        )

        # Step 4
        all_memories = tf.unstack(all_memory_obs, axis=1)

        all_rnn_observations = (
            tf.concat(axis=0, values=all_env_obs),
            tf.concat(axis=0, values=all_memories),
        )

        all_rnn_actions = (
            tf.concat(axis=0, values=all_env_actions),
            tf.concat(axis=0, values=all_write_actions),
        )

        self.qf_with_action_input = self.qf.get_weight_tied_copy(
            action_input=all_rnn_actions,
            observation_input=all_rnn_observations,
        )
        self.policy_surrogate_loss = - tf.reduce_mean(
            self.qf_with_action_input.output)
        self.train_policy_op = tf.train.AdamOptimizer(
            self.policy_learning_rate).minimize(
            self.policy_surrogate_loss,
            var_list=self.policy.get_params_internal())

    def _policy_feed_dict(self, obs, **kwargs):
        """
        :param obs: See output of `self._split_flat_action`.
        :return: Feed dictionary for policy training TensorFlow ops.
        """
        _, initial_memory_obs = self._get_time_step(obs, 0)
        env_obs, _ = obs
        return {
            self._rnn_inputs_ph: env_obs,
            self._rnn_init_state_ph: initial_memory_obs,
        }
