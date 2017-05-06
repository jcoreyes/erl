# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Basic linear combinations that implicitly generate variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.rnn import RNNCell
import tensorflow as tf

class BasicLSTMCell_LayerNorm(RNNCell):
    """Basic LSTM recurrent network cell.
  
    The implementation is based on: http://arxiv.org/pdf/1409.2329v5.pdf.
  
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
  
    Biases of the forget gate are initialized by default to 1 in order to reduce
    the scale of forgetting in the beginning of the training.
    """

    def __init__(self, num_units, forget_bias = 1.0, gpu_for_layer = 0, weight_initializer = "uniform_unit", orthogonal_scale_factor = 1.1, use_highway = False, num_highway_layers = 2,
                 use_recurrent_dropout = False, recurrent_dropout_factor = 0.90, is_training = True):
        self._num_units = num_units
        self._gpu_for_layer = gpu_for_layer
        self._weight_initializer = weight_initializer
        self._orthogonal_scale_factor = orthogonal_scale_factor
        self._forget_bias = forget_bias
        self.use_highway = use_highway
        self.num_highway_layers = num_highway_layers
        self.use_recurrent_dropout = use_recurrent_dropout
        self.recurrent_dropout_factor = recurrent_dropout_factor
        self.is_training = is_training

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return 2 * self._num_units

    def __call__(self, inputs, state, timestep = 0, scope=None):
        with tf.device("/gpu:"+str(self._gpu_for_layer)):
            """Long short-term memory cell (LSTM)."""
            with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
                # Parameters of gates are concatenated into one multiply for efficiency.
                h, c = tf.split(1, 2, state)

                concat = linear([inputs, h], self._num_units * 4, False, 0.0)

                concat = layer_norm(concat, num_variables_in_tensor = 4)

                # i = input_gate, j = new_input, f = forget_gate, o = output_gate
                i, j, f, o = tf.split(1, 4, concat)

                if self.use_recurrent_dropout and self.is_training:
                    input_contribution = tf.nn.dropout(tf.tanh(j), self.recurrent_dropout_factor)
                else:
                    input_contribution = tf.tanh(j)

                new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * input_contribution
                with tf.variable_scope('new_h_output'):
                    new_h = tf.tanh(layer_norm(new_c)) * tf.sigmoid(o)

            return new_h, tf.concat(1, [new_h, new_c]) #purposely reversed



use_weight_normalization_default = False
def linear(args, output_size, bias, bias_start=0.0, use_l2_loss = False, use_weight_normalization = use_weight_normalization_default, scope=None, timestep = -1, weight_initializer = None, orthogonal_scale_factor = 1.1):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  # assert args #was causing error in upgraded tensorflow
  if not isinstance(args, (list, tuple)):
    args = [args]

  if len(args) > 1 and use_weight_normalization: raise ValueError('you can not use weight_normalization with multiple inputs because the euclidean norm will be incorrect -- besides, you should be using multiple integration instead!!!')

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  if use_l2_loss:
    l_regularizer = tf.contrib.layers.l2_regularizer(1e-5)
  else:
    l_regularizer = None

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size],
                      initializer = tf.uniform_unit_scaling_initializer(), regularizer = l_regularizer)
    # if use_weight_normalization: matrix = weight_normalization(matrix, timestep = timestep)

    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(1, args), matrix)

    if not bias:
      return res
    bias_term = tf.get_variable("Bias", [output_size],
                                initializer=tf.constant_initializer(bias_start), regularizer = l_regularizer)

  return res + bias_term


def batch_timesteps_linear(input, output_size, bias, bias_start=0.0, use_l2_loss = False, use_weight_normalization = use_weight_normalization_default, scope=None,
  tranpose_input = True, timestep = -1):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 3D Tensor [timesteps, batch_size, input_size]
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  # Calculate the total size of arguments on dimension 2.
  if tranpose_input:
    input = tf.transpose(input, [1,0,2])

  shape_list = input.get_shape().as_list()
  if len(shape_list) != 3: raise ValueError('shape must be of size 3, you have inputted shape size of:', len(shape_list))

  num_timesteps = shape_list[0]
  batch_size = shape_list[1]
  total_arg_size = shape_list[2]

  if use_l2_loss:
    l_regularizer = tf.contrib.layers.l2_regularizer(1e-5)
  else:
    l_regularizer = None

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [total_arg_size, output_size], initializer = tf.uniform_unit_scaling_initializer(), regularizer = l_regularizer)
    # if use_weight_normalization: matrix = weight_normalization(matrix)
    matrix = tf.tile(tf.expand_dims(matrix, 0), [num_timesteps, 1, 1])

    res = tf.batch_matmul(input, matrix)

    if bias:
      bias_term = tf.get_variable(
          "Bias", [output_size],
          initializer=tf.constant_initializer(bias_start))
      res = res + bias_term

  if tranpose_input:
    res = tf.transpose(res, [1,0,2])

  return res

def layer_norm(input_tensor, num_variables_in_tensor = 1, initial_bias_value = 0.0, scope = "layer_norm"):
  with tf.variable_scope(scope):
    '''for clarification of shapes:
    input_tensor = [batch_size, num_neurons]
    mean = [batch_size]
    variance = [batch_size]
    alpha = [num_neurons]
    bias = [num_neurons]
    output = [batch_size, num_neurons]
    '''
    input_tensor_shape_list = input_tensor.get_shape().as_list()

    num_neurons = input_tensor_shape_list[1]/num_variables_in_tensor



    alpha = tf.get_variable('layer_norm_alpha', [num_neurons * num_variables_in_tensor],
            initializer = tf.constant_initializer(1.0))

    bias = tf.get_variable('layer_norm_bias', [num_neurons * num_variables_in_tensor],
            initializer = tf.constant_initializer(initial_bias_value))

    if num_variables_in_tensor == 1:
      input_tensor_list = [input_tensor]
      alpha_list = [alpha]
      bias_list = [bias]

    else:
      input_tensor_list = tf.split(1, num_variables_in_tensor, input_tensor)
      alpha_list = tf.split(0, num_variables_in_tensor, alpha)
      bias_list = tf.split(0, num_variables_in_tensor, bias)

    list_of_layer_normed_results = []
    for counter in range(num_variables_in_tensor):
      mean, variance = moments_for_layer_norm(input_tensor_list[counter], axes = [1], name = "moments_loopnum_"+str(counter)+scope) #average across layer

      output =  (alpha_list[counter] * (input_tensor_list[counter] - mean)) / variance + bias[counter]

      list_of_layer_normed_results.append(output)

    if num_variables_in_tensor == 1:
      return list_of_layer_normed_results[0]
    else:
      return tf.concat(1, list_of_layer_normed_results)


def moments_for_layer_norm(x, axes = 1, name = None, epsilon = 0.001):
  '''output for mean and variance should be [batch_size]'''

  if not isinstance(axes, list): axes = list(axes)

  with tf.op_scope([x, axes], name, "moments"):
    mean = tf.reduce_mean(x, axes, keep_dims = True)

    variance = tf.sqrt(tf.reduce_mean(tf.square(x - mean), axes, keep_dims = True) + epsilon)

    return mean, variance


