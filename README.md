# rail-rl
Reinforcement learning algorithms implemented for the RAIL lab at UC Berkeley.
So far, DDPG is implemented and NAF is a work in progress.

## Installation
This library requires rllab to be installed. See [rllab's installation docs](https://rllab.readthedocs.io/en/latest/user/installation.html).

## Usage
You can use the `main.py` script to launch different experiments:

```
$ python main.py --env=cheetah --algo=ddpg --name=my_amazing_experiment
```

`example.py` gives a minimal example of how to write a script that launches experiments.

### Creating your own policy/critic
See `policies/nn_policy.py`. You only need to implement

```
def create_network(self):
    """
    Use self.observations_placeholder to create an output. (To be
    refactored soon so that the input is passed in.)

    :return: TensorFlow tensor.
    """
    raise NotImplementedError
```

For example, to create a simple policy that just sums its observations:

```
class SumPolicy(NNPolicy):
    def create_network(self):
        W_obs = weight_variable((self.observation_dim, 1),
                                initializer=tf.constant_initializer(1.))
        return tf.matmul(self.observations_placeholder, W_obs)
```

For implementing a critic, see `qfunctions/nn_qfunction.py`. Another simple critic that simply sums its input is:
```
class SumCritic(NNCritic):
    """Just output the sum of the inputs. This is used to debug."""

    def create_network(self, action_input):
        with tf.variable_scope("actions_layer") as _:
            W_actions = weight_variable(
                (self.action_dim, 1),
                initializer=tf.constant_initializer(1.),
                reuse_variables=True)
        with tf.variable_scope("observation_layer") as _:
            W_obs = weight_variable(
                (self.observation_dim, 1),
                initializer=tf.constant_initializer(1.),
                reuse_variables=True)

        return (tf.matmul(action_input, W_actions) +
                tf.matmul(self.observations_placeholder, W_obs))
```