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

## Creating your own policy/critic
To create a policy, you need to create a class that subclasses ``NNPolicy`` and implements

```
def create_network(self):
    """
    Use self.observations_placeholder to create an output. (To be
    refactored soon so that the input is passed in.)

    :return: TensorFlow tensor.
    """
    raise NotImplementedError
```

and similarly for creating your own critic. See `policies/nn_policy.py` and `qfunctions/nn_qfunction.py` for detail.

For an example on creating your own policy, see the `SumPolicy` class in `policies/nn_policy.py`.
For an example on creating your own critic, see the `SumCritic` class in `qfunctions/nn_qfunction.py`.

## Visualizing a policy and seeing results
During training, the results will be saved to a file called under
```
PROJECT_PATH/data/local/<default>/<foldername>
```
 -  `PROJECT_PATH` is the directory set by `rllab.config.PROJECT_PATH`. By default the project path is your rllab directory.
 - For example.py `<default>` is `ddpg-half-cheetah`, and can be changed by passing an argument to `exp_prefix` when calling `run_experiment_lite`.
 - `<foldername>` is auto-generated and based off of what `<default>` is.
 - inside this folder, you should see a file called `params.pkl`. To visualize a policy, run

```
$ python sim_policy PROJECT_PATH/data/local/<default>/<foldername>/params.pkl
```

## FAQs
_I'm getting issues about a session being None. (e.g. "None has not method called run")._
This might happen if you run in stub mode and are unserializing a network.
This error is because the un-serialization code expects a default Tensorflow Session to exist.
Fix this by creating a default graph:
```python
with tf.Session().as_default():
    # the rest of your code
```


