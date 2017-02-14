# rail-rl
Reinforcement learning algorithms implemented for the RAIL lab at UC Berkeley.
So far, DDPG is implemented and NAF is a work in progress.

## Installation
This library requires rllab to be installed. See [rllab's installation docs](https://rllab.readthedocs.io/en/latest/user/installation.html).

One important difference: use my anaconda environment instead of rllab's conda environment. It can be installed and used with
```
$ conda env create -f railrl-env.yml
$ source activate railrl
(railrl) $ # Ready to run example.py
```

## Usage
You can use the `main.py` script to launch different experiments:

```
(railrl) $ python main.py --env=cheetah --algo=ddpg --name=my_amazing_experiment
```

`example.py` gives a minimal example of how to write a script that launches experiments.

## Creating your own policy/critic
To create a policy, you need to create a class that subclasses ``NNPolicy`` and implements

```
def _create_network_internal(self, observation_input=None):
```

and similarly for creating your own critic.

See `policies/nn_policy.py` and `qfunctions/nn_qfunction.py` for examples.

There are calls to `_process_layer` and `_add_network_and_get_output`.
If you plan on not using batch_norm (I recommend NOT using batch norm),
then you can ignore this.

## Visualizing a policy and seeing results
During training, the results will be saved to a file called under
```
PROJECT_PATH/data/local/<default>/<foldername>
```
 -  `PROJECT_PATH` is the directory set by `rllab.config.PROJECT_PATH`. By default the project path is your rllab directory.
 - For example.py `<default>` is `ddpg-half-cheetah`, and can be changed by passing an argument to `exp_prefix` when calling `run_experiment_lite`.
    - If you're using `main.py`, you can pass a `--name=foo` (e.g.) to set the `<default>` folder name to `foo`.
 - `<foldername>` is auto-generated and based off of what `<default>` is.
 - inside this folder, you should see a file called `params.pkl`. To visualize a policy, run

```
(railrl) $ python sim_policy PROJECT_PATH/data/local/<default>/<foldername>/params.pkl
```

## Comments
BatchNorm is implemented and tested, but it seems to hurt DDPG.
I suspect that there may be a bug with it, so use it with caution.

## FAQs
_I'm getting issues about a session being None. (e.g. "None has not method called run")._

This might happen if you run in stub mode and are unserializing a network.
This error is because the un-serialization code expects a default Tensorflow Session to exist.
Fix this by creating a default graph:
```python
with tf.Session().as_default():
    # the rest of your code
```


