README last updated on: 01/17/2018

# railrl
Reinforcement learning framework.
Some implemented algorithms:
 - [DDPG](examples/ddpg.py)
 - [Soft Actor Critic](examples/sac.py)
 - [(Double) DQN](examples/dqn_and_double_dqn.py)
 - [HER](examples/her.py)
 - [MPC with Neural Network Model](examples/model_based_dagger.py)
 - [NAF](examples/naf.py) (I haven't tested this implementation much)

To get started, checkout the example scripts, linked above.

## Installation
This library requires rllab to be installed.
See [rllab's installation docs](https://rllab.readthedocs.io/en/latest/user/installation.html).
(I'm hoping to eventually remove this dependency.)

One important difference: use my anaconda environment instead of rllab's conda
environment. It can be installed and used with
```
$ conda env create -f docker/railrl/railrl-env.yml
$ source activate railrl-env
(railrl-env) $ # Ready to run examples/ddpg_cheetah_no_doodad.py
```

Copy `railrl/launchers/config_template.py` to `railrl/launchers/config.py`
and edit the file as needed.
Update `LOCAL_LOG_DIR`.

### (Optional) Install doodad
I recommend installing [doodad](https://github.com/justinjfu/doodad) to
launch jobs. Some of its nice features include:
 - Easily switch between running code locally, on a remote compute with
 Docker, on EC2 with Docker
 - Easily add your dependencies that can't be installed via pip (e.g. you
 borrowed someone's code)

If you do install `doodad`, I wrote a wrapper for it. Check out
`examples/torch_ddpg_cheetah.py`.

If you install doodad, also modify `CODE_DIRS_TO_MOUNT` in `config.py` to
include:
- Path to rllab directory
- Path to railrl directory
- Path to other code you want to juse

You'll probably also need to update the other variables besides the docker
images/instance stuff.


## Visualizing a policy and seeing results
During training, the results will be saved to a file called under
```
PROJECT_PATH/data/local/<exp_prefix>/<foldername>
```
 -  `PROJECT_PATH` is the directory set by `railrl.launchers.config.AWS_S3_PATH`
 - `<exp_prefix>` is given either to `run_experiment`.
 - `<foldername>` is auto-generated and based off of what `<default>` is.
 - inside this folder, you should see a file called `params.pkl`. To visualize a policy, run

```
(railrl) $ python scripts/sim_policy
PROJECT_PATH/data/local/<default>/<foldername>/params.pkl
```

You can also visualize the results using `rllab`'s `viskit, described at
the bottom of [this page](http://rllab.readthedocs.io/en/latest/user/cluster.html)

tl;dr run

```bash
python rllab/viskit/frontend.py PROJECT_PATH/data/local/<exp_prefix>/
```

## Structure of library
For the most part, each (mini-)project has its own module, like `sac`,
`state_distance`, and `distributional`.
This isn't always the case. There's still some refactoring left to be done,
like how there's `policies/torch.py`, when that could should probably be in
`torch`.

Anyway, most of the sub-modules inside of railrl are hopefully self-explanatory.
Short notes/description of modules whose use may not be obvious

- data_management: code related to handling raw data (mostly just replay
buffers)
- misc: Random assortment of useful tools. Worth checking here if you're
looking for some generic tool.
- optimizers: literally optimizers (no torch or tf here!)
- qfunctions: deprecated
- tf: general tensorflow code (deprecated)
