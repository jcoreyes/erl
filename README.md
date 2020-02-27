README last updated on: 01/24/2018

# railrl
Reinforcement learning framework.
Some implemented algorithms:
 - [Deep Deterministic Policy Gradient (DDPG)](examples/ddpg.py)
 - [Soft Actor Critic](examples/sac.py)
 - [(Double) Deep Q-Network (DQN)](examples/dqn_and_double_dqn.py)
 - [Hindsight Experience Replay (HER)](examples/her.py)
 - [MPC with Neural Network Model](examples/model_based_dagger.py)
 - [Normalized Advantage Function (NAF)](examples/naf.py)
    - WARNING: I haven't tested this NAF implementation much, so it may not match the paper's performance. I'm pretty confident about the other two implementations though.

To get started, checkout the example scripts, linked above.

## Installation

### Some dependancies

- `sudo apt-get install swig`

### Create Conda Env

Install and use the included ananconda environment
```
$ conda env create -f docker/railrl/railrl-env.yml
$ source activate railrl-env
(railrl-env) $ # Ready to run examples/ddpg_cheetah_no_doodad.py
```
Or if you want you can use the docker image included.

### Download Simulation Env Code
- [multiworld](https://github.com/vitchyr/multiworld) (contains environments):```git clone https://github.com/vitchyr/multiworld```

### (Optional) Install doodad
I recommend installing [doodad](https://github.com/justinjfu/doodad) to
launch jobs. Some of its nice features include:
 - Easily switch between running code locally, on a remote compute with
 Docker, on EC2 with Docker
 - Easily add your dependencies that can't be installed via pip (e.g. you
 borrowed someone's code)

If you install doodad, also modify `CODE_DIRS_TO_MOUNT` in `config.py` to
include:
- Path to rllab directory
- Path to railrl directory
- Path to other code you want to juse

You'll probably also need to update the other variables besides the docker
images/instance stuff.

### Setup Config File

You must setup the config file for launching experiments, providing paths to your code and data directories. Inside `railrl/config/launcher_config.py`, fill in the appropriate paths. You can use `railrl/config/launcher_config_template.py` as an example reference.

```cp railrl/launchers/config-template.py railrl/launchers/config.py```


## Visualizing a policy and seeing results
During training, the results will be saved to a file called under
```
LOCAL_LOG_DIR/<exp_prefix>/<foldername>
```
 - `LOCAL_LOG_DIR` is the directory set by `railrl.launchers.config.LOCAL_LOG_DIR`
 - `<exp_prefix>` is given either to `setup_logger`.
 - `<foldername>` is auto-generated and based off of `exp_prefix`.
 - inside this folder, you should see a file called `params.pkl`. To visualize a policy, run

```
(railrl) $ python scripts/sim_policy LOCAL_LOG_DIR/<exp_prefix>/<foldername>/params.pkl
```

If you have rllab installed, you can also visualize the results
using `rllab`'s viskit, described at
the bottom of [this page](http://rllab.readthedocs.io/en/latest/user/cluster.html)

tl;dr run

```bash
python rllab/viskit/frontend.py LOCAL_LOG_DIR/<exp_prefix>/
```

### Add paths
```
export PYTHONPATH=$PYTHONPATH:/path/to/multiworld/repo
export PYTHONPATH=$PYTHONPATH:/path/to/doodad/repo
export PYTHONPATH=$PYTHONPATH:/path/to/viskit/repo
export PYTHONPATH=$PYTHONPATH:/path/to/railrl-private/repo
```

## Credit
A lot of the coding infrastructure is based on [rllab](https://github.com/rll/rllab).
Also, the serialization and logger code are basically a carbon copy.
