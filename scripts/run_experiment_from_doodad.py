import doodad as dd
from railrl.launchers.launcher_util import run_experiment_here

args_dict = dd.get_args()
method_call = args_dict['method_call']
run_experiment_kwargs = args_dict['run_experiment_kwargs']
run_experiment_here(
    method_call,
    **run_experiment_kwargs
)
