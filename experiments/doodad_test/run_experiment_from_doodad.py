import doodad as dd
from railrl.launchers.launcher_util import run_experiment

args_dict = dd.get_args()
method_call = args_dict['method_call']
base_log_dir = args_dict['output_dir']
run_experiment_kwargs = args_dict['run_experiment_kwargs']
run_experiment(
    method_call,
    mode="here",
    base_log_dir=base_log_dir,
    **run_experiment_kwargs
)
