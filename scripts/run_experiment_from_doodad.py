import doodad as dd
from railrl.launchers.launcher_util import run_experiment_here

args_dict = dd.get_args()
method_call = args_dict['method_call']
run_experiment_kwargs = args_dict['run_experiment_kwargs']
output_dir = args_dict['output_dir']
print("START from run_experiment_from_doodad:")
# for k, v in run_experiment_kwargs.items():
    # print(k, v)
print("output_dir", output_dir)
print("END from run_experiment_from_doodad:")
run_experiment_here(
    method_call,
    snapshot_dir=output_dir,
    **run_experiment_kwargs
)
