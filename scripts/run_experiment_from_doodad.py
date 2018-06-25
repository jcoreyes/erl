import doodad as dd
import uuid
from railrl.launchers.launcher_util import run_experiment_here
from railrl.launchers import config
import os

args_dict = dd.get_args()
method_call = args_dict['method_call']
run_experiment_kwargs = args_dict['run_experiment_kwargs']
output_dir = args_dict['output_dir']
run_mode = args_dict.get('mode', None)
if run_mode and run_mode == 'ec2':
    # Hacky. Turning the output dir from
    # OUTPUT_DIR_FOR_DOODAD_TARGET to OUTPUT_DIR_FOR_DOODAD_TARGET/exp-name
    subdir_prefix = args_dict['s3_log_name']
    output_dir = config.OUTPUT_DIR_FOR_DOODAD_TARGET
    # hopefully unique enough
    unique_sub_id = str(uuid.uuid4()).split('-')[0]
    output_dir = os.path.join(output_dir, subdir_prefix + "-" + unique_sub_id)
    try:
        import urllib.request
        instance_id = urllib.request.urlopen(
            'http://169.254.169.254/latest/meta-data/instance-id'
        ).read().decode()
        run_experiment_kwargs['variant']['EC2_instance_id'] = instance_id
    except Exception as e:
        print("Could not get instance ID. Error was...")
        print(e)

run_experiment_here(
    method_call,
    log_dir=output_dir,
    **run_experiment_kwargs
)
