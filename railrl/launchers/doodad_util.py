from typing import Union

from doodad.easy_launch.python_function import DoodadConfig
from railrl.launchers.launcher_util import setup_experiment


def auto_setup(exp_function, unpack_variant=True):
    """
    Automatically set up:
    1. the logger
    2. the GPU mode
    3. the seed

    :param exp_function: some function that should not depend on `logger_config`
    nor `seed`.
    :param unpack_variant: do you call exp_function with `**variant`?
    nor `seed`.
    :return: function output
    """
    def run_experiment_compatible_function(
            doodad_config: Union[DoodadConfig, None],
            variant
    ):
        if doodad_config:
            variant_to_save = variant.copy()
            variant_to_save['doodad_info'] = doodad_config.extra_launch_info
            setup_experiment(
                variant=variant_to_save,
                exp_name=doodad_config.exp_name,
                base_log_dir=doodad_config.base_log_dir,
                git_infos=doodad_config.git_infos,
                script_name=doodad_config.script_name,
                use_gpu=doodad_config.use_gpu,
                gpu_id=doodad_config.gpu_id,
            )
        variant.pop('logger_config', None)
        variant.pop('seed', None)
        if unpack_variant:
            exp_function(**variant)
        else:
            exp_function(variant)

    return run_experiment_compatible_function
