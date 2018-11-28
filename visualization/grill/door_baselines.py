import matplotlib
from visualization.grill.config import (
    output_dir,
    ashvin_base_dir,
    vitchyr_base_dir,
    format_func,
    our_method_name,
    configure_matplotlib,
)
import matplotlib.pyplot as plt
from railrl.misc import plot_util as plot
from railrl.misc import data_processing as dp

configure_matplotlib(matplotlib)

dirs = [
    ashvin_base_dir + 's3doodad/share/camera_ready_door',
]
exps = plot.load_exps(dirs, suppress_output=True)

plot.comparison(exps, ["Final angle_difference Mean"],
           [
#             "seed",
            "exp_prefix",
           ],
           default_vary={"env_kwargs.randomize_position_on_reset": True},
          smooth=plot.padded_ma_filter(10),
          print_final=False, print_min=False, print_plot=True,
          xlim=(0, 200000),
#           ylim=(0, 0.35),
          figsize=(6, 4),
          method_order=(3, 2, 4, 0, 1),
        )
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.xlabel("Timesteps")
plt.ylabel("Final Distance to Goal")
plt.legend([])
#     [],
#     [
#         "RIG",
#         "DSAE",
#         "HER",
#         "Oracle",
#         "L&R",
#     ],
#            # bbox_to_anchor=(0.49, -0.2), loc="upper center", ncol=5, handlelength=1)
#     bbox_to_anchor=(1.0, 0.5), loc="center left",
# )
plt.tight_layout()
plt.title("Visual Door Baselines")
plt.savefig(output_dir + "door_baselines_viz.pdf")
