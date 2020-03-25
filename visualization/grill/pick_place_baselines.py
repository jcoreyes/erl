import matplotlib
from visualization.grill.config import (
    output_dir,
    ashvin_base_dir,
    format_func,
    configure_matplotlib,
)
import matplotlib.pyplot as plt
from railrl.visualization import plot_util as plot

configure_matplotlib(matplotlib)

dirs = [
    ashvin_base_dir + 's3doodad/share/camera_ready_pick',
]
pick_exps = plot.load_exps(dirs, suppress_output=True)

plot.comparison(pick_exps, ["Final hand_and_obj_distance Mean"],
           [
#             "seed",
            "exp_prefix",
            "train_vae_variant.vae_type",
           ],
           default_vary={"train_vae_variant.vae_type": True},
          smooth=plot.padded_ma_filter(10),
          print_final=False, print_min=False, print_plot=True,
          xlim=(0, 500000),
#           ylim=(0, 0.35),
          figsize=(7.5,4),
          method_order=(2, 1, 3, 4, 0),
        )
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.xlabel("Timesteps")
plt.ylabel("")
# plt.legend()
plt.legend(
    [
        "RIG",
        "DSAE",
        "HER",
        "Oracle",
        "L&R",
    ],
           # bbox_to_anchor=(0.49, -0.2), loc="upper center", ncol=5, handlelength=1)
    bbox_to_anchor=(1.0, 0.5), loc="center left",
)
plt.tight_layout()
plt.title("Visual Pick and Place Baselines")
# L&R
# Throw away (SVAE)
# RIG
# DSAE
# HER
# Oracle
plt.savefig(output_dir + "pick_baselines_viz.pdf")
