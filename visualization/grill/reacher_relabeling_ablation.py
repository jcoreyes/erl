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

exps = plot.load_exps([ashvin_base_dir + "s3doodad/share/reacher/reacher-abalation-resample-strategy"], suppress_output=True)
# plot.tag_exps(exps, "name", "oracle")
plot.comparison(exps, "Final  distance Mean",
            vary = ["replay_kwargs.fraction_goals_are_env_goals", "replay_kwargs.fraction_goals_are_rollout_goals"],
#           smooth=plot.padded_ma_filter(10),
          ylim=(0.04, 0.2), xlim=(0, 10000), method_order=[2, 1, 0, 3], figsize=(6,4))
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.xlabel("Timesteps")
plt.ylabel("Final Distance to Goal")
plt.title("Visual Reacher")
# plt.legend([]) # ["Ours", "No Relabeling", "HER", "VAE Only", ])
plt.legend([])
plt.tight_layout()
plt.savefig(output_dir + "reacher_relabeling_ablation.pdf")
print("File saved to", output_dir + "reacher_relabeling_ablation.pdf")
