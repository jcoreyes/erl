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

f = plot.filter_by_flat_params({'replay_kwargs.fraction_goals_are_env_goals':
    0.5})
exps = plot.load_exps([ashvin_base_dir +
    "s3doodad/share/reward-reaching-sweep"], f, suppress_output=True)

plot.comparison(
    exps,
    "Final  distance Mean",
    vary=["reward_params.type"],
    # smooth=plot.padded_ma_filter(10),
    ylim=(0.0, 0.2), xlim=(0, 10000),
    # method_order=[1, 0, 2]),
    figsize=(6,4),
)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.xlabel("Timesteps")
plt.ylabel("Final Distance to Goal")
plt.title("Visual Reacher")
plt.legend([])
plt.tight_layout()
plt.savefig(output_dir + "reacher_reward_type_ablation.pdf")
print("File saved to", output_dir + "reacher_reward_type_ablation.pdf")
