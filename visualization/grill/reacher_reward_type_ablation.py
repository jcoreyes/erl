from visualization.grill.config import (
    output_dir,
    ashvin_base_dir,
    vitchyr_base_dir,
    format_func
)
import matplotlib.pyplot as plt
from railrl.misc import plot_util as plot
from railrl.misc import data_processing as dp

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
    figsize=(6,5),
)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.xlabel("Timesteps")
plt.ylabel("Final Distance to Goal")
plt.title("Visual Reacher")
plt.legend(["Latent Distance", "Log Prob", "Pixel Error" ],
        bbox_to_anchor=(0.49, -0.2), loc="upper center", ncol=4, handlelength=1)
plt.tight_layout()
plt.savefig(output_dir + "reacher_reward_type_ablation.pdf")
print("File saved to", output_dir + "reacher_reward_type_ablation.pdf")
