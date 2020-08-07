import matplotlib
from visualization.grill.config import (
    output_dir,
    ashvin_base_dir,
    format_func,
    our_method_name,
    configure_matplotlib,
)
import matplotlib.pyplot as plt
from rlkit.visualization import plot_util as plot

configure_matplotlib(matplotlib)

dirs = [
    ashvin_base_dir + 's3doodad/ashvin/vae/fixed3/sawyer-pusher/vae-dense-wider3/run1',
    ashvin_base_dir + 's3doodad/ashvin/vae/fixed3/sawyer-pusher/vae-dense-wider3-relabeling/run1',
]
f = plot.filter_by_flat_params({
    'algo_kwargs.num_updates_per_env_step': 4,
    "replay_kwargs.fraction_goals_are_env_goals": 0.5,
    "replay_kwargs.fraction_goals_are_rollout_goals": 0.2
})
exps = plot.load_exps(dirs, suppress_output=True)
plot.comparison(
    exps,
    ["Final  puck_distance Mean", "Final  hand_distance Mean"],
    vary=["replay_kwargs.fraction_goals_are_env_goals",
          "replay_kwargs.fraction_goals_are_rollout_goals"],
    default_vary={"replay_strategy": "future"},
    smooth=plot.padded_ma_filter(10),
    xlim=(0, 500000),
    ylim=(0.14, 0.26),
    figsize=(6, 5),
    method_order=[2, 1, 0, 3],
)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
# plt.ylabel("")
plt.xlabel("Timesteps")
plt.ylabel("Final Distance to Goal")
plt.title("Visual Pusher")
plt.legend(
    [our_method_name, "None", "Future", "VAE", ],
    bbox_to_anchor=(0.49, -0.2),
    loc="upper center",
    ncol=4,
    handlelength=1,
)
plt.tight_layout()
plt.savefig(output_dir + "pusher_relabeling_ablation_main.pdf")
print("File saved to", output_dir + "pusher_relabeling_ablation_main.pdf")
