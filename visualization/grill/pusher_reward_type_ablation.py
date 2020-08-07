import matplotlib
from visualization.grill.config import (
    output_dir,
    ashvin_base_dir,
    format_func,
    configure_matplotlib,
)
import matplotlib.pyplot as plt
from rlkit.visualization import plot_util as plot

configure_matplotlib(matplotlib)

f = plot.filter_by_flat_params(
    {'replay_kwargs.fraction_goals_are_env_goals': 0.5})
exps = plot.load_exps([
    ashvin_base_dir + 's3doodad/share/steven/pushing-multipushing/pusher-reward-variants'],
    f, suppress_output=True)
plot.tag_exps(exps, "name", "dsae")

plot.comparison(exps,
                ["Final  puck_distance Mean", "Final  hand_distance Mean"],
                figsize=(6, 4),
                vary=["vae_wrapped_env_kwargs.reward_params.type"],
                default_vary={"reward_params.type": "unknown"},
                smooth=plot.padded_ma_filter(10),
                xlim=(0, 250000), ylim=(0.15, 0.22), method_order=None)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.xlabel("Timesteps")
plt.ylabel("")
plt.title("Visual Pusher")
plt.legend([])
plt.tight_layout()
plt.savefig(output_dir + "pusher_reward_type_ablation.pdf")
