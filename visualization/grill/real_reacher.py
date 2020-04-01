import matplotlib
from visualization.grill.config import (
    output_dir,
    ashvin_base_dir,
    format_func,
    our_method_name,
    configure_matplotlib,
)
import matplotlib.pyplot as plt
from railrl.visualization import plot_util as plot

configure_matplotlib(matplotlib)

# f = plot.filter_by_flat_params({'algorithm': 'Ours'})
ours = plot.load_exps([ashvin_base_dir + "s3doodad/share/real-reacher/ours"],
                      suppress_output=True)
plot.tag_exps(ours, "name", "ours")
# f = plot.filter_by_flat_params({'algorithm': 'Sparse-HER', 'reward_params.epsilon': 0.1})
her = plot.load_exps([ashvin_base_dir + "s3doodad/share/real-reacher/her"],
                     suppress_output=True)
plot.tag_exps(her, "name", "her")
# f = plot.filter_by_flat_params({'algorithm': 'TDM'})
oracle = plot.load_exps(
    [ashvin_base_dir + "s3doodad/share/real-reacher/oracle"],
    suppress_output=True)
plot.tag_exps(oracle, "name", "oracle")

plot.comparison(
    ours + her + oracle,
    "Test Final End Effector Distance from Target Mean",
    vary=["name"],
    ylim=(0.0, 0.35),
    xlim=(0, 10000),
    method_order=[2, 0, 1],
    figsize=(6, 5),
)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.ylabel("Final Distance to Goal")
plt.title("Real-World Visual Reacher")
leg = plt.legend([our_method_name, "HER", "Oracle"],
    bbox_to_anchor=(0.49, -0.2),
    loc="upper center",
    ncol=4,
    handlelength=1,
)
leg.get_frame().set_alpha(0.9)
plt.tight_layout()
plt.savefig(output_dir + "real_reacher.pdf")
print("File saved to", output_dir + "real_reacher.pdf")
