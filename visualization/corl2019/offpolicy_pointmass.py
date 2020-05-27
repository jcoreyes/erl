import matplotlib
from visualization.grill.config import (
    output_dir,
    ashvin_base_dir,
    configure_matplotlib,
)
import matplotlib.pyplot as plt
from railrl.visualization import plot_util as plot

configure_matplotlib(matplotlib)


f = plot.filter_by_flat_params({
    'grill_variant.algo_kwargs.rl_offpolicy_num_training_steps': 10000,
})
ours = plot.load_exps([
  ashvin_base_dir + "s3doodad/ashvin/corl2019/offpolicy/rig-dcvae-offpolicy2-sweep1/run3",
  ashvin_base_dir + "s3doodad/ashvin/corl2019/offpolicy/rig-dcvae-offpolicy2-sweep1/run4",
  ashvin_base_dir + "s3doodad/ashvin/corl2019/offpolicy/rig-dcvae-offpolicy2-sweep1/run5",
  ], f, suppress_output=True)
plot.tag_exps(ours, "name", "off-policy")


baseline = plot.load_exps([ashvin_base_dir + "s3doodad/ashvin/corl2019/offpolicy/rig-dcvae-offpolicy2-baseline", ], suppress_output=True)
plot.tag_exps(baseline, "name", "on-policy")

plot.comparison(
    ours + baseline,
    "evaluation/distance_to_target Mean",
    vary=["name"],
    # xlabel="evaluation/num steps total",
    figsize=(6, 4),
    # figsize=(7.5, 4),
    # method_order=[4, 0, 1, 3, 2],
    ylim=(0.0, 5.0),
    xlim=(0, 10),
)
# plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))
plt.xlabel("Epochs")
plt.ylabel("Final Distance to Goal")
plt.title("Pointmass Off-Policy")

plt.legend()
# plt.tight_layout()
# plt.savefig(output_dir + "reacher_baselines.pdf")
# print("File saved to", output_dir + "reacher_baselines.pdf")

# plt.legend(
#     [
#         our_method_name,
#         "DSAE",
#         "HER",
#         "Oracle",
#         "L&R",
#     ],
#            # bbox_to_anchor=(0.49, -0.2), loc="upper center", ncol=5, handlelength=1)
#     bbox_to_anchor=(1.0, 0.5), loc="center left",
# )
plt.tight_layout()
output_file = output_dir + "offpolicy_pointmass.pdf"
plt.savefig(output_file)
print("File saved to", output_file)
