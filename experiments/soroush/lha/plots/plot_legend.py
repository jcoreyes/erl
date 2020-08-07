import os.path as osp
labels = [
    'DisCo RL (ours)',
    'GCRL',
    'VICE',
    'Ours: no mean relabeling',
    'Ours: no cov relabeling',
]

import pylab
fig = pylab.figure()
# figlegend = pylab.figure(figsize=(2,1))
figlegend = pylab.figure(figsize=(2.3,1.1)) #figsize=(1.65,1.30)
ax = fig.add_subplot(111)
inputs_to_ax_plot = [range(10), pylab.randn(10)] * len(labels)
lines = ax.plot(inputs_to_ax_plot)
# lines = ax.plot(range(10), pylab.randn(10), range(10), pylab.randn(10))

figlegend.legend(lines, labels, 'center')
fig.show()
figlegend.show()

plot_dir = '/home/soroush/research/railrl/experiments/soroush/lha/plots'
full_plot_name = osp.join(plot_dir, 'legend.pdf')
figlegend.savefig(full_plot_name)