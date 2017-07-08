"""
Drawing Vector Fields
https://stackoverflow.com/questions/25342072/computing-and-drawing-vector-fields

Adding colorbar to existing axis
https://stackoverflow.com/questions/32462881/add-colorbar-to-existing-axis
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
Create data
"""
y, x = np.mgrid[10:-10:100j, 10:-10:100j]

x_obstacle, y_obstacle = 0.0, 0.0
alpha_obstacle, a_obstacle, b_obstacle = 1.0, 1e3, 2e3

p = -alpha_obstacle * np.exp(-((x - x_obstacle)**2 / a_obstacle
                               + (y - y_obstacle)**2 / b_obstacle))
# For the absolute values of "dx" and "dy" to mean anything, we'll need to
# specify the "cellsize" of our grid.  For purely visual purposes, though,
# we could get away with just "dy, dx = np.gradient(p)".
dy, dx = np.gradient(p, np.diff(y[:2, 0])[0], np.diff(x[0, :2])[0])


"""
Version one
"""
skip = (slice(None, None, 10), slice(None, None, 10))

fig, axes = plt.subplots(2)
ax = axes[0]
im = ax.imshow(
    p,
    extent=[x.min(), x.max(), y.min(), y.max()],
    cmap=plt.get_cmap('plasma'),
)
ax.quiver(x[skip], y[skip], dx[skip], dy[skip])

# fig.colorbar(im)


divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
ax.set(aspect=1, title='Quiver Plot')


"""
Version two
"""
ax = axes[1]
ax.streamplot(x, y, dx, dy, color=p, density=0.5, cmap='gist_earth')

cont = ax.contour(x, y, p, cmap='gist_earth')
ax.clabel(cont)

ax.set(aspect=1, title='Streamplot with contours')
plt.show()
