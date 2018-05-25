import mujoco_py
import matplotlib.pyplot as plt
model = mujoco_py.load_model_from_path('sawyer_door.xml')
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)
# plt.imshow(sim.render(500, 500))
# plt.show()
while True:
    viewer.render()