import mujoco_py
model = mujoco_py.load_model_from_path('sawyer_kitchen.xml')
sim = mujoco_py.MjSim(model)
sim.render()
