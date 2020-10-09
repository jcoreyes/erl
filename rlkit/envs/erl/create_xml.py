import xml.etree.cElementTree as ET
from xml.dom import minidom
import numpy as np

x_gap_map = {'easy': 0.3, 'medium': 0.2, 'difficult': 0.1}
y_gap_map = {'easy': 0.3, 'medium': 0.2, 'difficult': 0.1}
friction_range_map = {'easy': [0, 0.01], 'medium': [0.01, 0.02], 'difficult': [0.02, 0.03]}
friction_var_map = {'easy': 1, 'medium': 2, 'difficult': 4}
sphere_size_map = {'easy': 0.04, 'medium': 0.06, 'difficult': 0.08}
west_side = -1.5 # xpos of west barrier
east_side = 50 # xpos of east barrier
full_length = east_side - west_side

comment_idx = 0 # TODO: add comment for the random barriers

# generates blocks in range [x_start, x_end) according to specified level of difficulty
def generate_blocks(worldbody, item_counts, x_start, x_end, level='easy'):
    x_gap, y_gap = x_gap_map[level], y_gap_map[level]
    min_y, max_y = -0.2, 0.3 - y_gap
    curr_pos = x_start
    while curr_pos < x_end:
        block_name = "block" + str(item_counts['block_cnt'])
        y_end = np.random.uniform(min_y, max_y)
        fromto = "{} -0.3 0.01 {} {} 0.01".format(curr_pos, curr_pos, y_end)
        ET.SubElement(worldbody, "geom", conaffinity="1", fromto=fromto, name=block_name, rgba="0.8 0.3 0 1", size=".02", type="capsule")
        item_counts['block_cnt'] += 1

        block_name = "block" + str(item_counts['block_cnt'])
        fromto = "{} {} 0.01 {} 0.3 0.01".format(curr_pos, y_end + y_gap, curr_pos)
        ET.SubElement(worldbody, "geom", conaffinity="1", fromto=fromto, name=block_name, rgba="0.8 0.3 0 1", size=".02", type="capsule")
        item_counts['block_cnt'] += 1
        curr_pos += x_gap
        print(y_end)
    # print('blocks')

def generate_spheres(worldbody, item_counts, x_start, x_end, level='easy'):
    x_start += sphere_size_map[level]
    size = str(sphere_size_map[level])
    body_name = "sphere-body" + str(item_counts['sphere_cnt'])
    item_counts['sphere_cnt']+= 1
    body = ET.SubElement(worldbody, "body", name=body_name, pos="{} 0 0".format(x_start))
    ET.SubElement(body, "geom", conaffinity="1", name="sphere{}".format(item_counts['sphere_cnt']), type="sphere", size=size, rgba="0 0.9 0.1 1")

    body_name = "sphere-body" + str(item_counts['sphere_cnt'])
    item_counts['sphere_cnt'] += 1
    body = ET.SubElement(worldbody, "body", name=body_name, pos="{} 0.2 0".format(x_start + 0.1))
    ET.SubElement(body, "geom", conaffinity="1", name="sphere{}".format(item_counts['sphere_cnt']), type="sphere", size=size, rgba="0 0.9 0.1 1")

    body_name = "sphere-body" + str(item_counts['sphere_cnt'])
    item_counts['sphere_cnt'] += 1
    body = ET.SubElement(worldbody, "body", name=body_name, pos="{} -0.2 0".format(x_start + 0.1))
    ET.SubElement(body, "geom", conaffinity="1", name="sphere{}".format(item_counts['sphere_cnt']), type="sphere", size=size, rgba="0 0.9 0.1 1")

    body_name = "sphere-body" + str(item_counts['sphere_cnt'])
    item_counts['sphere_cnt'] += 1
    body = ET.SubElement(worldbody, "body", name=body_name, pos="{} 0.1 0".format(x_start + 0.3))
    ET.SubElement(body, "geom", conaffinity="1", name="sphere{}".format(item_counts['sphere_cnt']), type="sphere", size=size, rgba="0 0.9 0.1 1")

    body_name = "sphere-body" + str(item_counts['sphere_cnt'])
    item_counts['sphere_cnt'] += 1
    body = ET.SubElement(worldbody, "body", name=body_name, pos="{} -0.1 0".format(x_start + 0.4))
    ET.SubElement(body, "geom", conaffinity="1", name="sphere{}".format(item_counts['sphere_cnt']), type="sphere", size=size, rgba="0 0.9 0.1 1")

    # print("spheres")

def generate_friction_planes(worldbody, item_counts, x_start, x_end, level='easy'):
    num_stripes = friction_var_map[level]
    stripe_width = (x_end - x_start) / num_stripes
    for i in range(num_stripes):
        stripe_x_start = x_start + stripe_width * i
        stripe_x_end = stripe_x_start + stripe_width
        friction_coeffs = np.random.uniform(friction_range_map[level][0], friction_range_map[level][1], size=3)
        stripe_name = "stripe" + str(item_counts['stripe_cnt'])
        stripe_body = ET.SubElement(worldbody, "body", name="friction-plane-body" + str(item_counts['stripe_cnt']),
                                    pos="{} 0 -0.03".format((stripe_x_start + stripe_x_end) / 2))
        ET.SubElement(stripe_body, "geom", conaffinity="1", name=stripe_name, \
            friction="{} {} {}".format(friction_coeffs[0], friction_coeffs[1], friction_coeffs[2]), \
            type="box", size="{} 0.3 0.01".format(stripe_width / 2), rgba="0.7 0 0.5 1")
        item_counts['stripe_cnt'] += 1
    # print("friction planes")

### set up some values that stays the same cross files
def create_xml():
    mujoco = ET.Element("mujoco", model="twod_point")
    compiler = ET.SubElement(mujoco, "compiler", inertiafromgeom="true", angle="radian", coordinate="local")
    option = ET.SubElement(mujoco, "option", timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")

    default = ET.SubElement(mujoco, "default")
    joint = ET.SubElement(default, "joint", limited="false", damping="1")
    default_geom = ET.SubElement(default, "geom", contype="2", conaffinity="1", condim="1", friction=".5 .1 .1", density="1000", margin="0.002")

    worldbody = ET.SubElement(mujoco, "worldbody")

    # pointmass
    worldbody.insert(0, ET.Comment('Pointmass'))
    pointmass = ET.SubElement(worldbody, "body", name="particle", pos="-1.4 0 0")
    cammera = ET.SubElement(pointmass, "camera", name="track", mode="trackcom", pos="0 0 1", xyaxes="1 0 0 0 1 0")
    pointmass_geom = ET.SubElement(pointmass, "geom", name="particle_geom", type="sphere", size="0.02", rgba="1.0 .77 0.09 1", contype="1")
    pointmass_site = ET.SubElement(pointmass, "site", name="particle_site", pos="0 0 0", size="0.01")
    pointmass_joint1 = ET.SubElement(pointmass, "joint", name="ball_x", type="slide", pos="0 0 0", axis="1 0 0")
    pointmass_joint2 = ET.SubElement(pointmass, "joint", name="ball_y", type="slide", pos="0 0 0", axis="0 1 0")

    # target
    # worldbody.insert(2, ET.Comment('Target'))
    # target = ET.SubElement(worldbody, "body", name="target", pos="1.4 0 0")
    # target_geom = ET.SubElement(target, "geom", conaffinity="2", name="target_geom", type="sphere", size="0.02", rgba="0 0.9 0.1 1")

    # arena
    worldbody.insert(4, ET.Comment('Arena'))
    geom1 = ET.SubElement(worldbody, "geom", conaffinity="1", fromto="{} -.3 .01 {} -.3 .01".format(west_side,
                                                                                                    east_side),
                          name="sideS", rgba="0.9 0.4 0.6 1", size=".02", type="capsule")
    geom2 = ET.SubElement(worldbody, "geom", conaffinity="1", fromto="{} -.3 .01 {}  .3 .01".format(east_side,
                                                                                                    east_side),
                          name="sideE", rgba="0.9 0.4 0.6 1", size=".02", type="capsule")
    geom3 = ET.SubElement(worldbody, "geom", conaffinity="1", fromto="{}  .3 .01 {}  .3 .01".format(west_side,
                                                                                                    east_side),
                          name="sideN", rgba="0.9 0.4 0.6 1", size=".02", type="capsule")
    geom4 = ET.SubElement(worldbody, "geom", conaffinity="1", fromto="{} -.3 .01 {} .3 .01".format(west_side,
                                                                                                   west_side),
                          name="sideW", rgba="0.9 0.4 0.6 1", size=".02", type="capsule")

    # actuator
    actuator = ET.SubElement(mujoco, "actuator")
    motor1 = ET.SubElement(actuator, "motor", joint="ball_x", ctrlrange="-1.0 1.0", ctrllimited="true")
    motor2 = ET.SubElement(actuator, "motor", joint="ball_y", ctrlrange="-1.0 1.0", ctrllimited="true")


    ### below is for randomly generating blocks, spheres, and friction planes
    # Here are the four segments for barriers. [-1.2, -0.6], [-0.6, 0], [0, 0.6], [0.6, 1.2]
    seg = 0
    num_seg = int(full_length // 0.6) - 1
    barrier_map = {0: generate_blocks, 1: generate_spheres, 2: generate_friction_planes}
    barrier_numbers = np.random.randint(3, size=num_seg)
    item_counts = dict(block_cnt = 0, sphere_cnt = 0, stripe_cnt = 0)

    for i in range(num_seg):
        curr = west_side + 0.2 + i * (0.6 + x_gap_map['medium'])
        if i == 0:
            barrier_func = generate_blocks # make sure first barrier is blocks
        else:
            barrier_func = barrier_map[barrier_numbers[i]]
        barrier_func(worldbody, item_counts, curr, curr + 0.6, level='medium')
        if i % 100000 == 0:
            print("hi")

    return mujoco

for i in range(5):

    # Write to destination file
    mujoco = create_xml()
    xmlstr = minidom.parseString(ET.tostring(mujoco)).toprettyxml(indent="   ")
    xmlstr = '\n'.join(xmlstr.split('\n')[1:])
    with open("railrl/envs/assets/point_obstacle_long_visualize%d.xml" % i, "w") as f:
        f.write(xmlstr)
