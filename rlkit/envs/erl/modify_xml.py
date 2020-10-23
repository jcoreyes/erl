import os
import xml.etree.ElementTree as ET
#import ipdb
import numpy as np
import rlkit.envs.erl.half_cheetah

def create_new_xml(model_path, mods={}):

    fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)

    tree = ET.parse(fullpath)
    root = tree.getroot()

    def mod_elem(node, elem_string, name, multiplier):
        if isinstance(multiplier, list):
            multiplier = 1.0
        new_values = [float(x) * multiplier for x in elem_string.split(' ')]
        new_values_str = ' '.join([str(x) for x in new_values])
        node.set(name, new_values_str)

    def rotate_gravity(node, elem_string, name, multiplier):
        # make a multipler of 1.1 correspond to 10% incline
        if isinstance(multiplier, list):
            multiplier = 1.0
        degrees = (multiplier - 1) * 10
        theta = np.radians(degrees)
        r = np.array(((np.cos(theta), -np.sin(theta)),
                      (np.sin(theta), np.cos(theta))))
        new_vector = r.dot(np.array([0, -9.8]))
        new_values = [new_vector[0], 0, new_vector[1]]
        new_values_str = ' '.join([str(x) for x in new_values])
        node.set(name, new_values_str)


    if 'gravity' in mods:
        for x in root.iter('option'):
            #mod_elem(x, x.get('gravity', default='0 0 -9.8'), 'gravity', mods['gravity'])
            #x.set('gravity', '0 0 %f' % (mods['gravity']))
            rotate_gravity(x, x.get('gravity', default='0 0 -9.8'), 'gravity', mods['gravity'])

    if 'friction' in mods:
        # Modify all values of friction
        for x in root.iter('geom'):
            friction = x.get('friction')
            if friction is not None:
                mod_elem(x, friction, 'friction', mods['friction'])

    if 'ctrlrange' in mods:
        # Modify all values of ctrlrange
        for x in root.iter('motor'):
            if x.get('ctrlrange') is not None:
                mod_elem(x, x.get('ctrlrange'), 'ctrlrange', mods['ctrlrange'])

    if 'gear' in mods:
        # Modify all values of ctrlrange
        for x in root.iter('motor'):
            if x.get('gear') is not None:
                mod_elem(x, x.get('gear'), 'gear', mods['gear'])

    new_path = os.path.join(os.path.dirname(__file__), "assets", 'tmp-' + model_path)
    tree.write(new_path)
    return new_path

if __name__ == '__main__':
    mods = dict(gravity=0.9,
                friction=0.9,
                ctrlrange=0.9,
                gear=0.9
                )
    # friction is only standard in geom for ant, half_cheetah
    # not standord for hooper, could apply to foot or all geoms
    #print(create_new_xml('ant.xml', mods))
    #env = HalfCheetahEnv(mods)
    multiplier = 0.9
    degrees = (multiplier - 1) * 10
    theta = np.radians(degrees)
    r = np.array(((np.cos(theta), -np.sin(theta)),
                  (np.sin(theta), np.cos(theta))))
    new_vector = r.dot(np.array([0, -9.8]))
    new_values = [new_vector[0], 0, new_vector[1]]
    print(new_values)