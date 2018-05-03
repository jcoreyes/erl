import numpy as np

def inverted_pendulum_v2_init_camera(camera):
    camera.trackbodyid = 0
    camera.lookat[2] = .3
    camera.distance=1
    camera.elevation = 0

def reacher_v2_init_camera(camera):
    camera.distance= .7
    camera.elevation = 90
    camera.azimuth = 90

def inverted_double_pendulum_init_camera(camera):
    camera.elevation=1.22
    camera.distance=1.8
    camera.lookat[2] = .6
    camera.trackbodyid = 0

def pusher_2d_init_camera(camera):
    camera.trackbodyid = 0
    camera.distance = 4.0
    rotation_angle = 90
    cam_dist = 4
    cam_pos = np.array([0, 0, 0, cam_dist, -45, rotation_angle])
    for i in range(3):
        camera.lookat[i] = cam_pos[i]
    camera.distance = cam_pos[3]
    camera.elevation = 90
    camera.azimuth = 90
    camera.trackbodyid = -1

def sawyer_init_camera(camera):
    camera.trackbodyid = 0
    camera.distance = 1.0

    # robot view
    #rotation_angle = 90
    #cam_dist = 1
    #cam_pos = np.array([0, 0.5, 0.2, cam_dist, -45, rotation_angle])

    # 3rd person view
    cam_dist = 0.3
    rotation_angle = 270
    cam_pos = np.array([0, 1.0, 0.5, cam_dist, -45, rotation_angle])

    # top down view
    #cam_dist = 0.2
    #rotation_angle = 0
    #cam_pos = np.array([0, 0, 1.5, cam_dist, -90, rotation_angle])

    for i in range(3):
        camera.lookat[i] = cam_pos[i]
    camera.distance = cam_pos[3]
    camera.elevation = cam_pos[4]
    camera.azimuth = cam_pos[5]
    camera.trackbodyid = -1
