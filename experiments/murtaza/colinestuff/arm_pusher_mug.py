import time
import copy
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import mujoco_py
from mujoco_py.mjlib import mjlib
import sys
sys.path.append('/home/coline/objectattention')
#from tf_model_example import get_mlp_layers
import matplotlib.pyplot as plt
import numpy as np

AGENT_MUJOCO= {
    'image_width': 360,
    'image_height': 360,
}

class ArmPusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        print("starting")
        self._viewer_bot = mujoco_py.MjViewer(visible=True, init_width=AGENT_MUJOCO['image_width'],
                                              init_height=AGENT_MUJOCO['image_height'])
        print("made bot")
        self._viewer_bot.start()
        self.boxes = []
        self.feats = []
        self.imgs = []
        self.boxes2 = []
        self.feats2 = []
        self.imgs2= []
        self.step_step = 0
        # self.cam_pos = np.array([0.435, -0.185, -0.15, 0.75, -55., 90.])    # 7DOF camera
        # self.cam_pos = np.array([0.45, -0.15, -0.323, 1.5, -45., 0.])
        # self.cam_pos2 = np.array([0.45, -0.15, -0.323, 1.2, -25., -90.])
        self.cam_pos = np.array([0.45, -0.15, -0.323, 2., -25., 0.])
        self.cam_pos2 = np.array([0.45, -0.15, -0.323, 1.6, -5., -90.])
        self._viewer_bot2 = mujoco_py.MjViewer(visible=True, init_width=AGENT_MUJOCO['image_width'],
                                              init_height=AGENT_MUJOCO['image_height'])
        print("made bot")
        self._viewer_bot2.start()


        from Featurizer import BBProposer, AlexNetFeaturizer
        self.im_w = AGENT_MUJOCO['image_width']
        self.im_h = AGENT_MUJOCO['image_height']
        self.proposer = BBProposer()
        self.featurizer = AlexNetFeaturizer()
        self.max_boxes = 60
        self.query = np.load("/home/coline/sac_master/softqlearning-private/mug_arm_feats.npy")*50
        self.gripper_feats = np.load("/home/coline/sac_master/softqlearning-private/gripper_arm_feats.npy")*50
        self.img_data = []
        self.pos_data = []
        self.suffix = 0
        mujoco_env.MujocoEnv.__init__(self, 'pusher_mug_textured.xml', 5, init_viewer=2)
        #mujoco_env.MujocoEnv.__init__(self, 'pr2_mug.xml', 5, init_viewer=2)
        self.init_qpos[7] = 0.7
        self.init_qpos[9] = 0.7

        #print("starting bot")
        #
        #print("started bot")

        # mujoco_env.MujocoEnv.__init__(self, '/home/larry/dev/data-collect/examples/textured.xml', 5)
        self.init_body_pos = copy.deepcopy(self.model.body_pos)
        self.horizon = 50

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        sites = self.model.data.site_xpos
        vec_1 = sites[0]-sites[1]#self.get_body_com("object")-self.get_body_com("r_gripper_r_finger_link")#tips_arm")
        vec_2 = self.get_body_com("object")-self.get_body_com("goal")
        reward_near = - np.linalg.norm(vec_1)
        reward_dist = - np.linalg.norm(vec_2)
        reward_ctrl = - np.square(a).sum()
        #print("vec1", vec_1, "vec2", vec_2)
        #the coefficients in the following line are ad hoc
        reward = 1.3*reward_dist + 0.7*reward_ctrl + 0.7*reward_near
        self.do_simulation(a, self.frame_skip)
        if self.step_step %20 == 0 and self.step_step > 0:
            self.move_mug()
        self.step_step +=1
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def _get_viewer(self):
        """Override mujoco_env method to put in the
        init_width and init_height

        """
        print("______________________________________________")
        print("in get_viewer")
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(init_width=200, init_height=175)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer

    def viewer_setup(self):
        # cam_pos = np.array([0.435, -0.275, -0.15, 0.55, -50., 90.])    # 7DOF camera
        print("______________________________________________")
        print("in viewer_setup")
        cam_pos = self.cam_pos
        self.viewer.cam.lookat[0] = cam_pos[0]
        self.viewer.cam.lookat[1] = cam_pos[1]
        self.viewer.cam.lookat[2] = cam_pos[2]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def move_mug(self):
        temp = copy.deepcopy(self.model.body_pos)
        idx = 16#11
        angle = np.random.rand(1)*np.pi/2- np.pi/2
        offset = np.array([np.cos(angle), np.sin(angle), 0])*0.2#(np.random.rand(3)-0.5)*0.4
        offset[2] = 0
        temp[idx, :] = self.init_body_pos[idx, :] +offset
        self.model.body_pos = temp
        self.model.step()


    def reset_model(self):
        # qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        self.boxes = []
        self.feats = []
        self.imgs = []
        self.boxes2 = []
        self.feats2 = []
        self.imgs2= []

        print("RESET!")
        qpos = self.init_qpos
        # if len(self.img_data) > 0:
        #     filename = 'data_{0:04d}'.format(self.suffix)
        #     np.save('imgdata/img_'+filename, np.asarray(self.img_data))
        #     np.save('imgdata/pos_'+filename, np.asarray(self.pos_data))
        #     self.img_data = []
        #     self.pos_data = []
        #     self.suffix += 1
        #     print("saved to", filename)
        self._viewer_bot.set_model(self.model)
        self._set_cam_position(self._viewer_bot, self.cam_pos)
        self._viewer_bot.loop_once()
        # self._viewer_bot2.set_model(self.model)
        # self._set_cam_position(self._viewer_bot2, self.cam_pos2)
        # self._viewer_bot2.loop_once()

        # while True:
        #     self.object = np.concatenate([self.np_random.uniform(low=-0.3, high=-0.05, size=1),
        #                              self.np_random.uniform(low=0.25, high=0.65, size=1)])
        #     self.goal = np.asarray([-0.05, 0.45])
        #     if np.linalg.norm(self.object-self.goal) > 0.17: break

        # qpos[-4:-2] = self.object
        # qpos[-2:] = self.goal
        temp = copy.deepcopy(self.init_body_pos)
        idx = 16#11
        angle = 0*np.random.rand(1)*np.pi/2- np.pi/2
        offset = np.array([np.cos(angle), np.sin(angle), 0])*0.2#(np.random.rand(3)-0.5)*0.4
        offset[2] = 0
        temp[idx, :] = temp[idx, :] +offset
        self.model.body_pos = temp
        self.model.step()


        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-4:] = 0
        #import IPython; IPython.embed()
        self.set_state(qpos, qvel)
        #import IPython; IPython.embed()
        obs =  self._get_obs()
        return obs

    def _plot_attention(self, img, box, c=0,save=False):
        #
        #print(probs[argmax])
        self.proposer.draw_box(((box+0.5)*AGENT_MUJOCO['image_height']), img, c, width=2)
        #self.proposer.draw_box(softbox, img, 1)
        #import IPython;IPython.embed()
        #plt.show(plt.imshow(img))
        if save:
            filename = '/home/coline/Videos/objects/imgs/sacarm_itr30_{0:04d}.png'.format(self.suffix)
            self.suffix+=1
            plt.imsave(filename, img)
    def _get_attention(self, boxes, feats, img, query):
        #
        q = query.copy()
        q = np.reshape(q, [feats.shape[1], 1])
        cos = np.abs(np.matmul(feats,q))
        exp = np.exp(cos)
        Z = np.sum(exp)
        probs = exp/Z
        nprobs = np.tile(probs, [1,4])
        softbox = np.sum(nprobs*boxes, axis = 0)
        argmax= np.argmax(probs)
        # print(probs[argmax])
        # self.proposer.draw_box(boxes[argmax], img, 0)
        # self.proposer.draw_box(softbox, img, 1)
        #import IPython;IPython.embed()
        #plt.show(plt.imshow(img))
        return boxes[argmax]


    def _get_obs(self):
        self._viewer_bot.loop_once()
        self._viewer_bot2.loop_once()
        img_string, width, height = self._viewer_bot.get_image()#CHANGES
        img = np.fromstring(img_string, dtype='uint8').reshape(height, width, 3)[::-1,:,:]
        boxes = np.array(self.proposer.extract_proposal(img)[:self.max_boxes])
        crops = [self.proposer.get_crop(b, img) for b in boxes]
        feats = np.array([self.featurizer.getFeatures(c) for c in crops])
        boxes = boxes/AGENT_MUJOCO['image_height'] -0.5
        self.boxes.append(boxes)
        self.feats.append(feats)
        self.imgs.append(img)

        img_string, width, height = self._viewer_bot2.get_image()#CHANGES
        img = np.fromstring(img_string, dtype='uint8').reshape(height, width, 3)[::-1,:,:]
        boxes = np.array(self.proposer.extract_proposal(img)[:self.max_boxes])
        crops = [self.proposer.get_crop(b, img) for b in boxes]
        feats = np.array([self.featurizer.getFeatures(c) for c in crops])
        boxes = boxes/AGENT_MUJOCO['image_height'] -0.5
        self.boxes2.append(boxes.copy())
        self.feats2.append(feats.copy())
        self.imgs2.append(img.copy())
        #print("Have", len(self.boxes), "boxes")
        # plotimg = img.copy()
        # box = self._get_attention(boxes, feats, img, self.query)
        # gripperbox = self._get_attention(boxes, feats, img, self.gripper_feats)
        # for b in range(boxes.shape[0]):
        #     self.proposer.draw_box((boxes[b]+0.5)*360, img, 2)

        # import IPython; IPython.embed()
        # self.last_box = box.copy()
        # self.last_gripperbox = gripperbox.copy()
        # self._plot_attention(plotimg, box, c= 0)
        # self._plot_attention(plotimg, gripperbox, c =1, save=True)# np.load("feats_500.npy"))# np.load('w_attention_280.npy'))

        # self.img_data.append(img)
        # self.pos_data.append(self.get_body_com("object"))
        #import IPython; IPython.embed()
        # img_data = img.flatten()
        sites = self.model.data.site_xpos#.flatten()
        #import IPython; IPython.embed()
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],
            #sites.flatten(),
            #self.get_body_com("tips_arm"),
            #self.get_body_com("object"),
            sites[0].flatten(),
            sites[1].flatten(),
            self.get_body_com("goal")
            # img_data
        ])
    def _set_cam_position(self, viewer, cam_pos):

        for i in range(3):
            viewer.cam.lookat[i] = cam_pos[i]
        viewer.cam.distance = cam_pos[3]
        viewer.cam.elevation = cam_pos[4]
        viewer.cam.azimuth = cam_pos[5]
        viewer.cam.trackbodyid = -1
