#!/usr/bin/env python
import os.path

import numpy as np
import rospy
from geometry_msgs.msg import Vector3, Vector3Stamped
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import TransformListener, Buffer
from ros_numpy import numpify, msgify
from robot_env import RobotSetup
from run_follow_the_leader import FollowTheLeaderController, PixelBasedPIController, StateMachine
from image_processor import FlowGANImageProcessor
from image_geometry import PinholeCameraModel
from std_srvs.srv import Empty
import subprocess as sp
import shlex
from arm_utils.srv import HandleJointPlan
from sensor_msgs.msg import JointState
import cv2 as cv
from PIL import Image as PILImage

class ROSFollowTheLeaderController(FollowTheLeaderController):
    def __init__(self, *args, **kwargs):
        self.bag_folder = kwargs.pop('bag_folder', None)
        self.bag_topics = kwargs.pop('bag_topics', [])
        self.bag_proc = None

        super().__init__(*args, **kwargs)
        self.last_update = rospy.Time()
        self.controller_freq = 0
        self.state = StateMachine.INACTIVE

        self.scan_start_pos = None
        self.scan_vertical_dist = 0.35
        self.scan_velocity = 0.10

        rospy.Service('activate_leader_follow', Empty, self.switch_to_leader_follow)
        self.servo_activate = rospy.ServiceProxy('servo_activate', Empty)
        self.servo_stop = rospy.ServiceProxy('servo_stop', Empty)
        self.diagnostic_pub = rospy.Publisher('diagnostic_img', Image, queue_size=1)

    def needs_update(self):
        current_update = self.robot.last_image_update()
        return current_update is not None and current_update != self.last_update

    def leader_follow_is_done(self):
        # if self.img_process.curve is None:
        #     return True
        ee_pos = self.robot.ee_position
        z_pos = ee_pos[2]
        if np.abs(z_pos - self.scan_start_pos[2]) > self.scan_vertical_dist:
            return True
        return False

    def update_image(self):
        img = self.robot.get_rgb_image()
        cur_image_stamp = self.robot.last_image_update()
        self.img_process.process_image(img)
        elapsed = (cur_image_stamp - self.last_update).to_sec()
        self.last_update = cur_image_stamp
        return elapsed

    def switch_to_leader_follow(self, *_):
        if self.state == StateMachine.LEADER_FOLLOW:
            rospy.logwarn('The system is already in leader following mode! Not doing anything...')
            return []

        self.robot.move_to_home_position()
        self.direction = -1
        self.scan_start_pos = self.robot.ee_position
        self.scan_velocity = rospy.get_param('scan_velocity', self.scan_velocity)

        self.servo_activate()
        if self.bag_folder is not None and self.bag_topics:
            num_files = len([x for x in os.listdir(self.bag_folder) if x.endswith('.bag')])
            file_path = os.path.join(self.bag_folder, f'{num_files}.bag')
            cmd = 'rosbag record -O {} {}'.format(file_path, ' '.join(self.bag_topics))
            self.bag_proc = sp.Popen(shlex.split(cmd), stderr=sp.PIPE, shell=False)

        self.state = StateMachine.LEADER_FOLLOW
        # Robot needs to receive optical flow before moving
        vel = np.array([0, 0.025, 0, 0, 0, 0]) * (-1 * self.direction)
        self.robot.handle_control_velocity(vel)

        return []

    def deactivate_leader_follow(self):
        super().deactivate_leader_follow()
        self.servo_stop()
        if self.bag_proc is not None:
            self.bag_proc.terminate()

        self.scan_start_pos = None

    def visualize(self):

        if self.img_process.mask is None:
            return

        img = self.img_process.image
        flow = self.img_process.flow
        mask = self.img_process.mask

        diag = (0.5 * img + 0.5 * mask).astype(np.uint8)
        arrows = self.viz_arrows
        if arrows is None:
            arrows = []
        h, w = diag.shape[:2]
        cv.line(diag, (w // 2, 0), (w // 2, h), (0, 0, 0), 2)
        cv.line(diag, (0, h // 2), (w, h // 2), (0, 0, 0), 2)
        cv.circle(diag, (w // 2, h // 2), 2, (0, 255, 0), -1)
        if self.current_target is not None:
            cv.circle(diag, (int(self.current_target[0]),int(self.current_target[1])), 2, (0, 0, 255), -1)
        for start, end, color in arrows:
            cv.arrowedLine(diag, start, end, color, 3)

        final_img = np.zeros((2*h, 2*w, 3), dtype=np.uint8)
        final_img[:h,:w] = img
        final_img[h:2*h,:w] = np.array(PILImage.fromarray(flow).resize((w,h))).astype(np.uint8)
        final_img[:h,w:2*w] = mask
        final_img[h:2*h,w:2*w] = diag

        self.diagnostic_pub.publish(msgify(Image, final_img, 'rgb8'))


class UR5RobotSetup(RobotSetup):

    def __init__(self, vel_topic='vel_command', image_topic='/camera/color/image_raw_throttled',
                 cam_info_topic='/camera/color/camera_info', ee_frame='tool0',
                 dummy_mode=False):


        self.ee_frame = ee_frame
        self.vel_pub = rospy.Publisher(vel_topic, Vector3Stamped, queue_size=1)
        self.img_sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1)
        self._last_image = None
        cam_info = rospy.wait_for_message(cam_info_topic, CameraInfo, 2.0)

        self.camera = PinholeCameraModel()
        self.camera.fromCameraInfo(cam_info)
        self.fov = self.camera.fovX()
        self.width = self.camera.width
        self.height = self.camera.height

        self.dummy_mode = dummy_mode
        if not self.dummy_mode:
            self.buffer = Buffer()
            TransformListener(self.buffer)
        else:
            self.buffer = None

        self.plan_joint_srv = rospy.ServiceProxy('plan_joints', HandleJointPlan)
        rospy.Service('move_home', Empty, self.move_to_home_position)

    def image_callback(self, img_msg):
        self._last_image = img_msg

    def get_rgb_image(self):
        if self._last_image is None:
            return None
        return numpify(self._last_image)[:,:,:3]

    def last_image_update(self):
        if self._last_image is None:
            return None
        return self._last_image.header.stamp

    def handle_control_velocity(self, velocity):
        msg = Vector3Stamped()
        msg.header.frame_id = self.ee_frame
        msg.header.stamp = rospy.Time.now()
        msg.vector = Vector3(*velocity[:3])
        self.vel_pub.publish(msg)

    @property
    def ee_pose(self):
        if not self.dummy_mode:
            tf = self.buffer.lookup_transform('base_link', self.ee_frame, rospy.Time(),
                                              timeout=rospy.Duration(0.5)).transform
            return numpify(tf)
        else:
            tf = np.identity(4)
            tf[:3,3] = [0, 0, 0.4]
            return tf

    @property
    def ee_position(self):
        return self.ee_pose[:3, 3]

    def robot_step(self):
        pass

    def reset(self):
        pass

    def move_to_home_position(self, *_):
        home_joints = [0.75, -1.68, 1.23, -2.70, -1.54, 0]
        joints = JointState()
        joints.position = home_joints
        self.plan_joint_srv(joints, True)

        return []


if __name__ == '__main__':

    bag_folder = os.path.join(os.path.expanduser('~'), 'data', 'follow_the_leader')

    im_topic = '/camera/color/image_raw_throttled'
    cam_topic = '/camera/color/camera_info'

    bag_topics = [im_topic, cam_topic, 'tool_pose', 'vel_command', 'diagnostic_img']

    rospy.init_node('follow_the_leader_node')
    env = UR5RobotSetup(dummy_mode=False)
    pi_controller = PixelBasedPIController(env.fov, env.width, kp=0.8)
    image_processor = FlowGANImageProcessor((env.width, env.height))
    state_machine = ROSFollowTheLeaderController(env, pi_controller, img_process=image_processor,
                                                 bag_folder=bag_folder, bag_topics=bag_topics, visualize=True)

    rospy.on_shutdown(state_machine.deactivate_leader_follow)
    rospy.sleep(0.5)

    rospy.loginfo('Follow the leader controller has been initialized!\nCall service activate_leader_follow to activate')

    while not rospy.is_shutdown():
        state_machine.step()
        rospy.sleep(0.01)


