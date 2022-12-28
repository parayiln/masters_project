#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import Vector3, Vector3Stamped
from sensor_msgs.msg import Image
from tf2_ros import TransformListener, Buffer
from ros_numpy import numpify
from robot_env import RobotSetup
from run_follow_the_leader import FollowTheLeaderController, PIController, StateMachine
from image_processor import FlowGANImageProcessor



class ROSFollowTheLeaderController(FollowTheLeaderController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_update = rospy.Time()
        self.controller_freq = 0

    def needs_update(self):
        current_update = self.robot.last_image_update()
        return current_update is not None and current_update != self.last_update

    def update_image(self):
        img = self.robot.get_rgb_image()
        cur_image_stamp = self.robot.last_image_update()
        self.img_process.process_image(img)
        elapsed = (cur_image_stamp - self.last_update).to_sec()
        self.last_update = cur_image_stamp
        return elapsed


class UR5RobotSetup(RobotSetup):

    def __init__(self, vel_topic='vel_command', image_topic='/camera/color/image_raw', ee_frame='tool0',
                 dummy_mode=False):


        self.ee_frame = ee_frame
        self.vel_pub = rospy.Publisher(vel_topic, Vector3Stamped, queue_size=1)
        self.img_sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1)
        self._last_image = None

        self.dummy_mode = dummy_mode
        if not self.dummy_mode:
            self.buffer = Buffer()
            TransformListener(self.buffer)
        else:
            self.buffer = None

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

if __name__ == '__main__':

    rospy.init_node('follow_the_leader_node')
    env = UR5RobotSetup(dummy_mode=True)
    pi_controller = PIController()
    image_processor = FlowGANImageProcessor()
    state_machine = ROSFollowTheLeaderController(env, pi_controller, img_process=image_processor)

    rospy.sleep(0.5)

    while not rospy.is_shutdown():
        state_machine.step()

        if state_machine.state == StateMachine.DONE:
            break

