import rospy
from geometry_msgs.msg import Vector3, Vector3Stamped
from sensor_msgs.msg import Image
from tf2_ros import TransformListener, Buffer
from ros_numpy import numpify
from robot_env import RobotSetup


class UR5RobotSetup(RobotSetup):

    def __init__(self, vel_topic='vel_command', image_topic='/camera/color/image_raw', ee_frame='tool0'):
        self.ee_frame = ee_frame

        self.vel_pub = rospy.Publisher(vel_topic, Vector3Stamped, queue_size=1)
        self.img_sub = rospy.Publisher(image_topic, Image, self.image_callback, queue_size=1)
        self._last_image = None
        self._img_callback = None
        self.buffer = Buffer()
        TransformListener(self.buffer)

    def image_callback(self, img_msg):
        self._last_image = img_msg
        if self._img_callback is not None:


    def get_rgb_image(self):
        if self._last_image is None:
            return None
        return numpify(self._last_image)

    def handle_control_velocity(self, velocity):
        msg = Vector3Stamped()
        msg.header.frame_id = self.ee_frame
        msg.header.stamp = rospy.Time.now()
        msg.vector = Vector3(*velocity[:3])
        self.vel_pub.publish(msg)

    @property
    def ee_pose(self):
        tf = self.buffer.lookup_transform('base_link', self.ee_frame, rospy.Time(),
                                          timeout=rospy.Duration(0.5)).transform
        return numpify(tf)

    @property
    def ee_position(self):
        return self.ee_pose[:3, 3]

    def set_image_msg_callback(self, func=None):
        self._img_callback = func


if __name__ == '__main__':
    from run_follow_the_leader import FollowTheLeaderController, PIController
    from image_processor import FlowGANImageProcessor
    rospy.init_node('follow_the_leader_node')
    env = UR5RobotSetup()
    pi_controller = PIController()
    image_processor = FlowGANImageProcessor()
    state_machine = FollowTheLeaderController(env, pi_controller, img_process=image_processor)
    state_machine.controller_freq = 999
    rospy.sleep(0.5)
