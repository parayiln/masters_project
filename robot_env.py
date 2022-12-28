import os
import pdb
import numpy as np
from scipy.spatial.transform import Rotation
from collections import namedtuple
from abc import ABC, abstractmethod
try:
    import pybullet as p
    import pybullet_data
except ModuleNotFoundError:
    pass

try:
    import rospy
    from geometry_msgs.msg import Vector3, Vector3Stamped
    from sensor_msgs.msg import Image
    from tf2_ros import TransformListener, Buffer
    from ros_numpy import numpify
except ModuleNotFoundError:
    print('Could not load some ROS utilities')

class RobotSetup(ABC):
    @abstractmethod
    def get_rgb_image(self):
        ...

    @abstractmethod
    def handle_control_velocity(self, velocity):
        ...

    @property
    @abstractmethod
    def ee_pose(self):
        ...

    @property
    @abstractmethod
    def ee_position(self):
        ...

    @property
    def has_linear_axis(self):
        return False

    def handle_linear_axis_control(self, target_vel):
        if not self.has_linear_axis:
            return
        raise NotImplementedError()

class PybulletRobotSetup(RobotSetup):
    def __init__(self):
        self.serverMode = p.GUI  # GUI/DIRECT

        root = os.path.dirname(os.path.realpath(__file__))
        self.robotUrdfPath = os.path.join(root, 'urdf', 'ur5_7thaxis.urdf')
        self.tree = os.path.join(root, 'meshes', 'ufo_trees_labelled', '1_l.ply')
        self.physicsClient = p.connect(self.serverMode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF("plane.urdf")
        self.robotStartPos = [0, 0, 0]
        self.robotStartOrn = p.getQuaternionFromEuler([0, 0, 0])
        print("----------------------------------------")
        print("Loading robot from {}".format(self.robotUrdfPath))
        self.robotID = p.loadURDF(self.robotUrdfPath, self.robotStartPos,
                                  self.robotStartOrn, flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)

        self.tf = np.identity(4)
        self.home_joints = [.543, -1.588, -1.2689, -1.08, 2.303, -1.67, 4.69]
        self.num_joints = p.getNumJoints(self.robotID)
        self.joint_limits = {}
        self.revolute_joints = []
        self.revolute_and_prismatic_joints = []
        self.joint_names_to_ids = {}
        self.joints = dict()
        self.control_joints = ["7thjoint_prismatic", "shoulder_pan_joint", "shoulder_lift_joint",
                               "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE",
                                "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", [
                                     "id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])
        self.load_joints()
        self.end_effector_name = "wrist_3_joint"

        self.current_step = 0
        self.steps_per_second = 240  # Pybullet default
        self.camera_freq = 20
        self.width = 424
        self.height = 240
        self.fov = 42  # changes to get the cutter in view
        self.aspect = self.width / self.height
        self.near = 0.01
        self.far = 10

    @property
    def has_linear_axis(self):
        return True

    def get_rgb_image(self):
        ee_position = self.get_link_kinematics('wrist_3_link-tool0_fixed_joint', as_matrix=True)
        rgb_raw = self.load_cam(ee_position)
        img = self.read_cam_video(rgb_raw)
        return img
####################################################################

    def read_cam_video(self, img):
        # frame = img[3]
        frame = img[:, :, :]
        return frame


    def load_cam(self, ee_position):
        pose = [ee_position[0][3], ee_position[1][3], ee_position[2][3]]
        pose_target = np.dot(ee_position, np.array([0, 0, .3, 1]))[:3]
        self.view_matrix = np.reshape(p.computeViewMatrix(
            cameraEyePosition=pose, cameraTargetPosition=pose_target, cameraUpVector=[0, 0, 1]), (4, 4))
        self.projection_matrix = p.computeProjectionMatrixFOV(
            self.fov, self.aspect, self.near, self.far)
        _, _, rgb_img, raw_depth_img, raw_seg_img = p.getCameraImage(width=self.width,
                                                                     height=self.height,
                                                                     viewMatrix=self.view_matrix.reshape(
                                                                         -1),
                                                                     projectionMatrix=self.projection_matrix,
                                                                     shadow=True,
                                                                     renderer=p.ER_BULLET_HARDWARE_OPENGL)

        return rgb_img

####################################################################
    def robot_step(self):
        p.stepSimulation()
        self.current_step += 1

    @property
    def elapsed_time(self):
        return self.current_step / self.steps_per_second

    @property
    def ee_pose(self):
        return self.get_link_kinematics('wrist_3_link-tool0_fixed_joint', as_matrix=True)

    @property
    def ee_position(self):
        return self.ee_pose[:3, 3]

    def reset(self):
        self.move_joints(self.home_joints, "position")
        print("----------------------------------------")
        print("Robot moved to initial state")
        for _ in range(100):
            self.robot_step()
        self.current_step = 0

    def get_joint_state(self, joint_name):
        joint_id = self.joint_names_to_ids[joint_name]
        joint_position, joint_vel, joint_force, torque = p.getJointState(
            self.robotID, joint_id)
        return joint_position, joint_vel, joint_force, torque

    def getJointStates(self):
        jointids = self.get_control_joints()
        joint_states = p.getJointStates(self.robotID, jointids)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques


    def load_joints(self):
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robotID, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            self.joint_names_to_ids[jointName] = i
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit,
                                   jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            self.joints[info.name] = info
            if jointType in {p.JOINT_PRISMATIC, p.JOINT_REVOLUTE}:
                self.revolute_and_prismatic_joints.append(i)

    def get_control_joints(self):
        joint_ids = []
        joint_ids.append(self.joint_names_to_ids["7thjoint_prismatic"])
        joint_ids.append(self.joint_names_to_ids["shoulder_pan_joint"])
        joint_ids.append(self.joint_names_to_ids["shoulder_lift_joint"])
        joint_ids.append(self.joint_names_to_ids["elbow_joint"])
        joint_ids.append(self.joint_names_to_ids["wrist_1_joint"])
        joint_ids.append(self.joint_names_to_ids["wrist_2_joint"])
        joint_ids.append(self.joint_names_to_ids["wrist_3_joint"])
        return joint_ids


    def move_joints(self, joint_values, control_type = "velocity"):
        # joint_ids = self.get_control_joints()
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_values[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        if control_type == "position":
            targetPositions = joint_values
            targetVelocities = [0]*len(poses)
            controller = p.POSITION_CONTROL
        elif control_type == "velocity":
            targetPositions = [0]*len(poses)
            controller = p.VELOCITY_CONTROL
            targetVelocities = joint_values

        p.setJointMotorControlArray(self.robotID, indexes, controller, targetPositions,
                                    targetVelocities, positionGains=[0.05]*len(poses), forces=forces)

    def handle_linear_axis_control(self, target_vel):
        id = self.joint_names_to_ids['7thjoint_prismatic']
        controller = p.VELOCITY_CONTROL
        target_pos = 0
        p.setJointMotorControl2(
            self.robotID, id, controller, target_pos, target_vel)


    def get_links(self):

        linkIDs = list(
            map(lambda linkInfo: linkInfo[1], p.getVisualShapeData(self.robotID)))
        linkNum = len(linkIDs)
        return linkNum


    def convert_link_name(self, name):
        if isinstance(name, int):
            return name
        return self.joint_names_to_ids[name]


    def get_link_kinematics(self, link_name_or_id, use_com_frame=False, as_matrix=False):

        link_id = self.convert_link_name(link_name_or_id)

        pos_idx = 4
        quat_idx = 5
        if use_com_frame:
            pos_idx = 0
            quat_idx = 1

        rez = p.getLinkState(self.robotID, link_id,
                             computeForwardKinematics=True)
        position, orientation = rez[pos_idx], rez[quat_idx]

        if as_matrix:
            tf = np.identity(4)
            tf[:3, :3] = Rotation.from_quat(orientation).as_matrix()
            tf[:3, 3] = position
            return tf
        else:
            return position, orientation

    def solve_end_effector_ik(self, link_name_or_id, target_position, target_orientation=None, threshold=None, max_iters=20):

        link_id = self.convert_link_name(link_name_or_id)

        return p.calculateInverseKinematics(self.robot.robotID, link_id, target_position[0:3])

    def calc_jacobian(self, joint_pose):
        link_id = self.convert_link_name(self.end_effector_name)
        zero_vec = [0.0]*len(self.control_joints)
        if joint_pose != "None":
            joint_pos, _, _ = self.getJointStates()
        else:
            joint_pos = list(joint_pose)
        joint_vel = zero_vec
        joint_acc = zero_vec
        _, _, position_local, _, _, _, _, _ = p.getLinkState(self.robotID,
                                                             link_id,
                                                             computeLinkVelocity=1,
                                                             computeForwardKinematics=1)

        return p.calculateJacobian(self.robotID, link_id, position_local, joint_pos, joint_vel, joint_acc)


    def solveForwardVelocityKinematics(self, joint_vel):
        print('Forward velocity kinematics')
        joint_pos, _, _ = self.getJointStates()
        J = self.calc_jacobian(joint_pos)
        eeVelocity = np.dot(J, joint_vel)
        return eeVelocity

    def getInverseVelocityKinematics(self, end_eff_velocity):
        joint_pos, _, _ = self.getJointStates()
        J = self.calc_jacobian(joint_pos)
        jacobian = np.zeros((6, 7))
        for i in range(0, 3):
            jacobian[:][i] = J[0][i]
            jacobian[:][i+3] = J[1][i]
        ctrl_joint = self.get_control_joints()
        if len(ctrl_joint) > 1:
            joint_vel = np.dot(np.linalg.pinv(jacobian), end_eff_velocity)
        else:
            joint_vel = np.dot(jacobian.T, end_eff_velocity)
        return joint_vel

    def handle_control_velocity(self, velocity):
        joint_value_vel = self.getInverseVelocityKinematics(velocity)
        self.move_joints(joint_value_vel)
        return joint_value_vel

        

    def move_end_effector_ik(self, link_name_or_id, target_position, target_orientation=None, threshold=0.005, retries=3):
        for _ in range(retries):
            ik = self.solve_end_effector_ik(link_name_or_id, target_position, target_orientation=target_orientation,
                                                  threshold=threshold)
            self.reset_joint_states(ik)
            pos, _ = self.get_link_kinematics(link_name_or_id)
            offset = np.linalg.norm(np.array(target_position) - pos[0])
            if offset < threshold:
                return True
        return False

    def __test__(self):
        self.reset()
        for i in range(1000):
            p.stepSimulation()
            self.get_rgb_image()
        p.stepSimulation()


class UR5RobotSetup(RobotSetup):

    def __init__(self, ee_frame='tool0'):

        self.ee_frame = ee_frame

        self.vel_pub = rospy.Publisher('vel_command', Vector3Stamped, queue_size=1)
        self.img_sub = rospy.Publisher('rgb_image', Image, self.image_callback, queue_size=1)
        self._last_image = None
        self.buffer = Buffer()
        TransformListener(self.buffer)

    def image_callback(self, img_msg):
        self._last_image = img_msg

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
        tf = self.buffer.lookup_transform('base_link', self.ee_frame, rospy.Time(), timeout=rospy.Duration(0.5)).transform
        return numpify(tf)

    @property
    def ee_position(self):
        return self.ee_pose[:3,3]


if __name__ == "__main__":
    robot = PybulletRobotSetup()
    robot.__test__()

