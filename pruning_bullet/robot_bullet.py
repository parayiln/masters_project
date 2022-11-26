import os
import pdb
from turtle import color
import pybullet as p
import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation
from collections import namedtuple


class Robot():
    def __init__(self):
        self.serverMode = p.GUI  # GUI/DIRECT
        self.robotUrdfPath = "/home/nidhi/masters_project/urdf/ur5_7thaxis.urdf"
        # connect to engine servers
        # Read the point cloud
        self.tree = "/home/nidhi/masters_project/meshes/ufo_trees_labelled/1_l.ply"
        self.physicsClient = p.connect(self.serverMode)

        # add search path for loadURDF
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # define world
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF("plane.urdf")
        # self.line = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
        #                             rgbaColor=[1, 0, 1, 1],
        #                             specularColor=[0.4, .4, 0],
        #                             visualFramePosition=[.58,.6,0],
        #                             radius=.01,
        #                             length= 2)

        # p.createMultiBody(baseVisualShapeIndex=self.line)
        # define robot
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
        # self.reset()
        # p.stepSimulation()

    def reset(self):
        self.move_joints(self.home_joints, "position")
        print("----------------------------------------")
        print("Robot moved to initial state")

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

    def move_joints(self, joint_values, control_type):
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

    def move_Onejoint(self, targets, joint_name,  control_type, include_prismatic=True):
        id = self.joint_names_to_ids[joint_name]
        if control_type == "position":
            controller = p.POSITION_CONTROL
            targetPositions = targets
            targetVelocities = 0
        elif control_type == "velocity":
            controller = p.VELOCITY_CONTROL
            targetPositions = 0
            targetVelocities = targets
        p.setJointMotorControl2(
            self.robotID, id, controller, targetPositions, targetVelocities)

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
            tf[:3, :3] = Rotation.from_quat(orientation).as_dcm()
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

    def start_sim(self, ):
        # start simulation
        try:
            flag = True
            textPose = list(p.getBasePositionAndOrientation(self.robotID)[0])
            textPose[2] += 1
            while (flag):
                p.stepSimulation()
            p.disconnect()
        except:
            p.disconnect()

    def __test__(self):
        self.reset()
        link_num = self.get_links()
        self.start_sim(link_num)


if __name__ == "__main__":
    robot = Robot()
    robot.__test__()
