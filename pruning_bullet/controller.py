
#!/usr/bin/env python
from __future__ import print_function
import os
import time
import pdb
import pybullet as p
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import functools
import sensor_rgbd as cam
import robot_bullet 
import numpy as np
import rospy
from std_msgs.msg import String
import geometry_msgs
import moveit_commander

import moveit_msgs.msg
import geometry_msgs.msg
import sys

from six.moves import input

import sys
import copy
import rospy


# try:
#     from math import pi, tau, dist, fabs, cos
# except:  # For Python 2 compatibility
#     from math import pi, fabs, cos, sqrt

#     tau = 2.0 * pi

#     def dist(p, q):
#         return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

## END_SUB_TUTORIAL


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True

class BulletController():
    def __init__(self): 
        self.sensor=cam.Sensor()
        self.robot= robot_bullet.Robot()
        self.count=0
        self.cam_offset = -.05
        self.ef_tree = .322

    def move_up_down(self, direction, control_type, points_, vel_x):
        ee_pose, ee_oreint = self.robot.get_link_kinematics('cutpoint',as_matrix=False)
        link_id = self.robot.convert_link_name('cutpoint')
        pose=[]
        orient = []
        for i in ee_oreint:
            orient.append(i)
        # pose.append(ee_pose[0])
        # pose.append(ee_pose[1])
        if direction == 'down':
            # pose.append(ee_pose[2]-.005)
            pose.append(points_[2])
            pose.append(points_[0])
            pose.append(points_[1])
        else:
            pose.append(ee_pose[2]+.005)
        if control_type == "position":
            joint_value = p.calculateInverseKinematics(self.robot.robotID,link_id,pose, orient)
        elif control_type == "velocity":
            joint_angles = np.array(self.robot.home_joints)
            joint_velocities = np.array([0, 0, 0, 0, 0, 0, 0])
            # end_eff_vel = self.robot.solveForwardVelocityKinematics(joint_angles, joint_velocities)
            if direction == 'down':
                end_eff_vel = np.array([vel_x, 0, -0.01, 0, 0, 0])
            elif direction == 'up':
                end_eff_vel = np.array([vel_x, 0, 0.01, 0, 0, 0])

            # print("end eff", end_eff_vel)
            joint_value = self.robot.getInverseVelocityKinematics(end_eff_vel)
        self.robot.move_joints(joint_value,control_type)


 
        
    def move_end_effector_ik(self, link_name_or_id, target_position, target_orientation=None, threshold=0.005, retries=3):
        # Hack for the fact that somehow the IK doesn't get solved properly
        for _ in range(retries):
            ik = self.robot.solve_end_effector_ik(link_name_or_id, target_position, target_orientation=target_orientation,
                                            threshold=threshold)
            self.reset_joint_states(ik)
            pos, _ = self.get_link_kinematics(link_name_or_id)
            offset = np.linalg.norm(np.array(target_position) - pos[0])
            if offset < threshold:
                return True
        return False

    def traj_on_tree(self, pixel, tool):
        # for i in range(1):
        # tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix)).round(1)
        pose=[tool[0][3],tool[1][3],tool[2][3]]
        # pose_target=[cutpoint[0][3],cutpoint[1][3],cutpoint[2][3]]

        pose_target = np.dot(tool, np.array([0,0,.3,1]))[:3]
        view_matrix=np.reshape(p.computeViewMatrix(cameraEyePosition = pose,cameraTargetPosition =pose_target,cameraUpVector=[0, 0, 1]),(4,4))
        tran_pix_world = np.linalg.inv(view_matrix).T.round(3)
        scale = .0001/self.ef_tree  # distance from ef to tree is .322

        point = np.matmul(tran_pix_world,[[1,0,0,(pixel[0]*scale)],[0,1,0,(pixel[1]*scale)],[0,0,1,(-.6+.322-.1)],[0,0,0,1]])
        point_ef = np.matmul(tran_pix_world,np.transpose([0,0,0,1]))
        p.addUserDebugPoints([[point[0][3],point[1][3],point[2][3]]],[[1,1,1]],3)
        p.addUserDebugPoints([[point_ef[0],point_ef[1],point_ef[2]]],[[0,0,1]],4)
        # time.sleep(2)
        # print(pixel, point[0][3], point[1][3], point[2][3])
        self.count =self.count+.001
        return [point[0][3],point[1][3],point[2][3]]  





if __name__ == "__main__":

    i=.543

    flag = True
    control= BulletController()
    control.robot.reset()
    while(flag):
        tool= control.robot.get_link_kinematics('wrist_3_link-tool0_fixed_joint',as_matrix=True)
        control.sensor.load_cam(tool)
        control.robot.move_Onejoint(i, '7thjoint_prismatic')
        # control.move_joints([0,0,0,0,0,0,0])
        i =i-.001
        p.stepSimulation()
