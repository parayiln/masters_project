
#!/usr/bin/env python
from __future__ import print_function

import pybullet as p
import sensor_rgbd as cam
import robot_bullet
import numpy as np

class BulletController():
    def __init__(self):
        self.sensor = cam.Sensor()
        self.robot = robot_bullet.Robot()
        self.cam_offset = -.05
        self.ef_tree = .322
        self._ef_velocity_z = 0.005
        self.tree_location = -.6

        ################################################

    def move_up_down(self, direction, control_type, points_, vel_x):
        ee_pose, ee_oreint = self.robot.get_link_kinematics(
            'wrist_3_link-tool0_fixed_joint', as_matrix=False)
        link_id = self.robot.convert_link_name('cutpoint')
        orient = []
        for i in ee_oreint:
            orient.append(i)
        if control_type == "position":
            joint_value = p.calculateInverseKinematics(
                self.robot.robotID, link_id, [points_[2],points_[0],points_[1]], orient)
        elif control_type == "velocity":
            if direction == 'down':
                end_eff_vel = np.array([vel_x, 0, -self._ef_velocity_z, 0, 0, 0])
            elif direction == 'up':
                end_eff_vel = np.array([vel_x, 0, self._ef_velocity_z, 0, 0, 0])

            # print("end eff", end_eff_vel)
            joint_value = self.robot.getInverseVelocityKinematics(end_eff_vel)
        self.robot.move_joints(joint_value, control_type)

 #######################################################       

    def move_end_effector_ik(self, link_name_or_id, target_position, target_orientation=None, threshold=0.005, retries=3):
        for _ in range(retries):
            ik = self.robot.solve_end_effector_ik(link_name_or_id, target_position, target_orientation=target_orientation,
                                                  threshold=threshold)
            self.reset_joint_states(ik)
            pos, _ = self.get_link_kinematics(link_name_or_id)
            offset = np.linalg.norm(np.array(target_position) - pos[0])
            if offset < threshold:
                return True
        return False
############################################################


    def traj_on_tree(self, pixel, tool):

        view_matrix = self.sensor.view_matrix
        tran_pix_world = np.linalg.inv(view_matrix).T.round(3)
        scale = .0001/self.ef_tree  # distance from ef to tree is .322

        point = np.matmul(tran_pix_world, [[1, 0, 0, (pixel[0]*scale)], [
                          0, 1, 0, (pixel[1]*scale)], [0, 0, 1, (self.tree_location+self.ef_tree)], [0, 0, 0, 1]])
        point_ef = np.matmul(tran_pix_world, np.transpose([0, 0, 0, 1]))
        p.addUserDebugPoints(
            [[point[0][3], point[1][3], point[2][3]]], [[1, 1, 1]], 3)
        p.addUserDebugPoints(
            [[point_ef[0], point_ef[1], point_ef[2]]], [[0, 0, 1]], 4)
        return [point[0][3], point[1][3], point[2][3]]


if __name__ == "__main__":

    i = .543
    control = BulletController()
    control.robot.reset()
    while (True):
        tool = control.robot.get_link_kinematics(
            'wrist_3_link-tool0_fixed_joint', as_matrix=True)
        control.sensor.load_cam(tool)
        control.robot.move_Onejoint(i, '7thjoint_prismatic')
        p.stepSimulation()
