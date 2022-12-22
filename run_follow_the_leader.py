#!/usr/bin/env python
import sys
sys.path.append("/home/nidhi/masters_project")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

class PIController():
    def __init__(self):
        self.cam_offset = -.05
        self.end_effector_to_tree = .322
        self.tree_location = -.6
        self.kp = 1.25
        self.ki = 0.05
        self.error_integral = 0

    def get_pi_values(self, ee_pos, desired_pt, time_step):
        end_effector_position_x = ee_pos[0]
        point = self.pixel_to_world(desired_pt)
        bezier = [point[0]+self.cam_offset, point[1], point[2]]
        error = (bezier[0]-end_effector_position_x)  # /self.time_step

        self.error_integral = error*time_step+self.error_integral
        end_effector_velocity = (self.kp*error+self.ki * self.error_integral)
        return end_effector_velocity, bezier
            

    def pixel_to_world(self, pixel):

        view_matrix = robot.view_matrix
        tran_pix_world = np.linalg.inv(view_matrix).T.round(3)
        scale = .0001/self.end_effector_to_tree  # distance from ef to tree is .322

        point = np.matmul(tran_pix_world, [[1, 0, 0, (pixel[0]*scale)], [
                          0, 1, 0, (pixel[1]*scale)], [0, 0, 1, (self.tree_location+self.end_effector_to_tree)], [0, 0, 0, 1]])
        point_ef = np.matmul(tran_pix_world, np.transpose([0, 0, 0, 1]))
        # p.addUserDebugPoints(
        #     [[point[0][3], point[1][3], point[2][3]]], [[1, 1, 1]], 3)
        # p.addUserDebugPoints(
        #     [[point_ef[0], point_ef[1], point_ef[2]]], [[0, 0, 1]], 4)
        return [point[0][3], point[1][3], point[2][3]]




class RunFollowTheLeader():

    def __init__(self, robot, controller, img_process):
        self.robot = robot
        self.ini_joint_pos_control = .54
        self.cmd_joint_vel_control = -.1
        self.bezier = [0., 0., 0.]
        self.end_effector_velocity = 0
        self.direction = "none"
        self.error_integral = 0
        self.branch_no_to_scan = 2
        self.branch_lower_limit = 0.32
        self.branch_upper_limit = 0.84
        self.last_time = 0.0
        self.controller_freq = 20
        self.controller = controller
        self.img_process = img_process
        self.tree_out_of_sight = False
        self.joint_pos_curr = 0
        self.end_effector_velocity_z = 0.005
        self.bezier_world_x = []
        self.bezier_world_y = []
        self.ef_traj_x = []
        self.ef_traj_y = []
        self.joint_velocity = []
        self.joint_position = []
        self.joint_torque = []
        # self.euler_joint = []
        # p.addUserDebugLine([.45,.6,0],[.45,.6,.85],[0,0,0],3,0)

        robot.reset()


    def reset(self):
        self.bezier_world = []
        self.ef_traj = []
        self.error_integral = 0
        self.joint_velocity = []
        self.joint_position = []
        self.joint_torque = []



    def update_step_values(self, ee_pos, joint_pos, joint_vel, joint_torque):
        self.bezier_world_x.append(self.bezier[0])
        self.bezier_world_y.append(self.bezier[2])
        self.ef_traj_x.append(ee_pos[0])
        self.ef_traj_y.append(ee_pos[2])
        self.joint_position.append(joint_pos)
        self.joint_velocity.append(joint_vel)
        self.joint_torque.append(joint_torque)


    def update_after_one_branch(self):
        self.img_process.leader_centered = False
        self.img_process.follow_leader = False
        self.img_process.no_of_branch_scaned = self.img_process.no_of_branch_scaned + 1
        self.img_process.move_curr_branch = True
        self.reset()

    
    def get_follow_leader_velocity(self, img, ee_pos):
        mask = self.img_process.image_to_mask(img)
        midpt = self.img_process.mask_to_curve(mask, img)
        if midpt is None:
            self.tree_out_of_sight = True
            print("---- Tree out of sight  ------")
            self.update_after_one_branch()
        else:
            self.end_effector_velocity, self.bezier = self.controller.get_pi_values(ee_pos, midpt, 1 / self.controller_freq)

    
    def move_to_center_leader(self, img, ee_pos):
        mask = self.img_process.image_to_mask(img)
        z_pos = ee_pos[2]
        if z_pos < 0.4:
            self.direction = "up"
        elif z_pos > .75:
            self.direction = "down"
        self.img_process.mask_to_update_center_flags(mask, img)
        robot.handle_control_centering(self.cmd_joint_vel_control)


    def move_to_follow_leader(self, ee_pos):
        euler_joint_vel = robot.handle_control_velocity(self.end_effector_velocity, self.end_effector_velocity_z,self.direction)
        joint_pos, joint_vel, joint_torque = robot.getJointStates()
        self.update_step_values(ee_pos, joint_pos, joint_vel, joint_torque)
        self.joint_pos_curr = joint_pos

        z_pos = ee_pos[2]
        if (z_pos < self.branch_lower_limit and self.direction == "down") or (z_pos > self.branch_upper_limit and self.direction == "up"):
            print("--- Finished scanning current branch moving to next --- ")
            self.update_after_one_branch()

    def run(self):

        time = robot.elapsed_time
        update = math.floor(time * self.controller_freq) != math.floor(self.last_time * self.controller_freq)
        if update:

            img = robot.get_rgb_image()
            ee_pos = robot.ee_position
            if self.img_process.follow_leader:
                self.get_follow_leader_velocity(img, ee_pos)
            if not self.img_process.leader_centered:
                self.move_to_center_leader(img, ee_pos)
            elif self.img_process.follow_leader:
                self.move_to_follow_leader(ee_pos)

        robot.robot_step()
        self.last_time = time


if __name__ == "__main__":
    from robot_env import PybulletRobotSetup
    from image_processor import HSVBasedImageProcessor
    picontroller = PIController()
    imgprocess = HSVBasedImageProcessor()
    robot = PybulletRobotSetup()
    run_FollowTheLeader = RunFollowTheLeader(robot, picontroller, imgprocess)

    while True:
        run_FollowTheLeader.run()
        if imgprocess.no_of_branch_scaned == run_FollowTheLeader.branch_no_to_scan:
            break
        if run_FollowTheLeader.tree_out_of_sight == True:
            break
    print("----Done Scanning -----")
    data = np.asarray([run_FollowTheLeader.bezier_world_x,run_FollowTheLeader.bezier_world_y,run_FollowTheLeader.ef_traj_x,run_FollowTheLeader.ef_traj_y])
    pd.DataFrame(data).to_csv('results/traj_hsv.csv')
    time.sleep(5)
