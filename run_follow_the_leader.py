#!/usr/bin/env python
import sys
sys.path.append("/home/nidhi/masters_project")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

class PIController():
    def __init__(self):
        self.cam_offset = -.05
        self.end_effector_to_tree = .322
        self.tree_location = -.6
        self.kp = 1.25
        self.ki = 0.05
        self.error_integral =0

    def get_pi_values(self, curr_pt, desired_pt, time_step):
        ee_position_x = curr_pt[0][3]
        point = self.pixel_to_world(desired_pt)
        bezier = [point[0]+self.cam_offset, point[1], point[2]]
        error = (bezier[0]-ee_position_x)  # /self.time_step

        self.error_integral = error*time_step+self.error_integral
        ee_velocity = (self.kp*error+self.ki * self.error_integral)
        return ee_velocity, bezier
            

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




class Interface():

    def __init__(self):
        robot.reset()
        self.sensor_flag = True
        self.ini_joint_pos_control = .54
        self.cmd_joint_vel_control = -.1
        self.bezier = [0., 0., 0.]
        self.ee_velocity = 0
        self.direction = "none"
        self.error_integral = 0
        self.branch_no_to_scan = 2
        self.branch_lower_limit = 0.32
        self.branch_upper_limit = 0.84
        self.time_step = .1
        self.sensor_time = 0

        self.ini_time = time.time()
        self.cur_time = time.time()
        self.tree_out_of_sight = False
        self.joint_pos_curr = 0
        self.ee_velocity_z = 0.005
        self.time_cumulative = []
        self.bezier_world_x = []
        self.bezier_world_y = []
        self.ef_traj_x = []
        self.ef_traj_y = []
        self.joint_velocity = []
        self.joint_position = []
        self.joint_torque = []
        # self.euler_joint = []
        # p.addUserDebugLine([.45,.6,0],[.45,.6,.85],[0,0,0],3,0)

#######################################

    def reset(self):
        self.time_cumulative = []
        self.bezier_world = []
        self.ef_traj = []
        self.error_integral = 0
        self.joint_velocity = []
        self.joint_position = []
        self.joint_torque = []

#####################################

    def update_step_values(self, tool, time_, joint_pos, joint_vel, joint_torque):
        self.time_cumulative.append(time_)
        self.bezier_world_x.append(self.bezier[0])
        self.bezier_world_y.append(self.bezier[2])
        self.ef_traj_x.append(tool[0][3])
        self.ef_traj_y.append(tool[2][3])
        self.joint_position.append(joint_pos)
        self.joint_velocity.append(joint_vel)
        self.joint_torque.append(joint_torque)

    ################################################

    def update_after_one_branch(self):
        img_process.leader_centered = False
        img_process.follow_leader = False
        img_process.no_of_branch_scaned = img_process.no_of_branch_scaned + 1
        img_process.move_curr_branch = True
        self.sensor_flag = True
        self.reset()

    ##############################################################
    def get_follow_leader_velocity(self, rgbimg, tool, time_):
        mask = img_process.image_to_mask(rgbimg)
        midpt = img_process.mask_to_curve(mask,rgbimg)
        if type(midpt) == type(None):
            self.tree_out_of_sight = True
            print("---- Tree out of sight  ------")
            self.update_after_one_branch()
        else:
            self.ee_velocity, self.bezier = controller.get_pi_values(tool, midpt, self.time_step)
            self.sensor_time = time_

    ############################################################
    def move_to_center_leader(self, rgbimg, endeffector):
        mask = img_process.image_to_mask(rgbimg)
        if endeffector[2][3]<0.4:
            self.direction="up"
        elif endeffector[2][3]>.75:
            self.direction="down"
        img_process.mask_to_centered(mask,rgbimg)
        robot.handle_control_centering(self.cmd_joint_vel_control)

###########################################################

    def move_to_follow_leader(self, time_, tool):
        self.sensor_flag = False
        euler_joint_vel = robot.handle_control_velocity(self.ee_velocity, self.ee_velocity_z,self.direction)
        joint_pos, joint_vel, joint_torque = robot.getJointStates()
        self.update_step_values(
            tool, time_, joint_pos, joint_vel, joint_torque)
        self.joint_pos_curr = joint_pos
        if (tool[2][3] < self.branch_lower_limit and self.direction == "down") or (tool[2][3] > self.branch_upper_limit and self.direction == "up"):
            print("--- Finished scanning current branch moving to next --- ")
            self.update_after_one_branch()
        if time_ - self.sensor_time > self.time_step:
            self.sensor_flag = True

    ##########################################################

    def run(self):
        time_ = time.time()-self.ini_time
        tool = robot.get_link_kinematics(
            'wrist_3_link-tool0_fixed_joint', as_matrix=True)
        if self.sensor_flag == True:
            rgbimg = robot.load_cam(tool)            
            if img_process.follow_leader == True:
                self.get_follow_leader_velocity(rgbimg, tool, time_)
        if img_process.leader_centered == False:
            self.move_to_center_leader(rgbimg, tool)

        elif img_process.follow_leader == True:
            self.move_to_follow_leader(time_, tool)
        robot.robot_step()

    #############################################################################


if __name__ == "__main__":
    from robot_env import PybulletRobotSetup
    from image_processor import HSVBasedImageProcessor
    controller = PIController()
    img_process = HSVBasedImageProcessor()
    robot = PybulletRobotSetup()
    interface = Interface()
    while (True):       
        interface.run()
        if img_process.no_of_branch_scaned == interface.branch_no_to_scan:
            break
        if interface.tree_out_of_sight == True:
            break
    print("----Done Scanning -----")
    data = np.asarray([interface.bezier_world_x,interface.bezier_world_y,interface.ef_traj_x,interface.ef_traj_y])
    pd.DataFrame(data).to_csv('results/traj_hsv.csv')
    time.sleep(5)
