#!/usr/bin/env python

import sys
sys.path.append("/home/nidhi/masters_project/pruning_pybullet")


import controller as ctrl
import pybullet as p
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



class Interface():

    def __init__(self):
        self.control = ctrl.BulletController()
        # self.ros_control= ctrl.RosController()
        self.control.robot.reset()
        self.sensor_flag = True
        for i in range(100):
            p.stepSimulation()
        self.ini_joint_pos_control = .54
        self.cmd_joint_vel_control = -.1
        self.bezier = [0., 0., 0.]
        self.ef_position_x = [0, 0, 0]
        self.ef_velocity = 0
        self.direction = "none"
        self.error_integral = 0
        self.branch_no_to_scan = 3
        self.branch_lower_limit = 0.32
        self.branch_upper_limit = 0.84
        self.time_step = .1
        self.sensor_time = 0
        self.kp = 0.125
        self.ki = 0.005
        self.ini_time = time.time()
        self.cur_time = time.time()
        self.tree_out_of_sight = False

        self.fig_traj, self.ax_traj = plt.subplots(figsize=(15, 15))
        self.fig_joints_pos, self.ax_joints_pos = plt.subplots(figsize=(15, 15))
        self.fig_joints_vel, self.ax_joints_vel = plt.subplots(figsize=(15, 15))
        self.fig_joints_torque, self.ax_joints_torque = plt.subplots(figsize=(15, 15))
        self.fig_joint_linear, self.ax_joint_linear = plt.subplots(
            3, figsize=(25, 25))

        self.time_cumulative = []
        self.bezier_world_x = []
        self.bezier_world_y = []
        self.ef_traj_x = []
        self.ef_traj_y = []
        self.joint_velocity = []
        self.joint_position = []
        self.joint_torque = []
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

######################################

    def generate_plots(self, ax, x_data, y_data, x_label, y_label):
        ax.plot(x_data, y_data)
        ax.set_xlabel(x_label, fontsize=20)
        ax.set_ylabel(y_label, fontsize=20)

#####################################

    def add_plot_labels(self, fig, legends, title, plot_name):
        fig.suptitle(title+" | kp ="+str(self.kp)+" ki ="+str(self.ki), fontsize=20)
        fig.legend(legends)
        fig.savefig('/home/nidhi/masters_project/plots/' +
                    plot_name+str(self.kp)+str(self.ki)+'0.png')

#########################################

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
        self.control.sensor.leader_centered = False
        self.control.sensor.follow_leader = False
        self.control.sensor.no_of_branch_scaned = self.control.sensor.no_of_branch_scaned + 1
        self.control.sensor.move_curr_branch = True
        self.sensor_flag = True

        self.generate_plots(self.ax_traj,
                            self.bezier_world_x, self.bezier_world_y, "bezier x coordinate (m)", "bezier y coordinate")
        self.generate_plots(self.ax_traj,
                            self.ef_traj_x, self.ef_traj_y, "end effector x coordinate (m)", "end effector y coordinate")

        temp_pose = zip(*self.joint_position)
        temp_vel = zip(*self.joint_velocity)
        temp_torque = zip(*self.joint_torque)

        for i in range(1, 7):
            self.generate_plots(
                self.ax_joints_pos, self.time_cumulative, temp_pose[i], "time in seconds", "Joint position (rad)")
            self.generate_plots(
                self.ax_joints_vel, self.time_cumulative, temp_vel[i], "time in seconds", "Joint velocity (rad/s)")
            self.generate_plots(
                self.ax_joints_torque, self.time_cumulative, temp_torque[i], "time in seconds", "Joint torque (N/m)")

        self.generate_plots(
            self.ax_joint_linear[0], self.time_cumulative, temp_pose[0], "time in seconds", "Linear Joint position (m)")
        self.generate_plots(
            self.ax_joint_linear[1], self.time_cumulative, temp_vel[0], "time in seconds", "Linear Joint velocity (m/s)")
        self.generate_plots(
            self.ax_joint_linear[2], self.time_cumulative, temp_torque[0], "time in seconds", "Linear Joint force (N)")

        self.reset()

    ##############################################################
    def follow_leader_img_process(self, rgbimg, tool, time_):
        midpt = self.control.sensor.scan_leader(rgbimg)
        if type(midpt) == type(None):
            self.tree_out_of_sight = True
            print("---- Tree out of sight  ------")
            self.update_after_one_branch()
        else:
            self.ef_position_x = tool[0][3]
            point = self.control.traj_on_tree(midpt, tool)
            self.bezier = [
                point[0]+self.control.cam_offset, point[1], point[2]]
            error = (self.bezier[0]-self.ef_position_x)/self.time_step

            self.error_integral = error*self.time_step+self.error_integral
            self.ef_velocity = (self.kp*error+self.ki *
                                self.error_integral+0*(error)/(self.time_step*self.time_step))
            self.sensor_time = time_

    ############################################################
    def move_to_center_leader(self, controller_type, rgbimg, tool):
        self.direction = self.control.sensor.leader_centering(rgbimg, tool)
        self.ef_current = [tool[0], tool[1], tool[2]]

        if controller_type == "position":
            joint_name = '7thjoint_prismatic'
            self.ini_joint_pos_control = self.ini_joint_pos_control-0.0001
            self.control.robot.move_Onejoint(
                self.ini_joint_pos_control, joint_name, controller_type)

        elif controller_type == "velocity":
            joint_name = '7thjoint_prismatic'
            self.control.robot.move_Onejoint(
                self.cmd_joint_vel_control, joint_name, controller_type)

###########################################################

    def move_to_follow_leader(self, time_, tool, control_type="velocity"):
        self.sensor_flag = False
        self.control.move_up_down(
            self.direction, control_type, self.bezier, self.ef_velocity)
        joint_pos, joint_vel, joint_torque = self.control.robot.getJointStates()
        self.update_step_values(
            tool, time_, joint_pos, joint_vel, joint_torque)

        if (tool[2][3] < self.branch_lower_limit and self.direction == "down") or (tool[2][3] > self.branch_upper_limit and self.direction == "up"):
            print("--- Finished scanning current branch moving to next --- ")
            self.update_after_one_branch()
        if time_ - self.sensor_time > self.time_step:
            self.sensor_flag = True

    ##########################################################

    def run_pybullet(self):
        time_ = time.time()-self.ini_time
        tool = self.control.robot.get_link_kinematics(
            'wrist_3_link-tool0_fixed_joint', as_matrix=True)
        if self.sensor_flag == True:
            rgbimg = self.control.sensor.load_cam(tool)
            if self.control.sensor.follow_leader == True:
                self.follow_leader_img_process(rgbimg, tool, time_)
        if self.control.sensor.leader_centered == False:
            self.move_to_center_leader("velocity", rgbimg, tool)

        elif self.control.sensor.follow_leader == True:
            self.move_to_follow_leader(time_, tool)
        p.stepSimulation()

    #############################################################################


if __name__ == "__main__":

    interface = Interface()

    while (True):
        interface.run_pybullet()
        if interface.control.sensor.no_of_branch_scaned == interface.branch_no_to_scan:
            break
        if interface.tree_out_of_sight == True:
            break
    print("----Done Scanning -----")

    interface.add_plot_labels(interface.fig_traj, [
                              "bezier", "end effctor"], "End effector trajectory and Bezeir points", "traj/traj")
    interface.add_plot_labels(interface.fig_joints_pos, [
                              "joint 1", "joint 2", "joint 3", "joint 4", "joint 5", "joint 6"], "Revolute joint positions", "joint_pos/joint_pos")
    interface.add_plot_labels(interface.fig_joints_vel, [
                              "joint 1", "joint 2", "joint 3", "joint 4", "joint 5", "joint 6"], "Revolute joint velocity", "joint_vel/joint_vel")
    interface.add_plot_labels(interface.fig_joints_torque, [
                              "joint 1", "joint 2", "joint 3", "joint 4", "joint 5", "joint 6"], "Revolute joint torque", "joint_force/joint_tor")

    interface.ax_joint_linear[0].set_title("Position vs time", fontsize=20)
    interface.ax_joint_linear[1].set_title("Velocity vs time", fontsize=20)
    interface.ax_joint_linear[2].set_title("Force vs time", fontsize=20)
    interface.add_plot_labels(
        interface.fig_joint_linear, " ", "Linear Joint ", "linear/linear")

    data = np.asarray(interface.bezier_world_x,interface.bezier_world_y,interface.ef_traj_x,interface.ef_traj_y)
    pd.DataFrame(data).to_csv('traj.csv')  
    time.sleep(1)
