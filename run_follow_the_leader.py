#!/usr/bin/env python
import sys
sys.path.append("/home/nidhi/masters_project/pruning_bullet")


import matplotlib.pyplot as plt
import numpy as np
import time
import pybullet as p
import controller as ctrl



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
        self.bezier = [0.45, 0.6, 0.8]
        self.ef_position_x = [0, 0, 0]
        self.ef_velocity = 0
        self.direction = "none"
        self.error_integral = 0
        self.branch_no_to_scan = 1
        self.branch_lower_limit = 0.32
        self.branch_upper_limit = 0.84
        self.time_step = 5
        self.kp = 10000
        self.ki = 0.005

        self.fig_traj, self.ax_traj = plt.subplots()
        self.fig_joints_pos, self.ax_joints_pos = plt.subplots()
        self.fig_joints_vel, self.ax_joints_vel = plt.subplots()
        self.fig_joints_torque, self.ax_joints_torque = plt.subplots()
        self.fig_joint_linear, self.ax_joint_linear = plt.subplots(3)

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

    def generate_plots(self, fig, ax, x_data, y_data):
        ax.plot(x_data, y_data)

#####################################

    def add_plot_labels(self, fig, legends, title, plot_name):
        fig.suptitle(title)
        fig.legend(legends)
        fig.savefig('/home/nidhi/masters_project/plots/'+plot_name+'.png')

#########################################

    def update_step_values(self, tool, dt, joint_pos, joint_vel, joint_torque):
        self.time_cumulative.append(dt)
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

        self.generate_plots(self.fig_traj, self.ax_traj,
                            self.bezier_world_x, self.bezier_world_y)
        self.generate_plots(self.fig_traj, self.ax_traj,
                            self.ef_traj_x, self.ef_traj_y)

        temp_pose = zip(*self.joint_position)
        temp_vel = zip(*self.joint_velocity)
        temp_torque = zip(*self.joint_torque)

        for i in range(1, 7):
            self.generate_plots(
                self.fig_joints_pos, self.ax_joints_pos, self.time_cumulative, temp_pose[i])
            self.generate_plots(
                self.fig_joints_vel, self.ax_joints_vel, self.time_cumulative, temp_vel[i])
            self.generate_plots(
                self.fig_joints_torque, self.ax_joints_torque, self.time_cumulative, temp_torque[i])

        self.generate_plots(
            self.fig_joint_linear, self.ax_joint_linear[0], self.time_cumulative, temp_pose[0])
        self.generate_plots(
            self.fig_joint_linear, self.ax_joint_linear[1], self.time_cumulative, temp_vel[0])
        self.generate_plots(
            self.fig_joint_linear, self.ax_joint_linear[2], self.time_cumulative, temp_torque[0])

        self.reset()

    ##############################################################
    def follow_leader_img_process(self, rgbimg, tool):
        p1, p2, pts, midpt = self.control.sensor.scan_leader(rgbimg)
        self.ef_position_x = tool[0][3]
        point = self.control.traj_on_tree(midpt, tool)
        self.bezier = [
            point[0]+self.control.cam_offset, point[1], point[2]]
        error = (self.bezier[0]-self.ef_position_x)/dt

        self.error_integral = error*dt+self.error_integral
        self.ef_velocity = (self.kp*error+self.ki *
                            self.error_integral+0*(error)/(dt*dt))

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

    def move_to_follow_leader(self, tool, control_type="velocity"):
        self.sensor_flag = False
        self.control.move_up_down(
            self.direction, control_type, self.bezier, self.ef_velocity)
        joint_pos, joint_vel, joint_torque = self.control.robot.getJointStates()
        self.update_step_values(
            tool, dt, joint_pos, joint_vel, joint_torque)

        if (tool[2][3] < self.branch_lower_limit and self.direction == "down") or (tool[2][3] > self.branch_upper_limit and self.direction == "up"):
            print("--- Finished scanning current branch moving to next --- ")
            self.update_after_one_branch()

        if dt % self.time_step == 0:
            self.sensor_flag = True

    ##########################################################

    def run_pybullet(self, dt):
        tool = self.control.robot.get_link_kinematics(
            'wrist_3_link-tool0_fixed_joint', as_matrix=True)
        if self.sensor_flag == True:
            rgbimg = self.control.sensor.load_cam(tool)
            if self.control.sensor.follow_leader == True:
                self.follow_leader_img_process(rgbimg, tool)
        if self.control.sensor.leader_centered == False:
            self.move_to_center_leader("velocity", rgbimg, tool)

        elif self.control.sensor.follow_leader == True:
            self.move_to_follow_leader(tool)
        p.stepSimulation()
    #############################################################################


if __name__ == "__main__":

    interface = Interface()
    i = 0
    while (True):
        dt = i
        interface.run_pybullet(dt)
        i = i+1
        if interface.control.sensor.no_of_branch_scaned == interface.branch_no_to_scan:
            break
    print("----Done Scanning -----")

    interface.add_plot_labels(interface.fig_traj, [
                              "bezier", "end effctor"], "End effector trajectory and Bezeir points", "traj")
    interface.add_plot_labels(interface.fig_joints_pos, [
                              "joint 1", "joint 2", "joint 3", "joint 4", "joint 5", "joint 6"], "Revolute joint positions", "joint_pos")
    interface.add_plot_labels(interface.fig_joints_vel, [
                              "joint 1", "joint 2", "joint 3", "joint 4", "joint 5", "joint 6"], "Revolute joint velocity", "joint_vel")
    interface.add_plot_labels(interface.fig_joints_torque, [
                              "joint 1", "joint 2", "joint 3", "joint 4", "joint 5", "joint 6"], "Revolute joint torque", "joint_tor")

    interface.ax_joint_linear[0].set_title("Position vs time")
    interface.ax_joint_linear[1].set_title("Velocity vs time")
    interface.ax_joint_linear[2].set_title("Force vs time")
    interface.add_plot_labels(interface.fig_joint_linear, " ", "Linear Joint", "linear")

    time.sleep(10)
