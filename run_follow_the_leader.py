#!/usr/bin/env python
import sys
sys.path.append("/home/nidhi/masters_project")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import pybullet as p
# import controller as ctrl

class PIController():
    def __init__(self):
        # self.sensor = cam.Sensor()
        # self.robot = robot_bullet.Robot()
        self.cam_offset = -.05
        self.ef_tree = .322
        
        self.tree_location = -.6

    def traj_on_tree(self, pixel, tool):

        view_matrix = robot.view_matrix
        tran_pix_world = np.linalg.inv(view_matrix).T.round(3)
        scale = .0001/self.ef_tree  # distance from ef to tree is .322

        point = np.matmul(tran_pix_world, [[1, 0, 0, (pixel[0]*scale)], [
                          0, 1, 0, (pixel[1]*scale)], [0, 0, 1, (self.tree_location+self.ef_tree)], [0, 0, 0, 1]])
        point_ef = np.matmul(tran_pix_world, np.transpose([0, 0, 0, 1]))
        # p.addUserDebugPoints(
        #     [[point[0][3], point[1][3], point[2][3]]], [[1, 1, 1]], 3)
        p.addUserDebugPoints(
            [[point_ef[0], point_ef[1], point_ef[2]]], [[0, 0, 1]], 4)
        return [point[0][3], point[1][3], point[2][3]]




class Interface():

    def __init__(self):
        robot.reset()
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
        self.branch_no_to_scan = 2
        self.branch_lower_limit = 0.32
        self.branch_upper_limit = 0.84
        self.time_step = .1
        self.sensor_time = 0
        self.kp = 1.25
        self.ki = 0.05
        self.ini_time = time.time()
        self.cur_time = time.time()
        self.tree_out_of_sight = False
        self.joint_pos_curr = 0
        self._ef_velocity_z = 0.005

        self.fig_traj, self.ax_traj = plt.subplots(figsize=(15, 15))
        self.fig_joints_pos, self.ax_joints_pos = plt.subplots(
            figsize=(15, 15))
        # self.fig_joints_vel, self.ax_joints_vel = plt.subplots(figsize=(15, 15))
        # self.fig_joints_torque, self.ax_joints_torque = plt.subplots()
        # self.fig_joints_force, self.ax_joints_force = plt.subplots()
        # self.fig_joint_linear, self.ax_joint_linear = plt.subplots(
        #     3, figsize=(25, 25))

        self.time_cumulative = []
        self.bezier_world_x = []
        self.bezier_world_y = []
        self.ef_traj_x = []
        self.ef_traj_y = []
        self.joint_velocity = []
        self.joint_position = []
        self.joint_torque = []
        self.euler_joint = []
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
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)

#####################################

    def add_plot_labels(self, fig, legends, title, plot_name):
        fig.suptitle(title+" | kp ="+str(self.kp) +
                     " ki ="+str(self.ki), fontsize=10)
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
        img_process.leader_centered = False
        img_process.follow_leader = False
        img_process.no_of_branch_scaned = img_process.no_of_branch_scaned + 1
        img_process.move_curr_branch = True
        self.sensor_flag = True

        self.generate_plots(self.ax_traj,
                            self.bezier_world_x, self.bezier_world_y, "bezier x coordinate (m)", "bezier y coordinate")
        self.generate_plots(self.ax_traj,
                            self.ef_traj_x, self.ef_traj_y, "end effector x coordinate (m)", "end effector y coordinate")
        temp_pose = zip(*self.joint_position)
        temp_pose_euler = zip(*self.euler_joint)
        temp_vel = zip(*self.joint_velocity)
        temp_torque = zip(*self.joint_torque)

        # for i in range(1, 7):
            # self.generate_plots(
            #     self.ax_joints_pos, self.time_cumulative, temp_pose[i], "time in seconds", "Joint position (rad)")
            # self.generate_plots(
            #     self.ax_joints_pos, self.time_cumulative, temp_pose_euler[i], "time in seconds", "Joint position (rad)")
        #     self.generate_plots(
        #         self.ax_joints_vel, self.time_cumulative, temp_vel[i], "time in seconds", "Joint velocity (rad/s)")
        #     self.generate_plots(
        #         self.ax_joints_torque, self.time_cumulative, temp_torque[i], "time in seconds", "Joint torque (N/m)")
        #     self.generate_plots(
        #         self.ax_joints_force, self.time_cumulative, temp_torque[0], "time in seconds", "Linear joint force (N)")

        # self.generate_plots(
        #     self.ax_joint_linear[0], self.time_cumulative, temp_pose[0], "time in seconds", "Linear Joint position (m)")
        # self.generate_plots(
        #     self.ax_joint_linear[1], self.time_cumulative, temp_vel[0], "time in seconds", "Linear Joint velocity (m/s)")
        # self.generate_plots(
        #     self.ax_joint_linear[2], self.time_cumulative, temp_torque[0], "time in seconds", "Linear Joint force (N)")

        self.reset()

    ##############################################################
    def follow_leader_img_process(self, rgbimg, tool, time_):
        mask = img_process.image_to_mask(rgbimg)
        midpt = img_process.mask_to_curve(mask,rgbimg)
        if type(midpt) == type(None):
            self.tree_out_of_sight = True
            print("---- Tree out of sight  ------")
            self.update_after_one_branch()
        else:
            self.ef_position_x = tool[0][3]
            point = controller.traj_on_tree(midpt, tool)
            self.bezier = [
                point[0]+controller.cam_offset, point[1], point[2]]
            error = (self.bezier[0]-self.ef_position_x)  # /self.time_step

            self.error_integral = error*self.time_step+self.error_integral
            self.ef_velocity = (self.kp*error+self.ki * self.error_integral)
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

        euler_joint_vel = robot.handle_control_velocity(self.ef_velocity, self._ef_velocity_z,self.direction)
        euler_joint_pos = self.joint_pos_curr+euler_joint_vel * self.time_step
        self.euler_joint.append(euler_joint_pos)
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

    def run_pybullet(self):
        time_ = time.time()-self.ini_time
        tool = robot.get_link_kinematics(
            'wrist_3_link-tool0_fixed_joint', as_matrix=True)
        if self.sensor_flag == True:
            rgbimg = robot.load_cam(tool)
            
            if img_process.follow_leader == True:
                self.follow_leader_img_process(rgbimg, tool, time_)
        if img_process.leader_centered == False:
            self.move_to_center_leader(rgbimg, tool)

        elif img_process.follow_leader == True:
            self.move_to_follow_leader(time_, tool)
        p.stepSimulation()

    #############################################################################


if __name__ == "__main__":
    from robot_env import PybulletRobotSetup
    from image_processor import HSVBasedImageProcessor
    controller = PIController()
    img_process = HSVBasedImageProcessor()
    robot = PybulletRobotSetup()
    interface = Interface()


    while (True):
        
        interface.run_pybullet()

        if img_process.no_of_branch_scaned == interface.branch_no_to_scan:
            break
        if interface.tree_out_of_sight == True:
            break
    print("----Done Scanning -----")

    # interface.add_plot_labels(interface.fig_traj, [
    #                           "bezier", "end effctor", "Euler integral EE"], "End effector trajectory and Bezeir points", "traj/traj")
    # interface.add_plot_labels(interface.fig_joints_pos, [
    #                           "joint 1", "joint 2", "joint 3", "joint 4", "joint 5", "joint 6"], "Revolute joint positions", "joint_pos/joint_pos")
    # # interface.add_plot_labels(interface.fig_joints_vel, [
    # #                           "joint 1", "joint 2", "joint 3", "joint 4", "joint 5", "joint 6"], "Revolute joint velocity", "joint_vel/joint_vel")
    # # interface.add_plot_labels(interface.fig_joints_torque, [
    # #                           "joint 1", "joint 2", "joint 3", "joint 4", "joint 5", "joint 6"], "Revolute joint dynamics", "joint_force/joint_tor")
    # # interface.add_plot_labels(interface.fig_joints_force, [""], "Prismatic joint dynamics", "joint_force/joint_force")

    # interface.ax_joint_linear[0].set_title("Position vs time", fontsize=20)
    # # interface.ax_joint_linear[1].set_title("Velocity vs time", fontsize=20)
    # # interface.ax_joint_linear[2].set_title("Force vs time", fontsize=20)
    # # interface.add_plot_labels(
    # #     interface.fig_joint_linear, " ", "Linear Joint ", "linear/linear")

    data = np.asarray([interface.bezier_world_x,interface.bezier_world_y,interface.ef_traj_x,interface.ef_traj_y])
    pd.DataFrame(data).to_csv('results/traj_hsv.csv')
    time.sleep(5)
