#!/usr/bin/env python
import sys
sys.path.append("/home/nidhi/masters_project")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from enum import Enum


class StateMachine(Enum):
    START = 0
    SCANNING = 1
    LEADER_FOLLOW = 2
    DONE = 3
    SCANNING_MOVE_AWAY = 4

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

        # Environment
        self.robot = robot
        self.controller = controller
        self.img_process = img_process

        # Scanning behavior setup
        self.branch_no_to_scan = 2
        self.post_scan_move_dist = 0.10
        self.branch_lower_limit = 0.32
        self.branch_upper_limit = 0.84
        self.controller_freq = 20
        self.ini_joint_pos_control = .54
        self.cmd_joint_vel_control = -.1
        self.vertical_scan_velocity = 0.05

        # State variables
        self.state = StateMachine.START
        self.direction = 0
        self.num_branches_scanned = 0
        self.last_finished_scan_pos = None
        self.last_time = 0.0

        # Logging variables
        self.error_integral = 0
        self.bezier = [0., 0., 0.]
        self.bezier_world_x = []
        self.bezier_world_y = []
        self.ef_traj_x = []
        self.ef_traj_y = []
        self.joint_velocity = []
        self.joint_position = []
        self.joint_torque = []

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


    def mark_branch_as_complete(self, success):
        if success:
            self.num_branches_scanned += 1
        self.reset()

    def compute_velocity_from_curve(self, curve, ee_pos, execute=True):
        if curve is None:
            return None

        midpoint = curve.pt_axis(0.5)
        vel, self.bezier = self.controller.get_pi_values(ee_pos, midpoint, 1 / self.controller_freq)
        if execute:
            vel_array = np.array([vel, 0, self.vertical_scan_velocity * self.direction, 0, 0, 0])
            robot.handle_control_velocity(vel_array)
        return vel

    def leader_follow_is_done(self):
        if self.img_process.curve is None:
            return True
        ee_pos = self.robot.ee_position
        z_pos = ee_pos[2]
        if (z_pos < self.branch_lower_limit and self.direction < 0) or (
                z_pos > self.branch_upper_limit and self.direction > 0):
            return True
        return False


    def run(self):

        cur_time = self.robot.elapsed_time
        update = math.floor(cur_time * self.controller_freq) != math.floor(self.last_time * self.controller_freq)
        if update:

            start_state = self.state

            img = self.robot.get_rgb_image()
            ee_pos = self.robot.ee_position
            self.img_process.process_image(img)

            # STATE MACHINE TRANSITION LOGIC
            if self.state == StateMachine.START:
                if self.robot.has_linear_axis:
                    self.state = StateMachine.SCANNING
                else:
                    self.direction = -1
                    self.state = StateMachine.LEADER_FOLLOW

            elif self.state == StateMachine.SCANNING:

                center_dist = self.img_process.get_center_distance(normalize=False)
                if center_dist is not None and -10 < center_dist < 10:
                    # TODO: More robust switching criterion - Especially when it has just gotten done scanning a leader
                    self.robot.handle_linear_axis_control(0)
                    dist_from_lower = abs(ee_pos[2] - self.branch_lower_limit)
                    dist_from_upper = abs(ee_pos[2] - self.branch_upper_limit)
                    self.direction = 1 if dist_from_lower < dist_from_upper else -1
                    self.state = StateMachine.LEADER_FOLLOW
                else:
                    self.robot.handle_linear_axis_control(self.cmd_joint_vel_control)

            elif self.state == StateMachine.LEADER_FOLLOW:

                curve = self.img_process.curve
                self.compute_velocity_from_curve(curve, ee_pos, execute=True)
                if self.leader_follow_is_done():
                    self.mark_branch_as_complete(success=curve is not None)
                    robot.handle_control_velocity(np.zeros(6))

                    if self.num_branches_scanned >= self.branch_no_to_scan:
                        self.state = StateMachine.DONE
                    elif self.robot.has_linear_axis:
                        self.state = StateMachine.SCANNING_MOVE_AWAY
                        self.last_finished_scan_pos = ee_pos
                    else:
                        self.state = StateMachine.DONE

            elif self.state == StateMachine.SCANNING_MOVE_AWAY:
                if np.linalg.norm(self.last_finished_scan_pos - ee_pos) > self.post_scan_move_dist:
                    self.state = StateMachine.SCANNING
                else:
                    self.robot.handle_linear_axis_control(self.cmd_joint_vel_control)

            if self.state != start_state:
                print('STATE SWITCHED FROM {} to {}'.format(start_state, self.state))

        robot.robot_step()
        self.img_process.visualize()
        self.last_time = cur_time


if __name__ == "__main__":
    from robot_env import PybulletRobotSetup
    from image_processor import HSVBasedImageProcessor
    picontroller = PIController()
    imgprocess = HSVBasedImageProcessor()
    robot = PybulletRobotSetup()
    state_machine = RunFollowTheLeader(robot, picontroller, imgprocess)

    while True:
        state_machine.run()
        if state_machine.state == StateMachine.DONE:
            break

    print("----Done Scanning -----")
    data = np.asarray([state_machine.bezier_world_x,state_machine.bezier_world_y,state_machine.ef_traj_x,state_machine.ef_traj_y])
    pd.DataFrame(data).to_csv('results/traj_hsv.csv')
