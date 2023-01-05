#!/usr/bin/env python
import sys
# import pandas as pd
import numpy as np
import math
from enum import Enum


class StateMachine(Enum):
    START = 0
    SCANNING = 1
    LEADER_FOLLOW = 2
    DONE = 3
    SCANNING_MOVE_AWAY = 4
    INACTIVE = 5

class PIController():
    def __init__(self):
        self.cam_offset = -.05
        self.end_effector_to_tree = .322
        self.tree_location = -.6
        ## values for HSV
        # self.kp = 1.25
        # self.ki = 0.05
        # self.error_integral = 0

        #values for flownet
        self.kp = 0.00125
        self.ki = 0.00005
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

    def reset(self):
        self.error_integral = 0

class PixelBasedPIController:
    def __init__(self, camera_fov, pix_width, kp=1.0, ki=0.0):
        self.fov = camera_fov   # Horizontal FOV
        self.width = pix_width
        self.kp = kp
        self.ki = ki

        self._integral_error = 0.0

    def reset(self):
        self._integral_error = 0.0

    def get_pi_values(self, target_px, time_step):
        # Error is normalized to the FOV of the camera to make sure that different
        # image resolutions don't cause the controller to behave differently
        pix_err = target_px[0] - self.width / 2
        normalized_err = self.fov * pix_err / self.width
        self._integral_error += normalized_err * time_step
        correction = self.kp * normalized_err + self.ki * self._integral_error
        return correction


class FollowTheLeaderController():

    def __init__(self, robot, controller, img_process, visualize=False):

        # Environment
        self.robot = robot
        self.controller = controller
        self.img_process = img_process
        self.do_visualize = visualize

        # Scanning behavior setup
        self.branch_no_to_scan = 5
        self.post_scan_move_dist = 0.10
        self.branch_lower_limit = 0.32
        self.branch_upper_limit = .8
        # self.controller_freq = 20 #HSV            # Set to 0 if you want the controller to run every iteration
        self.controller_freq = 5   #flownet
        self.cmd_joint_vel_control = -.1
        self.scan_velocity = 0.075
        self.no_curve_tolerance = 5     # Number of missed curve detections before exiting follow the leader

        # State variables
        self.state = StateMachine.START
        self.direction = 0
        self.num_branches_scanned = 0
        self.last_finished_scan_pos = None
        self.last_time = 0.0
        self.current_target = None
        self.viz_arrows = []
        self.no_curve_counter = 0
        self.last_gradient = None

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

        self.cur_img = self.robot.get_rgb_image()

        self.robot.reset()


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

    def get_pixel_target_from_curve(self, curve, eval_pts=50):

        if curve is None:
            return None, None

        ts = np.linspace(0, 1, num=eval_pts)
        eval_pts = curve.pt_axis(ts)
        closest_idx = np.argmin(np.abs(eval_pts[:, 1] - self.robot.height / 2))
        target_pt = eval_pts[closest_idx]
        self.current_target = target_pt
        return target_pt, ts[closest_idx]

    def compute_velocity_from_curve(self, curve, ee_pos, time_elapsed, execute=True):
        target_pt, t = self.get_pixel_target_from_curve(curve)
        if target_pt is None:
            if self.last_gradient is not None:
                gradient = self.last_gradient
                vel = self.last_gradient * self.scan_velocity
                correction_term = 0.0
                target_pt = np.array([self.robot.width / 2, self.robot.height / 2])
            else:
                return None
        else:

            gradient = curve.tangent_axis(t)
            gradient /= np.linalg.norm(gradient)
            # Flip the sign of the gradient if the gradient is in the opposite direction of the scan direction
            # Note that positive gradient means moving down whereas positive direction means moving up, hence the ==
            if np.sign(gradient[1]) == self.direction:
                gradient *= -1
            self.last_gradient = gradient

            vel = gradient * self.scan_velocity
            correction_term = self.controller.get_pi_values(target_pt, time_elapsed)
            vel[0] += correction_term
            vel *= self.scan_velocity / np.linalg.norm(vel)

        if execute:
            vel_array = np.array([vel[0], vel[1], 0, 0, 0, 0])
            self.robot.handle_control_velocity(vel_array)

        # Visualization stuff
        vel_norm = vel / np.linalg.norm(vel)
        vl = 50
        self.viz_arrows = [
            [tuple(target_pt.astype(np.int64)), tuple((target_pt + gradient * vl).astype(np.int64)), (0, 255, 0)],
            [tuple(target_pt.astype(np.int64)), tuple((target_pt + np.array([(correction_term / self.scan_velocity), 0]) * vl).astype(np.int64)), (0, 0, 255)],
            [tuple(target_pt.astype(np.int64)), tuple((target_pt + vel_norm * vl).astype(np.int64)), (0, 255, 255)],
        ]

        return vel

    def leader_follow_is_done(self):
        # if self.img_process.curve is None:
        #     return True
        ee_pos = self.robot.ee_position
        z_pos = ee_pos[2]
        
        if (z_pos < self.branch_lower_limit and self.direction < 0) or (
                z_pos > self.branch_upper_limit and self.direction > 0):
            return True
        return False

    def step(self):
        if self.needs_update():
            # print('ROBOT UPDATING')
            time_elapsed = self.update_image()
            self.update_state_machine(time_elapsed)
        self.robot.robot_step()
        if self.do_visualize:
            self.visualize()

    def needs_update(self):
        cur_time = self.robot.elapsed_time
        update = math.floor(cur_time * self.controller_freq) != math.floor(self.last_time * self.controller_freq)
        self.last_time = cur_time
        return update

    def update_image(self):
        img = self.robot.get_rgb_image()
        self.img_process.process_image(img)
        return 1 / self.controller_freq

    def switch_to_leader_follow(self):
        self.state = StateMachine.LEADER_FOLLOW
        self.img_process.set_following_mode(True)

    def deactivate_leader_follow(self):
        self.robot.handle_control_velocity(np.zeros(6))
        self.controller.reset()
        self.img_process.reset()
        self.viz_arrows = []

        if self.num_branches_scanned >= self.branch_no_to_scan:
            self.state = StateMachine.DONE
        elif self.robot.has_linear_axis:
            self.state = StateMachine.SCANNING_MOVE_AWAY
            self.last_finished_scan_pos = self.robot.ee_position
        else:
            self.state = StateMachine.DONE

    def visualize(self):
        self.img_process.visualize(self.current_target, self.viz_arrows)

    def update_state_machine(self, time_elapsed):

        start_state = self.state
        ee_pos = self.robot.ee_position

        # STATE MACHINE TRANSITION LOGIC
        if self.state == StateMachine.START:
            if self.robot.has_linear_axis:
                self.state = StateMachine.SCANNING
            else:
                self.direction = -1
                self.switch_to_leader_follow()

        elif self.state == StateMachine.SCANNING:

            target, _ = self.get_pixel_target_from_curve(self.img_process.curve)
            if target is not None and -10 < target[0] - self.robot.width // 2 < 10:
                # TODO: More robust switching criterion - Especially when it has just gotten done scanning a leader
                self.robot.handle_linear_axis_control(0)
                dist_from_lower = abs(ee_pos[2] - self.branch_lower_limit)
                dist_from_upper = abs(ee_pos[2] - self.branch_upper_limit)
                self.direction = 1 if dist_from_lower < dist_from_upper else -1
                self.switch_to_leader_follow()
            else:
                self.robot.handle_linear_axis_control(self.cmd_joint_vel_control)
        
        elif self.state == StateMachine.LEADER_FOLLOW:

            curve = self.img_process.curve
            if curve is None:
                self.no_curve_counter += 1
                if self.no_curve_counter >= self.no_curve_tolerance:
                    self.img_process.reset()
                    self.img_process.set_following_mode(True)
                    self.no_curve_counter = 0
            else:
                self.no_curve_counter = 0

            self.compute_velocity_from_curve(curve, ee_pos, time_elapsed, execute=True)
            if self.leader_follow_is_done():
                print("this should not be the case")
                self.mark_branch_as_complete(success=curve is not None)
                self.deactivate_leader_follow()

        elif self.state == StateMachine.SCANNING_MOVE_AWAY:
            if np.linalg.norm(self.last_finished_scan_pos - ee_pos) > self.post_scan_move_dist:
                self.state = StateMachine.SCANNING
            else:
                self.robot.handle_linear_axis_control(self.cmd_joint_vel_control)

        if self.state != start_state:
            print('STATE SWITCHED FROM {} to {}'.format(start_state, self.state))


if __name__ == "__main__":
    from robot_env import PybulletRobotSetup
    from image_processor import FlowGANImageProcessor, HSVBasedImageProcessor
    # picontroller = PIController()
    # imgprocess = HSVBasedImageProcessor()
    robot = PybulletRobotSetup()
    img_processor = FlowGANImageProcessor((robot.width, robot.height))

    kp = 0.8
    pi_controller = PixelBasedPIController(np.radians(robot.fov), robot.width, kp=kp)
    state_machine = FollowTheLeaderController(robot, pi_controller, img_processor, visualize=True)

    while True:
        state_machine.step()
        if state_machine.state == StateMachine.DONE:
            break

    print("----Done Scanning -----")
    # data = np.asarray([state_machine.bezier_world_x,state_machine.bezier_world_y,state_machine.ef_traj_x,state_machine.ef_traj_y])
    # pd.DataFrame(data).to_csv('results/traj_hsv.csv')
