#!/usr/bin/env python
# from enum import Flag
import rospy
from std_msgs.msg import String
import geometry_msgs
import moveit_commander
import sys
sys.path.append("/home/nidhi/masters_project/pruning_bullet")
import robot_bullet as robot
import controller as ctrl
import sensor_rgbd as cam
import pybullet as p
import time
import numpy as np
import matplotlib.pyplot as plt

class Interface():

    def __init__(self):
        self.control= ctrl.BulletController()
        # self.ros_control= ctrl.RosController()
        self.control.robot.reset()
        self.sensor_flag = True
        for i in range(100):
            p.stepSimulation()
        self.val =.54
        self.point_ =[0,0,0]
        self.pre_point = [0,0,0]
        self.vel_=[0.001,0.001]
        self.direction = "none"
        p.addUserDebugLine([.45,.6,0],[.45,.6,.85],[0,0,0],3,0)
        # p.addUserDebugLine([.53,.6,0],[.53,.6,.85],[0,0,0],3,0)

    def run_pybullet(self, dt):
        ee_pose, ee_oreint = self.control.robot.get_link_kinematics('cutpoint',as_matrix=False)
        tool= self.control.robot.get_link_kinematics('wrist_3_link-tool0_fixed_joint',as_matrix=True)
        if self.sensor_flag == True:
            
            # cutpoint=self.control.robot.get_link_kinematics('cutpoint',as_matrix=True)
            rgbimg=self.control.sensor.load_cam(tool)
            # print("new image")
            if self.control.sensor.followLeader == True: 
                v1, v2 , pts, mpt =self.control.sensor.scan_leader(rgbimg)
                self.pre_point = self.point_
                print("entered following, tool", tool[2][3], self.vel_)
                point= self.control.traj_on_tree(mpt, tool)
                # self.control.traj_on_tree(v2, tool)
                self.point_ =[point[0], point[1],point[2]]
                v= [0,0,-.01]
                for i in range(2):
                    self.vel_[i] =1*(self.point_[i]-self.pre_point[i])/dt+0*(self.point_[i]-self.pre_point[i])+0*(self.point_[i]-self.pre_point[i])/(dt*dt)
                    
                    # print("vel", self.vel_)
                    # print(self.point_)
                    # self.vel_[i] = v[i]      

        if self.control.sensor.centerLeader == False:
            print("entered centering")
            controller_type = "velocity"
            
            self.direction = self.control.sensor.leader_centering(rgbimg, tool)
            ee_pose, ee_oreint = self.control.robot.get_link_kinematics('cutpoint',as_matrix=False)
            self.point_= [ee_pose[0],ee_pose[1],ee_pose[2]]
            if controller_type == "position":
                joint_name = '7thjoint_prismatic'   
                joint_val = self.control.robot.get_joint_state(joint_name)    
                self.val = self.val-0.0001
                self.control.robot.move_Onejoint(self.val, joint_name, controller_type)


            elif controller_type =="velocity":
                self.val =-.1
                joint_name = '7thjoint_prismatic'
                    # self.val = self.control.robot.getInverseVelocityKinematics(end_eff_vel)
                self.control.robot.move_Onejoint(self.val, joint_name, controller_type)
                
        if self.control.sensor.followLeader == True: 
            self.sensor_flag = False
            control_type = "velocity"
            self.control.move_up_down(self.direction, control_type, self.point_, self.vel_)
            # print("move one step")
            if (tool[2][3] < .35 and self.direction =="down") or (tool[2][3] > .81 and self.direction =="up"):

                print("move to next tree")
                self.control.sensor.centerLeader = False
                self.control.sensor.followLeader = False
                self.control.sensor.move_curr_branch = True
                self.sensor_flag =True
            if dt%10 == 0:
                self.sensor_flag = True

        p.stepSimulation()


    def run_rosmoveit():
        pass
        







if __name__ == "__main__":

    interface = Interface()
    
    val=.54
    flag = True
    i=0

    while(i<100000):
        dt = i 
        interface.run_pybullet(dt)
        i=i+1
    print("done moving")
    time.sleep(100)




