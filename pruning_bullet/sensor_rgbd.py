

from ast import Pass
from turtle import width
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import time
import pybullet_data
import robot_bullet 
import cv2 as cv
import pca as pca

class Sensor():
    def __init__(self):

        self.width = 424
        self.height = 240
        self.fov = 42 # changes to get the cutter in view
        self.aspect = self.width / self.height
        self.near = 0.01
        self.far = .35
        self.center_leader=False
        self.follow_leader=False
        self.first_scan = True
        self.Leader_Tree = False
        self.traj =[]
        self.vel=[]
        self.move_curr_branch = False
        self.no_of_branch_scaned = 0

    def read_cam_video(self, img):
        # frame = img[3]
        frame=img[:,:,:]
        return frame

    def show_cam_video(self, img):
        cv.imshow('centering', img)
        cv.waitKey(1)

    
    
    def load_cam(self, tool):
        # self.view_matrix=np.linalg.inv(vm).T.reshape(-1)
        pose=[tool[0][3],tool[1][3],tool[2][3]]
        # pose_target=[cutpoint[0][3],cutpoint[1][3],cutpoint[2][3]]

        pose_target = np.dot(tool, np.array([0,0,.3,1]))[:3]
        self.view_matrix=np.reshape(p.computeViewMatrix(cameraEyePosition = pose,cameraTargetPosition =pose_target,cameraUpVector=[0, 0, 1]),(4,4))
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        _, _, rgb_img, raw_depth_img, raw_seg_img = p.getCameraImage(width = self.width,
                          height = self.height,
                          viewMatrix = self.view_matrix.reshape(-1),
                          projectionMatrix = self.projection_matrix,
                          shadow=True,
                          renderer = p.ER_BULLET_HARDWARE_OPENGL)

        return rgb_img


    def leader_on_right(self, y , center_y):
        if y- center_y>0:
            self.Leader_Tree = False
        else:
            self.Leader_Tree=True

    def leader_centering(self, rgb_img, tool):     
        img = self.read_cam_video(rgb_img)
        direction = "none"
        height = img.shape[0]
        width = img.shape[1]
        lower_red = np.array([60,40,40])
        upper_red = np.array([140,255,255])
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        thresh = cv.inRange(hsv, lower_red, upper_red)
        # thresh = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        # print(thresh)
        # ret,thresh = cv.threshold(mask,127,255,cv.THRESH_BINARY_INV)
        im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # cv.imshow('test', mask)
        # cv.waitKey(1) 
        cv.drawContours(img, contours, -1, (0,255,0), 3)
        cnt_y =[]
        for c in contours:
            M = cv.moments(c)
            if M["m00"] !=0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX =0
                cY =0
            cv.circle(img, (cX, cY), 2, (255, 255, 255), -1)
            cnt_y.append(cX)


        if len(cnt_y) != 0: 
            print(cnt_y)
            for y in cnt_y:
                Leaders_y =[]
                self.leader_on_right(y,int(width/2))
                if self.Leader_Tree == True:
                    Leaders_y.append(y)                
       
            
                
        if self.move_curr_branch == True:
            print("move out of the wayyyyy", cnt_y[-1], width)
            if cnt_y[-1] - int(2*width/3) == 0:
                self.move_curr_branch=False
                print("------moved the current leader out of sight-------")  

        else:
            if self.Leader_Tree == True:
                print("setting up next branch")
                if Leaders_y[-1] - int(width/2) == 0:
                    self.center_leader=True
                    print("------centered! starting to follow the leader -------")
                    self.follow_leader=True

                    if tool[2][3]<.4:
                        direction = "up"
                    else:
                        direction = "down"
                else:
                    self.center_leader=False
    
            

        
        cv.line(img, (int(width/2), 0), (int(width/2),height), (255,0,0), 2)
        
        self.show_cam_video(img)
        return direction


    def scan_leader(self, rgb_img):
        img = self.read_cam_video(rgb_img)
        self.show_cam_video(img)
        for k in range(0,3):
            name = str(k)
            self.pca = pca.PCA(rgb_img)
            self.pca.get_centers()
            self.pca.get_center_onSameTree(width)
            flag = 0
            v0 = np.array([self.pca.vector_list_sameTree[0][0], self.pca.vector_list_sameTree[0][1]])
            v1=np.array([self.pca.vector_list_sameTree[1][0],self.pca.vector_list_sameTree[1][1]])
            v2=np.array([self.pca.vector_list_sameTree[2][0],self.pca.vector_list_sameTree[2][1]])
            self.quad = pca.Quad(v0, v2, 10, v1)
            img, m_pt, traj = self.quad.draw_quad(self.pca.img, v0,v2)
            a, b = self.quad.tangent_curve_points(m_pt)
            max=len(self.pca.vector_list_sameTree)
            for j in range(0,max):
                cv.circle(self.pca.img, (self.pca.vector_list_sameTree[j][0], self.pca.vector_list_sameTree[j][1]), 1, (255, 255, 255), -1)
            self.quad.drawAxis(img, b, a, (0,0,255), 2)
            self.pca.save_image("{}_.png".format(name))
            for i in range(len(self.quad.traj)):
                self.traj.append(self.quad.traj[i])
                self.vel.append(self.quad.vel[i])
        return b, a, traj, m_pt
            # cv.waitKey(200)

 

        self.show_cam_video(self.pca.img)
        cv.waitKey(1)

    def get_delta_X(self):
          pass

# Get depth values using the OpenGL renderer
if __name__ == "__main__":
    robot = robot_bullet.Robot()
    robot.reset()
    p.stepSimulation()
    s=Sensor()
    t=0
    # tf= np.identity(4)
    while(True):
        tool= robot.get_link_kinematics('wrist_3_link-tool0_fixed_joint',as_matrix=True)
        # time.sleep(1)
        s.load_cam(tool)
        t=t+1
        p.stepSimulation()


