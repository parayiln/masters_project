

from ast import Pass
from turtle import width
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import robot_bullet
import cv2 as cv
import time
import pca as pca


class Sensor():
    def __init__(self):

        self.width = 424
        self.height = 240
        self.fov = 42  # changes to get the cutter in view
        self.aspect = self.width / self.height
        self.near = 0.01
        self.far = .35
        self.leader_centered = False
        self.follow_leader = False
        self.first_scan = True
        self.Leader_Tree = False
        self.traj = []
        self.vel = []
        self.move_curr_branch = False
        self.no_of_branch_scaned = 0
        self.thres_out_of_sight =100
####################################################################

    def read_cam_video(self, img):
        # frame = img[3]
        frame = img[:, :, :]
        return frame
####################################################################

    def show_cam_video(self, img):
        cv.imshow('centering', img)
        cv.waitKey(1)
####################################################################

    def load_cam(self, tool):
        pose = [tool[0][3], tool[1][3], tool[2][3]]
        pose_target = np.dot(tool, np.array([0, 0, .3, 1]))[:3]
        self.view_matrix = np.reshape(p.computeViewMatrix(
            cameraEyePosition=pose, cameraTargetPosition=pose_target, cameraUpVector=[0, 0, 1]), (4, 4))
        self.projection_matrix = p.computeProjectionMatrixFOV(
            self.fov, self.aspect, self.near, self.far)
        _, _, rgb_img, raw_depth_img, raw_seg_img = p.getCameraImage(width=self.width,
                                                                     height=self.height,
                                                                     viewMatrix=self.view_matrix.reshape(
                                                                         -1),
                                                                     projectionMatrix=self.projection_matrix,
                                                                     shadow=True,
                                                                     renderer=p.ER_BULLET_HARDWARE_OPENGL)

        return rgb_img
####################################################################

    def leader_on_right(self, y, center_y):
        if y - center_y > 0:
            self.Leader_Tree = False
        else:
            self.Leader_Tree = True
####################################################################

    def leader_centering_img_process(self, rgb_img):
        img = self.read_cam_video(rgb_img)
        
        height = img.shape[0]
        width = img.shape[1]
        lower_red = np.array([60, 40, 40])
        upper_red = np.array([140, 255, 255])
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        thresh = cv.inRange(hsv, lower_red, upper_red)
        im2, contours, hierarchy = cv.findContours(
            thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(img, contours, -1, (0, 255, 0), 3)
        cnt_y = []
        for c in contours:
            M = cv.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX = 0
                cY = 0
            cv.circle(img, (cX, cY), 2, (255, 255, 255), -1)
            cnt_y.append(cX)
        return cnt_y, width, height, img
####################################################################

    def get_right_leader(self, width, cnt_y):
        for y in cnt_y:
            Leaders_y = [0]
            self.leader_on_right(y, int(width/2))
            if self.Leader_Tree == True:
                Leaders_y.append(y)
        return Leaders_y
####################################################################

    def leader_centering(self, rgb_img, tool):

        direction = "none"
        cnt_y, width, heigth, img = self.leader_centering_img_process(rgb_img)
        if len(cnt_y) != 0:
            Leaders_y = self.get_right_leader(width, cnt_y)
        else:
            Leaders_y = [0]

        if self.move_curr_branch == True:
            print("move out of the wayyyyy", cnt_y)
            if max(cnt_y) - int(self.thres_out_of_sight+width/2) == 0:
                self.move_curr_branch = False
                print("------moved the current leader out of sight-------")

        else:
            if self.Leader_Tree == True:
                print("setting up next branch",(Leaders_y[-1] - int(width/2)))
                if Leaders_y[-1] - int(width/2) == 0:
                    self.leader_centered = True
                    print("------centered! starting to follow the leader -------")
                    self.follow_leader = True

                    if tool[2][3] < .4:
                        direction = "up"
                    else:
                        direction = "down"
                else:
                    self.leader_centered = False

        cv.line(img, (int(width/2), 0), (int(width/2), heigth), (255, 0, 0), 2)
        self.show_cam_video(img)
        
        return direction
 ####################################################################

    def scan_leader_bezier(self, v0, v1, v2, name, w, h):

        self.quad = pca.Quad(v0, v2, 10, v1)
        img, m_pt, traj = self.quad.draw_quad(self.pca.img, v0, v2)
        a, b = self.quad.tangent_curve_points(m_pt)

        self.quad.drawAxis(img, b, a, (0, 0, 255), 2, w, h)
        # self.pca.save_image(
        #         "/home/nidhi/masters_project/plots/"+"{}_.png".format(name))    
        max = len(self.pca.vector_list_sameTree)
        for j in range(0, max):
            cv.circle(
                self.pca.img, (self.pca.vector_list_sameTree[j][0], self.pca.vector_list_sameTree[j][1]), 1, (255, 255, 255), -1)

            for i in range(len(self.quad.traj)):
                self.traj.append(self.quad.traj[i])
                self.vel.append(self.quad.vel[i])
        return m_pt

####################################################################
    def scan_leader_img_process(self, rgb_img):
        img = self.read_cam_video(rgb_img)
        # cv.imwrite('/home/nidhi/masters_project/plots/center.png',img)
        height = img.shape[0]
        width = img.shape[1]
        self.show_cam_video(img)
        for k in range(0, 3):
            name = str(k)
            self.pca = pca.PCA(rgb_img)
            self.pca.get_centers()
            self.pca.get_center_onSameTree(width)
            if len(self.pca.vector_list_sameTree) <= 2:
                return None
            else:
                v0 = np.array([self.pca.vector_list_sameTree[0][0],
                               self.pca.vector_list_sameTree[0][1]])
                v1 = np.array([self.pca.vector_list_sameTree[1][0],
                               self.pca.vector_list_sameTree[1][1]])
                v2 = np.array([self.pca.vector_list_sameTree[2][0],
                               self.pca.vector_list_sameTree[2][1]])

                m_pt = self.scan_leader_bezier(v0, v1, v2, name, width, height)
        return m_pt

####################################################################
    def scan_leader(self, rgb_img):
        m_pt = self.scan_leader_img_process(rgb_img)
        return m_pt
        # cv.waitKey(200)


####################################################################
# Get depth values using the OpenGL renderer
if __name__ == "__main__":
    robot = robot_bullet.Robot()
    robot.reset()
    p.stepSimulation()
    s = Sensor()
    t = 0
    # tf= np.identity(4)
    while (True):
        tool = robot.get_link_kinematics(
            'wrist_3_link-tool0_fixed_joint', as_matrix=True)
        # time.sleep(1)
        s.load_cam(tool)
        t = t+1
        p.stepSimulation()
