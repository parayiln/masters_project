from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
from math import atan2, cos, sin, sqrt, pi
from matplotlib import pyplot as plt
import os
import time


class Quad:
    def __init__(self, v1, v2, radius, mid_pt):
        self.bezier = np.zeros((2, 3))
        self.p0, self.p2, self.orientation = self._orientation((np.array(v1)), np.array(v2))
        if mid_pt is None:
            self.p1 = 0.5 * (self.p0 + self.p2)
        else:
            self.p1 = np.array(mid_pt)
        self.depth_values = []
        self.radius_2d = radius
        self.current_time=[]


    def _orientation(self, v1, v2):
        """Set the orientation and ensure left-right or right-left
        @param v1 vertex 1 from branchpointdetection
        @param v2 vertex 2
        @returns three points, orientation as a text string"""
        if abs(v1[1] - v2[1]) > abs(v1[0] - v2[0]):
            ret_orientation = "vertical"
            if v1[1] > v2[1]:
                p0 = v1
                p2 = v2
            else:
                p0 = v1
                p2 = v2
        else:
            ret_orientation = "horizontal"
            if v1[0] > v2[0]:
                p0 = v1
                p2 = v2
            else:
                p0 = v1
                p2 = v2
        return p0, p2, ret_orientation

    def pt_axis(self, t):
        """ Return a point along the bezier
        @param t in 0, 1
        @return 2d point"""
        pts = np.array([self.p0[i] * (1-t) ** 2 + 2 * (1-t) * t * self.p1[i] + t ** 2 * self.p2[i] for i in range(0, 2)])
        return pts.transpose()

        #return self.p0 * (1-t) ** 2 + 2 * (1-t) * t * self.p1 + t ** 2 * self.p2
   
    def diff_pt_axis(self, t):
        """ Return a point along the bezier
        @param t in 0, 1
        @return 2d point"""
        dy = np.array([self.p0[i] * (t-1)* 2+  2* self.p1[i] - 4 * t * self.p1[i] + t *2 * self.p2[i]for i in range(0, 2)])
        pts=dy
        return pts.transpose()
        #return self.p0 * (1-t) ** 2 + 2 * (1-t) * t * self.p1 + t ** 2 * self.p2
   


    def tangent_slope(self, m_pt):
            # compute dy/dx
        t= m_pt
        dy_dt = self.p0[1] * (t[1]-1)* 2+  2* self.p1[1] - 4 * t[1] * self.p1[1] + t[1] *2 * self.p2[1] 
        dx_dt = self.p0[0] * (t[0]-1)* 2+  2* self.p1[0] - 4 * t[0] * self.p1[0] + t[0] *2 * self.p2[0] 
        slope = -dx_dt/dy_dt
        return slope


    def tangent_curve_points(self, m_pt, scale=1):
        slope = self.tangent_slope(m_pt)
        x_a = int(m_pt[0]+scale)
        x_b = int(m_pt[0]-scale)
        y_a = m_pt[1]+int(slope*(x_a-m_pt[0]))
        y_b = m_pt[1]+int(slope*(x_b-m_pt[0]))
        return [x_a, y_a] , [x_b, y_b]

    def _setup_least_squares(self, ts):
        """Setup the least squares approximation
        @param ts - t values to use
        @returns A, B for Ax = b """
        # Set up the matrix - include the 3 current points plus the centers of the mask
        a_constraints = np.zeros((len(ts) + 3, 3))
        ts_constraints = np.zeros((len(ts) + 3))
        b_rhs = np.zeros((len(ts) + 3, 2))
        ts_constraints[-3] = 0.0
        ts_constraints[-2] = 0.5
        ts_constraints[-1] = 1.0
        ts_constraints[:-3] = np.transpose(ts)
        a_constraints[:, -3] = (1-ts_constraints) * (1-ts_constraints)
        a_constraints[:, -2] = 2 * (1-ts_constraints) * ts_constraints
        a_constraints[:, -1] = ts_constraints * ts_constraints
        for i, t in enumerate(ts_constraints):
            b_rhs[i, :] = self.pt_axis(ts_constraints[i])
        return a_constraints, b_rhs


    def draw_quad(self, im, a , b):
        """ Set the pixels corresponding to the quad to white
        @im numpy array"""
        n_pts_quad = 6
        image = im.copy()
        self.traj=[]
        self.vel =[]
        pts = self.pt_axis(np.linspace(0, 1, n_pts_quad))
        vel_pts = self.diff_pt_axis(np.linspace(0, 1, n_pts_quad))
        col_div = 120 // (n_pts_quad - 1)
        for p1, p2 in zip(pts[0:-1], pts[1:]):
            cv.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,255,0), thickness=1)
        for vel_p1 in vel_pts:
            # cv.circle(image, (int(vel_p1[0]), int(vel_p1[1])), 3, (0, 255, 0), 1)
            self.vel.append([vel_p1[0],vel_p1[1]])
        for p1 in pts:
            # cv.circle(image, (int(vel_p1[0]), int(vel_p1[1])), 3, (0, 255, 0), 1)
            self.traj.append([p1[0],p1[1]])           
        # cv.imshow('curve', image)
        # cv.waitKey(200)
        
        cv.line(image, (a[0],a[1]), (b[0],b[1]), (0,0,0), thickness=1)
        # cv.circle(image, (a[0],a[1]), 3, (0, 255, 0), 2)
        # cv.circle(image, (b[0],b[1]), 3, (0, 255, 0), 2)

        curve_mid_pt = pts[int(len(pts[0:-1])/2)]
        return image, curve_mid_pt, self.traj

    def drawAxis(self, img,  p, q, colour, scale):
        # p = list(self.curr_target)
        # q = list(q_)
        angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
        hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
        # Here we lengthen the arrow by a factor of scale
        q[0] = p[0] - scale * hypotenuse * cos(angle)
        q[1] = p[1] - scale * hypotenuse * sin(angle)
        cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
        # create the arrow hooks
        p[0] = q[0] + 9 * cos(angle + pi / 4)
        p[1] = q[1] + 9 * sin(angle + pi / 4)
        cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
        p[0] = q[0] + 9 * cos(angle - pi / 4)
        p[1] = q[1] + 9 * sin(angle - pi / 4)
        cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
        cv.imshow('follow the leader', img)
        cv.waitKey(1)        
        
        return hypotenuse




class PCA(object):
    def __init__(self, image):
        self.dir_labeled=r'/home/nidhi/ws_pruning/src/ur5_pruning_trails/scripts/pca_output'
        # self.dir_data_org = r'D:\academic\osu\imml\pruning\summer22\pca\labled_Tieqiao\original'
        # self.dir_data = r'D:\academic\osu\imml\pruning\summer22\pca\labled_Tieqiao\binary'
        # self.img, self.img_binary = self.get_image(image)
        # self.img_binary = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        self.lower_red = np.array([60,40,40])
        self.upper_red = np.array([140,255,255])
        self.hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        self.img_binary = cv.inRange(self.hsv, self.lower_red, self.upper_red)
        self.img = image
        # self.crop_h=110
        self.img_w= self.img.shape[1]
        self.img_h = self.img.shape[0]
        # self.iter= int(self.img_h/self.crop_h)
        self.iter =3
        self.crop_h = int(self.img_h/self.iter)
        self.cropped_height=0
        self.curr_target=0
        self.particle_curr_pose=[0,0]
        self.particle_pre_pose=[60,0]
        self.ctrl=0
        self.angle=0
        self.vector_len=0
        self.angle_pre=0
        self.vector_len_pre=0
        self.dt=10
        self.ki=.2
        self.kp=1
        self.kd=1
        self.pixel=100
        self.curve_point=[]
        self.vector_list=[]
        self.vector_list_sameTree=[]
        self.vec_list={}
        # cv.imshow('original image', self.img)
        # cv.waitKey(10)


    def drawAxis(self, img,  p_, q_, colour, scale):
        p = list(p_)
        q = list(q_)
        angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
        hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
        # Here we lengthen the arrow by a factor of scale
        q[0] = p[0] - scale * hypotenuse * cos(angle)
        q[1] = p[1] - scale * hypotenuse * sin(angle)
        cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
        # create the arrow hooks
        p[0] = q[0] + 9 * cos(angle + pi / 4)
        p[1] = q[1] + 9 * sin(angle + pi / 4)
        cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
        p[0] = q[0] + 9 * cos(angle - pi / 4)
        p[1] = q[1] + 9 * sin(angle - pi / 4)
        cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
        # cv.imshow('new', img)
        # cv.waitKey(1)        
        

    def get_contours_var(self,n):
        for i in range(n):
            self.vec_list[i]={}

    def check_con_num(self, con):
        num = con[1]
        return num



    def get_image(self, image):
        os.chdir(self.dir_data_org)
        img = cv.imread(cv.samples.findFile(image))

        os.chdir(self.dir_data)
        img_binary = cv.imread(cv.samples.findFile(image))

        # Check if image is loaded successfully
        if img is None:
            print('Could not open or find the image: ', args.input)
            exit(0)
        return img, img_binary

    def save_image(self,image):
        os.chdir(self.dir_labeled)
        cv.imwrite(image, self.img)


    def getOrientation(self, pts, img2):
        self.angle_pre=self.angle
        self.vector_len_pre=self.vector_len
        sz = len(pts)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i, 0] = pts[i, 0, 0]
            data_pts[i, 1] = pts[i, 0, 1]
        # Perform PCA analysis
        mean = np.empty((0))
        mean, eigenvectors = cv.PCACompute(data_pts, mean)
        # Store the center of the object
        cntr = (int(mean[0, 0])+0, int(mean[0, 1])+self.cropped_height)
        self.curr_target=cntr
        # cv.circle(self.img, self.curr_target, 3, (0, 255, 0), 2)
        p1 = (self.curr_target[0] + 0.02 * eigenvectors[0, 0] * eigenvectors[0, 0], self.curr_target[1]  + 0.02 * eigenvectors[0, 1] * eigenvectors[0, 0])
        p2 = (
        self.curr_target[0] -  eigenvectors[1, 0] , self.curr_target[1]  -  eigenvectors[1, 1] )
        # self.drawAxis(self.img, cntr, p1, (0, 255, 0), 5)
        self.drawAxis(self.img, cntr, p2, (0, 255, 0), 15)
        angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radian
        self.vector_list.append(cntr)
        # self.vector_list.append(p1)
        return angle, cntr

    def image_process(self, step):

        # cv.imshow('output', self.img)
        # cv.waitKey(0)
        self.cropped_height = 0 + self.crop_h * step
        cropped_image = self.img[self.cropped_height:self.cropped_height + self.pixel, 0:self.img.shape[1]]
        # hsv = cv.cvtColor(cropped_image, cv.COLOR_BGR2HSV)
        # bw = cv.inRange(hsv, (H_low, S_low, V_low), (H_high, S_high, V_high))
        bw=self.img_binary[self.cropped_height:self.cropped_height + self.pixel, 0:self.img.shape[1]]
        # bw=cv.cvtColor(bw, cv.COLOR_BGR2GRAY)

        # ret,bw = cv.threshold(bw,127,255,cv.THRESH_BINARY_INV)
 
        # contours, hera = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2:]
        im2, contours, hera = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        num = self.check_con_num(hera.shape)
        self.get_contours_var(num)



        for i, c in enumerate(contours):
            # Calculate the area of each contour
            area = cv.contourArea(c)
            # Ignore contours that are too small or too large
            if area < 1e2 or 1e5 < area:
                continue
            # Draw each contour only for visualisation purposes
            # cv.drawContours(bw, contours, i, (255, 255, 255), 2)
            # cv.drawContours(cropped_image, contours, i, (0, 0, 255), 2)
            # Find the orientation of each shape
            # self.angle, self.vector_len, center = self.getOrientation(c, cropped_image)
            self.angle, center = self.getOrientation(c, cropped_image)

    def get_center_onSameTree(self, cnt):
        for vectors in self.vector_list:
            if vectors[0]> (0) and vectors[0]< (400): # cutter on right side in the image
                self.vector_list_sameTree.append(vectors)

        

    def get_centers(self):
        for i in range(0, self.iter):
            self.image_process(i)
            # cv.circle(self.img, [int(self.vector_list[i][0]), int(self.vector_list[i][1])], 3, (255, 255, 255), 2)
if __name__ == "__main__":
    for k in range(0,3):
        # pca=PCA()

        name = str(k)
        pca = PCA("{}.png".format(name))
        pca.get_centers()
        flag = 0
        v0 = np.array([pca.vector_list[0][0], pca.vector_list[0][1]])
        for j in range(3,len(pca.vector_list)):
            if flag==0:
                v1=np.array([pca.vector_list[j-2][0],pca.vector_list[j-2][1]])
                v_2=np.array([pca.vector_list[j-1][0],pca.vector_list[j-1][1]])
                v_=np.array([pca.vector_list[j][0],pca.vector_list[j][1]])
                if j ==len(pca.vector_list)-1:
                    v2=v_
                else:
                    v2=(v_2+v_)/2
                quad = Quad(v0, v2, 10, v1)
                v0=v2
                step_size_to_use = int(quad.radius_2d * 1.5)  # Go along the edge at 1.5 * the estimated radius
                perc_width_to_use = 0.3  # How "fat" to make the edge rectangles
                perc_width_to_use_mask = 1.4  # How "fat" a rectangle to cover the mask
                quad.draw_quad(pca.img)
                flag=0
            else:
                flag=0
            # for i in range(1,2):
            #     cv.line(pca.img, (int(v[i-1][0]), int(v[i-1][1])), (int(v[i][0]), int(v[i][1])), (0, 0, 0), 1, cv.LINE_AA)
        pca.save_image("{}_.png".format(name))

    cv.imshow('output', pca.img)
    cv.waitKey(1)
