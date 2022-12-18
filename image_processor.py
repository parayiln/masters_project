

from ast import Pass
import numpy as np
import cv2 as cv
import time
import fit_bezier as fit_bezier
from abc import ABC, abstractmethod

#to do : more than one tree visible


class ImageProcessor(ABC):
    @abstractmethod
    def mask_to_curve(self, mask):
        pass
        #output is desired end effector pose in pixel coordinate

    @abstractmethod
    def image_to_mask(self, img):
        pass
        #output binary  - leader only image

    @abstractmethod
    def mask_to_centered(self, mask):
        pass
        #updates flags (is centered or not)


class HSVBasedImageProcessor(ImageProcessor):
    def __init__(self):
        self.lower_red = np.array([60, 40, 40])
        self.upper_red = np.array([140, 255, 255])
        self.iter = 3
        self.pixel = 100
        self.cropped_height = 0
        self.curr_target = 0
        self.leader_centered = False
        self.follow_leader = False
        self.first_scan = True
        self.leader_on_right = False
        self.move_curr_branch = False
        self.no_of_branch_scaned = 0
        self.thres_out_of_sight =220 

    def image_to_mask(self, image):
        hsv =cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower_red, self.upper_red)
        self.img_w = image.shape[1]
        self.img_h = image.shape[0]
        return mask

    def divide_mask(self,step, mask, img):
        self.crop_h = int(self.img_h/self.iter)
        self.cropped_height = 0 + self.crop_h * step
        cropped_image = img[self.cropped_height:self.cropped_height +
                                 self.pixel, 0:img.shape[1]]
        bw = mask[self.cropped_height:self.cropped_height +
                             self.pixel, 0:img.shape[1]]
        return cropped_image, bw

    def get_contour_centers(self, bw, img):
        contours, hera = cv.findContours(
            bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour_center = [0,0]
        for c in contours:
            M = cv.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX = 0
                cY = 0
            
            contour_center = [cX,cY+self.cropped_height]
            
        return contour_center, contours

    def get_pca_of_contours(self, contours, img):
        for i, c in enumerate(contours):
            area = cv.contourArea(c)
            if area < 1e2 or 1e5 < area:
                continue
            sz = len(c)
            data_pts = np.empty((sz, 2), dtype=np.float64)
            for i in range(data_pts.shape[0]):
                data_pts[i, 0] = c[i, 0, 0]
                data_pts[i, 1] = c[i, 0, 1]
            mean = np.empty((0))
            mean, eigenvectors = cv.PCACompute(data_pts, mean)
            center = (int(mean[0, 0])+0, int(mean[0, 1])+self.cropped_height)
            return center


    def mask_to_curve(self,mask,img):
        contour_centers=[]
        for i in range(self.iter):
            img_crop, bw = self.divide_mask(i,mask, img)
            contour_center, _ = self.get_contour_centers(bw, img_crop)
            contour_centers.append(contour_center)
        self.quad = fit_bezier.Quad(contour_centers[0], contour_centers[2], 10, contour_centers[1])
        img_bezier, m_pt, traj = self.quad.draw_quad(img, contour_centers[0], contour_centers[2])
        cv.circle(img_bezier, (int(m_pt[0]), int(m_pt[1])), 2, (0, 0, 255), -1)
        cv.line(img_bezier, (int(self.img_w/2), 0), (int(self.img_w/2), self.img_h), (0, 0, 0), 2)
        for i in range(self.iter):
            cv.circle(img_bezier, (contour_centers[i][0], contour_centers[i][1]), 2, (255, 255, 255), -1)
        cv.imshow('Camera input: Bezier curve', img_bezier)
        cv.waitKey(1)
        return m_pt
    
    def check_leader_on_right(self, y, center_y):
        if y - center_y > 0:
            self.leader_on_right = False
        else:
            self.leader_on_right = True
    
    
    def mask_to_centered(self, mask, img):
        self.cropped_height =0
        contour_center, contours = self.get_contour_centers(mask, img)
        pca_center = self.get_pca_of_contours(contours, img)
        self.check_leader_on_right(contour_center[0], int(self.img_w/2))
        cnt_y = contour_center[0]
        if self.move_curr_branch == True:
            # print("moving scanned branch", cnt_y)
            if cnt_y - int(self.thres_out_of_sight) < 0:
                self.move_curr_branch = False
                print("------moved the scanned leader out of sight-------")
        else:
            if self.leader_on_right == True:
                if cnt_y - int(self.img_w/2) ==  0:
                    self.leader_centered = True
                    print("------centered! starting to follow the leader -------")
                    self.follow_leader = True
                else:
                    self.leader_centered = False
        cv.circle(img, (contour_center[0], contour_center[1]), 2, (255, 255, 255), -1)
        if pca_center is not None:
            cv.circle(img, (pca_center[0], pca_center[1]), 2, (255, 0, 255), -1)
        cv.line(img, (int(self.img_w/2), 0), (int(self.img_w/2), self.img_h), (255, 0, 0), 2)
        cv.imshow('Camera input: centering', img)
        cv.waitKey(1)    
####################################################################
