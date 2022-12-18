from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse
from math import atan2, cos, sin, sqrt, pi
from matplotlib import pyplot as plt
import os
import time

### input 3 points
### output a curve

class Quad:
    def __init__(self, v1, v2, radius, mid_pt):
        self.bezier = np.zeros((2, 3))
        self.p0, self.p2, self.orientation = self._orientation(
            (np.array(v1)), np.array(v2))
        if mid_pt is None:
            self.p1 = 0.5 * (self.p0 + self.p2)
        else:
            self.p1 = np.array(mid_pt)
        self.depth_values = []
        self.radius_2d = radius
        self.current_time = []
####################################################################
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
####################################################################
    def pt_axis(self, t):
        """ Return a point along the bezier
        @param t in 0, 1
        @return 2d point"""
        pts = np.array([self.p0[i] * (1-t) ** 2 + 2 * (1-t) * t *
                       self.p1[i] + t ** 2 * self.p2[i] for i in range(0, 2)])
        return pts.transpose()

####################################################################
    def diff_pt_axis(self, t):
        """ Return a point along the bezier
        @param t in 0, 1
        @return 2d point"""
        dy = np.array([self.p0[i] * (t-1) * 2 + 2 * self.p1[i] - 4 *
                      t * self.p1[i] + t * 2 * self.p2[i]for i in range(0, 2)])
        pts = dy
        return pts.transpose()
        # return self.p0 * (1-t) ** 2 + 2 * (1-t) * t * self.p1 + t ** 2 * self.p2
####################################################################
    def tangent_slope(self, m_pt):
        # compute dy/dx
        t = m_pt
        dy_dt = self.p0[1] * (t[1]-1) * 2 + 2 * self.p1[1] - \
            4 * t[1] * self.p1[1] + t[1] * 2 * self.p2[1]
        dx_dt = self.p0[0] * (t[0]-1) * 2 + 2 * self.p1[0] - \
            4 * t[0] * self.p1[0] + t[0] * 2 * self.p2[0]
        slope = -dx_dt/dy_dt
        return slope
####################################################################
    def tangent_curve_points(self, m_pt, scale=1):
        slope = self.tangent_slope(m_pt)
        x_a = int(m_pt[0]+scale)
        x_b = int(m_pt[0]-scale)
        y_a = m_pt[1]+int(slope*(x_a-m_pt[0]))
        y_b = m_pt[1]+int(slope*(x_b-m_pt[0]))
        return [x_a, y_a], [x_b, y_b]
####################################################################
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
####################################################################

    def draw_quad(self, im, a, b):
        """ Set the pixels corresponding to the quad to white
        @im numpy array"""
        n_pts_quad = 6
        image = im.copy()
        self.traj = []
        self.vel = []
        pts = self.pt_axis(np.linspace(0, 1, n_pts_quad))
        vel_pts = self.diff_pt_axis(np.linspace(0, 1, n_pts_quad))
        col_div = 120 // (n_pts_quad - 1)
        for p1, p2 in zip(pts[0:-1], pts[1:]):
            cv.line(image, (int(p1[0]), int(p1[1])), (int(
                p2[0]), int(p2[1])), (0, 255, 0), thickness=1)
        for vel_p1 in vel_pts:
            self.vel.append([vel_p1[0], vel_p1[1]])
        for p1 in pts:
            self.traj.append([p1[0], p1[1]])
        # cv.line(image, (a[0], a[1]), (b[0], b[1]), (0, 0, 0), thickness=1)
        curve_mid_pt = pts[int(len(pts[0:-1])/2)]
        return image, curve_mid_pt, self.traj

 ####################################################################       

    def drawAxis(self, img,  p, q, colour, scale, w, h):
        angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
        hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) +
                          (p[0] - q[0]) * (p[0] - q[0]))
        q[0] = p[0] - scale * hypotenuse * cos(angle)
        q[1] = p[1] - scale * hypotenuse * sin(angle)
        cv.line(img, (int(p[0]), int(p[1])),
                (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
        p[0] = q[0] + 9 * cos(angle + pi / 4)
        p[1] = q[1] + 9 * sin(angle + pi / 4)
        cv.line(img, (int(p[0]), int(p[1])),
                (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
        p[0] = q[0] + 9 * cos(angle - pi / 4)
        p[1] = q[1] + 9 * sin(angle - pi / 4)
        cv.line(img, (int(p[0]), int(p[1])),
                (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
        cv.line(img, (int(w/2), 0), (int(w/2), h), (0, 0, 0), 2)
        # cv.imshow('Camer input: Bezier curve', img)
        # cv.waitKey(1)

        return img

###########---------- PCA ------------###############################
