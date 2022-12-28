import os.path
from ast import Pass
import numpy as np
import cv2 as cv
import time
import fit_bezier as fit_bezier
from abc import ABC, abstractmethod
from PIL import Image
from cyl_fit_2d import Quad

#to do : more than one tree visible


class ImageProcessor(ABC):
    # input is the masked single leader only image
    #output is desired end effector pose in pixel coordinate

    @abstractmethod
    def process_image(self, img):
        pass

    @abstractmethod
    def mask_to_curve(self, mask):
        pass
        
    # imput is raw rgb image   
    #output binary  - leader only image
    @abstractmethod
    def image_to_mask(self, img):
        pass

    @property
    @abstractmethod
    def mask(self):
        pass

    @property
    @abstractmethod
    def curve(self):
        pass

    @property
    @abstractmethod
    def center(self):
        pass

    @abstractmethod
    def get_center_distance(self, normalize=False):
        pass
        

class FlowGANImageProcessor(ImageProcessor):
    def __init__(self):
        self.img_size = (424,240)
        from flowgan import FlowGAN
        self.flowgan = FlowGAN(self.img_size, self.img_size, use_flow=True, gan_name='synthetic_flow_pix2pix',
                               gan_input_channels=6, gan_output_channels=1)



        self._last_img = None
        self._last_mask = None
        self._last_curve = None
        self._last_center = None
        self._last_contours = None
        self._last_contour_centers = None
        self.img_divisions = 3

    @property
    def mask(self):
        return self._last_mask

    @property
    def center(self):
        return self._last_center

    @property
    def curve(self):
        return self._last_curve

    def show_image(self, image, title):
        cv.imshow(title, image)
        cv.waitKey(1)

    def image_to_mask(self, img, img_pre):
        img_0 = np.asanyarray(img).astype(np.uint8)[:,:,:3]
        img_1 = np.asanyarray(img_pre).astype(np.uint8)[:,:,:3]
        self.flowgan.process(img_0)
        mask = self.flowgan.process(img_1)
        flow = self.flowgan.last_flow
        # self.show_image(mask, 'Flow mask')
        return mask, img_0, img_1, flow
    

    def get_image_dict(self, mask, rgb0, rgb1, flow):
        """ Read in all of the mask, rgb, flow images
        If Edge image does not exist, create it
        @param image_name: image number/name as a string
        @returns dictionary of images with image_type as keywords """

        images = {}
        images["Mask"] = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        images["RGB0"] = rgb0
        images["RGB1"] = rgb1
        images["Flow"] = flow
        images["Edge"] = None
      

        im_gray = cv.cvtColor(images["RGB0"], cv.COLOR_BGR2GRAY)
        images["Edge"] = cv.Canny(im_gray, 50, 150, apertureSize=3)
        return images
    
    def process_image(self, img, pre_img):
        mask, rgb0, rgb1, flow = self.image_to_mask(img, pre_img)
        images = self.get_image_dict(mask, rgb0, rgb1, flow)
        curve, m_pt = self.mask_to_curve(images)
        # if curve is not None:
        branch_center = m_pt
        #     pass
        # else:
        #     branch_center = None
        #     pass

        
        self._last_center = branch_center
        self._last_curve = curve
        self._last_img = img
        self._last_mask = mask

    def mask_to_curve(self, images):
        from  LeaderDetector import LeaderDetector
        leader_detect = LeaderDetector(images, b_output_debug=True, b_recalc=True)
        curve = Quad([0, 0], [1,1], 1)
        curve_mpt = leader_detect.bezier_mpt
        return curve, curve_mpt

    def visualize(self):

        if self._last_img is None:
            return

        img = self._last_img.copy()
        h, w = img.shape[:2]
        img_bezier = img
        cv.line(img_bezier, (w // 2, 0), (w // 2, h), (0, 0, 0), 2)
        cv.circle(img_bezier, (int(self._last_center[0]),int(self._last_center[1])), 2, (0, 0, 255), -1)
        self.show_image(img_bezier, 'Camera input: Bezier curve mpt')


    def show_image(self, image, title):
        cv.imshow(title, image)
        cv.waitKey(1)

    def get_center_distance(self, normalize=False):

        if self._last_center is None:
            return None

        dist = self._last_center[0] - self.mask.shape[1] / 2
        if normalize:
            # Normalizes distance to a number between -1 and 1
            dist = dist / (self.mask.shape[1] / 2)

        return dist

    def find_branch_center_pixel(self, mask):
        contour_center, contours = self.get_contour_centers(mask)
        self._last_contours = (contour_center, contours)
        pca_center = self.get_pca_of_contours(contours)
        if not contours or pca_center is None:
            return None

        return pca_center[0]

    def get_contour_centers(self, mask, y_offset=0):
        contours, hera = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        contour_center = [0,0]
        for c in contours:
            M = cv.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX = 0
                cY = 0
            
            contour_center = [cX, cY + y_offset]
        cv.drawContours(mask, contours, -1, (0, 255, 0), 3)
        self.show_image(mask, ' mask')
        return contour_center, contours



    def get_pca_of_contours(self, contours):

        for i, contour in enumerate(contours):

            area = cv.contourArea(contour)
            if area < 1e2 or 1e5 < area:
                continue
            sz = len(contour)
            data_pts = np.empty((sz, 2), dtype=np.float64)
            for i in range(data_pts.shape[0]):
                data_pts[i, 0] = contour[i, 0, 0]
                data_pts[i, 1] = contour[i, 0, 1]
            mean = np.empty((0))
            mean, eigenvectors = cv.PCACompute(data_pts, mean)
            center = (int(mean[0, 0])+0, int(mean[0, 1]))
            return center


class HSVBasedImageProcessor(ImageProcessor):
    def __init__(self):
        self.lower_red = np.array([60, 40, 40])
        self.upper_red = np.array([140, 255, 255])
        self.img_divisions = 3
        self._last_img = None
        self._last_mask = None
        self._last_curve = None
        self._last_center = None
        self._last_contours = None
        self._last_contour_centers = None

    @property
    def mask(self):
        return self._last_mask

    @property
    def center(self):
        return self._last_center

    @property
    def curve(self):
        return self._last_curve

    def process_image(self, img, img_):
        mask = self.image_to_mask(img)
        curve = self.mask_to_curve(mask)
        if curve is not None:
            branch_center = self.find_branch_center_pixel(mask)
        else:
            branch_center = None

        self._last_center = branch_center
        self._last_curve = curve
        self._last_img = img
        self._last_mask = mask


    def image_to_mask(self, image):
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.lower_red, self.upper_red)
        return mask

    # divide the image to 3 parts (fit a curve using 3 points)
    def divide_mask(self, step, mask):

        height = mask.shape[0]
        crop_start = int(height * (step / self.img_divisions))
        crop_end = int(height * ((step + 1) / self.img_divisions))
        return mask[crop_start:crop_end]


    def get_contour_centers(self, mask, y_offset=0):
        contours, hera = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        contour_center = [0,0]
        for c in contours:
            M = cv.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX = 0
                cY = 0
            
            contour_center = [cX, cY + y_offset]
        return contour_center, contours

    def get_pca_of_contours(self, contours):

        for i, contour in enumerate(contours):

            area = cv.contourArea(contour)
            if area < 1e2 or 1e5 < area:
                continue
            sz = len(contour)
            data_pts = np.empty((sz, 2), dtype=np.float64)
            for i in range(data_pts.shape[0]):
                data_pts[i, 0] = contour[i, 0, 0]
                data_pts[i, 1] = contour[i, 0, 1]
            mean = np.empty((0))
            mean, eigenvectors = cv.PCACompute(data_pts, mean)
            center = (int(mean[0, 0])+0, int(mean[0, 1]))
            return center

    def mask_to_curve(self, mask):
        contour_centers = []

        for idx in range(self.img_divisions):
            submask = self.divide_mask(idx, mask)
            offset = (idx * mask.shape[0]) // self.img_divisions
            contour_center, _ = self.get_contour_centers(submask, y_offset=offset)
            if contour_center is None:
                self._last_contour_centers = None
                return None
            else:
                contour_centers.append(contour_center)

        curve = fit_bezier.Quad(contour_centers[0], contour_centers[2], 10, contour_centers[1])
        self._last_contour_centers = contour_centers
        return curve

    def visualize(self):

        if self._last_img is None:
            return

        img = self._last_img.copy()
        h, w = img.shape[:2]

        contour_centers = self._last_contour_centers
        if contour_centers is not None:
            img_bezier, m_pt, traj = self._last_curve.draw_quad(img, contour_centers[0], contour_centers[2])
            cv.circle(img_bezier, (int(m_pt[0]), int(m_pt[1])), 2, (0, 0, 255), -1)
            cv.line(img_bezier, (w // 2, 0), (w // 2, h), (0, 0, 0), 2)
            for idx in range(self.img_divisions):
                cv.circle(img_bezier, (contour_centers[idx][0], contour_centers[idx][1]), 2, (255, 255, 255), -1)
        else:
            img_bezier = img
        self.show_image(img_bezier, 'Camera input: Bezier curve')


    def show_image(self, image, title):
        cv.imshow(title, image)
        cv.waitKey(1)

    def get_center_distance(self, normalize=False):

        if self._last_center is None:
            return None

        dist = self._last_center - self.mask.shape[1] / 2
        if normalize:
            # Normalizes distance to a number between -1 and 1
            dist = dist / (self.mask.shape[1] / 2)

        return dist

    def find_branch_center_pixel(self, mask):
        contour_center, contours = self.get_contour_centers(mask)
        self._last_contours = (contour_center, contours)
        pca_center = self.get_pca_of_contours(contours)
        if not contours or pca_center is None:
            return None

        return pca_center[0]


if __name__ == '__main__':

    from PIL import Image
    test_img_loc = os.path.join(os.path.expanduser('~'), 'test')
    files = ['test_0.png', 'test_1.png']
    imgs = [np.asanyarray(Image.open(os.path.join(test_img_loc, file))).astype(np.uint8)[:,:,:3] for file in files]
    img_size = (imgs[0].shape[1], imgs[0].shape[0])
    img_proc = FlowGANImageProcessor(img_size)

    img_proc.image_to_mask(imgs[0])
    mask = img_proc.image_to_mask(imgs[1])
    Image.fromarray(mask).save(os.path.join(test_img_loc, 'mask.png'))