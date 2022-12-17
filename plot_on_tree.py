import numpy as np
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt


class Plot_on_tree():
    def __init__(self):
        # self.tree_location_urdf = [1.2, 0.5]
        # self.tree_location_urdf_last = [0.35, 0.5]
        self.tree_location_urdf = [1.25, 0.5]
        self.tree_location_urdf_last = [0.35, 0.5]
        self.scale= [500,500*4/3]
        self.img_tree_base=[445,-200]
        self.cvimg = cv.imread('/home/nidhi/masters_project/tree02d.png')
        self.img_dim = self.cvimg.shape
        self.image= plt.imread("/home/nidhi/masters_project/tree02d.png")
        self.fig, self.ax = plt.subplots()
        self.img= self.ax.imshow(self.image, extent=[0, self.img_dim[1], 0, self.img_dim[0]])
        # self.plot_traj()
        # self.df = pd.read_csv("traj_hsv.csv")
        self.df_last = pd.read_csv("results/traj_hsv.csv")
        # self.trajs = self.df.to_numpy()
        self.trajs_last = self.df_last.to_numpy()
        # self.ef_x, self.ef_y = self.scale_traj(self.trajs[2], self.trajs[3], self.tree_location_urdf) 
        self.ef_x_last, self.ef_y_last = self.scale_traj(self.trajs_last[2], self.trajs_last[3], self.tree_location_urdf_last) 
        # self.plot_traj(self.ef_x,self.ef_y,'red')
        self.plot_traj(self.ef_x_last,self.ef_y_last,'red')


    def scale_traj(self, xdata, ydata, urdf_offset):
        xval = (xdata + urdf_offset[0]) * self.scale[0]+self.img_tree_base[0]
        yval = (ydata + urdf_offset[1]) * self.scale[1]+self.img_tree_base[1]
        return list(xval), list(yval)

    def plot_traj(self, x, y, color_):
        del x[0]
        del y[0]
        self.ax.plot(x, y, linewidth=3, color=color_)

            


if __name__ == "__main__":

    tree_plot = Plot_on_tree()
    # tree_plot.ax.plot([0,tree_plot.img_tree_base[0]],[0, tree_plot.img_tree_base[1]],ls='dotted', linewidth=5, color='red')
    plt.show()