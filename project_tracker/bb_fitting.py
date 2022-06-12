from multiprocessing.pool import CLOSE
import numpy as np
import logging
import time

logging.basicConfig(level=logging.DEBUG)

class LShapeFitter():

    def __init__(self):
        self.AREA_CRIT = 1
        self.CLOSENESS_CRIT = 2
        self.VAR_CRIT = 3

        self.criteria = self.VAR_CRIT
        self.min_dist_closeness_crit = 0.01 # m
        self.d_theta_deg_search = 3.0 # deg
    
    def fit_rectangle(self, cloud):
        """Takes in a pointcloud for a single cluster and fits a bounding box based on L-shaped fitting. Based on 
        paper linked below.
        https://www.ri.cmu.edu/wp-content/uploads/2017/07/Xiao-2017-Efficient-L-Shape-Fitting.pdf
        c1 and c2 represent edges of the bounding box.

        """
        start = time.time()
        logging.debug(str(cloud.shape))
        cloud_xy = cloud[:,0:2] # only use bird's eye view for L shaped fitting
        logging.debug(f"Min: ({min(float(point[0]) for point in cloud_xy):.2f}, {min(float(point[1]) for point in cloud_xy):.2f}), Max: ({max(float(point[0]) for point in cloud_xy)}, {max(float(point[1]) for point in cloud_xy)})")
        d_theta = np.deg2rad(self.d_theta_deg_search)
        min_cost = -float('inf')
        best_theta = float('inf')

        logging.debug(f"Time to initialise: {time.time() - start}")


        start = time.time()
        for theta in np.arange(0.0, np.pi / 2.4 - d_theta, d_theta):
            # e1 and e2 from paper. rectangle edge direction vectors
            e1 = np.array([np.cos(theta), np.sin(theta)])
            e2 = np.array([-np.sin(theta), np.cos(theta)])
            # project points onto the edge
            c1 = cloud_xy @ e1.T
            c2 = cloud_xy @ e2.T
            # logging.debug(f"c1: {c1[0:5]}, c2: {c2[0:5]}")

            # calculate costs based on criteria
            if self.criteria == self.AREA_CRIT:
                cost = self.calc_area_crit(c1, c2) # CAN BE IMPLEMENTED LATER BUT PAPER SHOWS VAR_CRIT PERFORMS BEST
            elif self.criteria == self.CLOSENESS_CRIT:
                cost = self.calc_closeness_crit(c1, c2) # CAN BE IMPLEMENTED LATER BUT PAPER SHOWS VAR_CRIT PERFORMS BEST
            elif self.criteria == self.VAR_CRIT:
                cost = self.calc_var_crit(c1, c2)

            # update the cost and angle based on best fit
            if min_cost < cost:
                min_cost = cost
                best_theta = theta

        logging.debug(f"Time to go through search: {time.time() - start}")
        logging.debug(f"Best theta found to be {best_theta*180/np.pi}")


        start = time.time()
        # get best rectangle
        sin_best = np.sin(best_theta)
        cos_best = np.cos(best_theta)

        c1_best = cloud_xy @ np.array([cos_best, sin_best]).T
        c2_best = cloud_xy @ np.array([-sin_best, cos_best]).T

        # calculate the 4 corners defining the rectangle
        a = [cos_best, -sin_best, cos_best, -sin_best]
        b = [sin_best, cos_best, sin_best, cos_best]
        c = [min(c1_best), min(c2_best), max(c1_best), max(c2_best)]
        logging.debug(f"a: {a}\nb: {b}\nc: {c}")
        bl_x, bl_y = self.calc_cross_point(a[0:2], b[0:2], c[0:2])
        br_x, br_y = self.calc_cross_point(a[1:3], b[1:3], c[1:3])
        tr_x, tr_y = self.calc_cross_point(a[2:4], b[2:4], c[2:4])
        tl_x, tl_y = self.calc_cross_point([a[3], a[0]], [b[3], b[0]], [c[3], c[0]])
        logging.debug(f"tl: ({tl_x},{tl_y}), tr: ({tr_x},{tr_y})")
        logging.debug(f"bl: ({bl_x},{bl_y}), br: ({br_x},{br_y})")

        logging.debug(f"Time to determine corners: {time.time() - start}")

        # from 4 points calculate the centre and width/length
        start = time.time()
        centre = [(tl_x + tr_x + bl_x + br_x )/4, (tl_y + tr_y + bl_y + br_y)/4]
        length = np.sqrt((bl_x - br_x)**2 + (bl_y - br_y)**2)
        width = np.sqrt((tl_x - bl_x)**2 + (tl_y - bl_y)**2)

        logging.debug(f"centre: {centre}, width: {width}, length: {length}, yaw: {best_theta*180/np.pi:.4f}")

        logging.debug(f"Time to calc stuff for ROS: {time.time() - start}")
        return centre, width, length, best_theta

    def calc_var_crit(self, c1, c2):
        c1_max = max(c1)
        c2_max = max(c2)
        c1_min = min(c1)
        c2_min = min(c2)

        # Vectorization
        D1 = np.minimum(c1_max - c1, c1 - c1_min)
        D2 = np.minimum(c2_max - c2, c2 - c2_min)
        E1 = D1[D1 < D2]
        E2 = D2[D1 >= D2]
        V1 = - np.var(E1) if len(E1) > 0 else 0.
        V2 = - np.var(E2) if len(E2) > 0 else 0.
        gamma = V1 + V2

        return gamma

    def calc_cross_point(self, a, b, c):
        x = (b[0] * -c[1] - b[1] * -c[0]) / (a[0] * b[1] - a[1] * b[0])
        y = (a[1] * -c[0] - a[0] * -c[1]) / (a[0] * b[1] - a[1] * b[0])
        return x, y

