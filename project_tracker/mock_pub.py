#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import os
import numpy as np
import glob 
import time

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2 as PCL2
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2

from project_tracker.utils import numpy_2_PCL2

class PointCloudToPCL2(Node):

    def __init__(self):
        super(PointCloudToPCL2, self).__init__('point_cloud_to_pcl2')
        self._publisher = self.create_publisher(PCL2, '/velodyne_points', 10)

        # timer_period = 0.5
        # self.timer = self.create_timer(timer_period, self.publish_pcl2)
        
        self.declare_parameter('velodyne_data_dir', "/home/mcav/DATASETS/KITTI/2011_09_26/2011_09_26_drive_0048_sync/velodyne_points/data/")
        velodyne_data_dir = self.get_parameter('velodyne_data_dir').get_parameter_value().string_value
        self.velodyne_glob_path = os.path.join(velodyne_data_dir, '*.bin')
        self.velodyne_file_paths = None

        while True:
            self.publish_pcl2()
            time.sleep(0.05) # don't overcook the CPU 

    def publish_pcl2(self):
        """Callback to publish"""
        
        if self.velodyne_file_paths:
            msg = self.convert_bin_to_PCL2(self.velodyne_file_paths.pop())

            self._publisher.publish(msg)
        else: # restart
            self.velodyne_file_paths = sorted(glob.glob(self.velodyne_glob_path))
            self.get_logger().info("Restarting from beginning!")

    def convert_bin_to_PCL2(self, velodyne_file_path):
        """Method to convert Lidar data in binary format to PCL2 message"""
        
        cloud = np.fromfile(velodyne_file_path, np.float32)
        cloud = cloud.reshape((-1, 4))

        timestamp =  self.get_clock().now().to_msg()
        pcl2_msg = numpy_2_PCL2(cloud, timestamp)

        self.get_logger().info(f"Publishing, first point: {cloud[0:5]}")

        return pcl2_msg


def main(args=None):
    rclpy.init(args=args)

    point_cloud_to_pcl2 = PointCloudToPCL2()

    try:
        rclpy.spin(point_cloud_to_pcl2)
    except KeyboardInterrupt:
        point_cloud_to_pcl2.get_logger().debug("Keyboard interrupt")

    # destroy node explicity
    point_cloud_to_pcl2.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
