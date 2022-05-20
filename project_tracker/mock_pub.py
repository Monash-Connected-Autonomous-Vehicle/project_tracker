#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import os
import numpy as np
import glob 
import time
from datetime import date

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2 as PCL2
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2

from project_tracker.utils import numpy_2_PCL2

from geometry_msgs.msg import TwistStamped
from builtin_interfaces.msg import Time

class PointCloudToPCL2(Node):

    def __init__(self):
        super(PointCloudToPCL2, self).__init__('point_cloud_to_pcl2')
        self._publisher = self.create_publisher(PCL2, '/velodyne_points', 10)
        self._oxts_publisher = self.create_publisher(TwistStamped, '/oxts_twist', 10)
        
        # kitti data directory setup
        # velodyne_points and oxts should be directories within kitta_data_dir
        self.declare_parameter('kitti_data_dir', "/home/home/DATASETS/KITTI/2011_09_26/2011_09_26_drive_0048_sync/")
        kitti_data_dir = self.get_parameter('kitti_data_dir').get_parameter_value().string_value
        # velodyne data directory setup
        self.velodyne_glob_path = os.path.join(kitti_data_dir, 'velodyne_points', 'data', '*.bin')
        self.velodyne_file_paths = sorted(glob.glob(self.velodyne_glob_path))
        with open(os.path.join(kitti_data_dir, 'velodyne_points', 'timestamps.txt')) as f:
            lines = f.readlines()
            # pull only the seconds as we are only worried about relative time
            self.velodyne_timestamps = [float(line.rstrip('\n').split()[1].split(':')[2]) for line in lines]
        # oxts data directory setup
        self.oxts_glob_path = os.path.join(kitti_data_dir, 'oxts', 'data',  '*.txt')
        self.oxts_file_paths = sorted(glob.glob(self.oxts_glob_path))
        with open(os.path.join(kitti_data_dir, 'oxts', 'timestamps.txt')) as f:
            lines = f.readlines()
            # pull only the seconds as we are only worried about relative time
            self.oxts_timestamps = [float(line.rstrip('\n').split()[1].split(':')[2]) for line in lines]

        # loop publishing PCL2 and oxts Twist data
        self.counter = 0
        self.start_epoch = time.time()
        while True:
            if self.counter >= len(self.velodyne_file_paths):
                self.get_logger().info("Restarting from beginning!")
                self.counter = 0
            self.get_logger().info(f"Publishing point {self.counter}")
            self.publish_pcl2()
            self.publish_oxts()
            self.counter += 1
            time.sleep(0.5) # don't overcook the CPU 

    def publish_pcl2(self):
        """Callback to publish PCL2 data"""
        msg = self.convert_bin_to_PCL2(self.velodyne_file_paths[self.counter], self.velodyne_timestamps[self.counter])
        self._publisher.publish(msg)            

    def convert_bin_to_PCL2(self, velodyne_file_path, timestamp):
        """Method to convert Lidar data in binary format to PCL2 message"""
        
        cloud = np.fromfile(velodyne_file_path, np.float32)
        cloud = cloud.reshape((-1, 4))

        # create Time from timestamp provided in velodyne kitti data
        stamp = Time()
        epoch_time = self.start_epoch + timestamp
        stamp.sec = int(epoch_time)
        stamp.nanosec = int((epoch_time - stamp.sec)*10**9)
        pcl2_msg = numpy_2_PCL2(cloud, stamp)

        return pcl2_msg

    def publish_oxts(self):
        """Callback to publish oxts data as a Twist message for use in tracker"""
        twist_stamped = TwistStamped()

        twist_stamped.header = Header()
        twist_stamped.header.frame_id = 'velodyne'
        # create Time from timestamp provided in velodyne kitti data
        stamp = Time()
        epoch_time = self.start_epoch + self.oxts_timestamps[self.counter]
        stamp.sec = int(epoch_time)
        stamp.nanosec = int((epoch_time - stamp.sec)*10**9)
        twist_stamped.header.stamp = stamp

        with open(self.oxts_file_paths[self.counter]) as f:
            oxts_data = [float(num) for num in f.read().split()]
            twist_stamped.twist.linear.x = oxts_data[8]
            twist_stamped.twist.linear.y = oxts_data[9]
            twist_stamped.twist.linear.z = oxts_data[10]
        
        self._oxts_publisher.publish(twist_stamped)



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
