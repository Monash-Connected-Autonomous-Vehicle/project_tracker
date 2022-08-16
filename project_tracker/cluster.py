#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import logging
import time

import numpy as np

from sensor_msgs.msg import PointCloud2 as PCL2
from geometry_msgs.msg import TwistStamped
from visualization_msgs.msg import MarkerArray, Marker

from utils import numpy_2_PCL2, PCL2_2_numpy, create_colour_list, pcl_to_ros
from tracking import Tracker

from mcav_interfaces.msg import DetectedObject, DetectedObjectArray

from transforms3d.euler import euler2quat
import pcl

import config

class PCL2Subscriber(Node):
    def __init__(self):
        super(PCL2Subscriber, self).__init__('pcl2_subscriber')
        self.pcl2_subscription = self.create_subscription(
            PCL2,
            '/velodyne_filtered',
            self._tracker_callback,
            10
        )

        self.twist_subscription = self.create_subscription(
            TwistStamped,
            '/oxts_twist',
            self._oxts_callback,
            10
        )

        self.pointcloud: pcl.PointCloud
        self.np_pointcloud: np.ndarray
        self.original_frame_id: str

        # TODO create cloud cluster publisher via creating a custom msg
        self._cloud_cluster_publisher = self.create_publisher(PCL2, 'clustered_pointclouds', 10)
        self._bounding_boxes_publisher = self.create_publisher(MarkerArray, 'bounding_boxes', 10)
        self._detected_objects_publisher = self.create_publisher(DetectedObjectArray, 'detected_objects', 10)
        

        # parameters for Euclidean Clustering
        self.declare_parameter('min_cluster_size', 10)
        self.min_cluster_size = self.get_parameter('min_cluster_size').get_parameter_value().integer_value
        self.declare_parameter('max_cluster_size', 20000)
        self.max_cluster_size = self.get_parameter('max_cluster_size').get_parameter_value().integer_value
        self.declare_parameter('cluster_tolerance', 0.6) # range from 0.5 -> 0.7 seems suitable. Test more when have more data
        self.cluster_tolerance = self.get_parameter('cluster_tolerance').get_parameter_value().double_value

        # parameters for merging clusters based on side to side distance from car
        self.merge_y_thresh = 8. # [m] either side of car

        # create tracker for identifying and following objects over time
        # self.tracker = Tracker(max_frames_before_forget=2, max_frames_length=30, tracking_method="centre_distance", dist_threshold=5)
        # self.tracker = Tracker(max_frames_before_forget=2, max_frames_length=30, tracking_method="iou", iou_threshold=0.85)
        self.tracker = Tracker(
            max_frames_before_forget=2, 
            max_frames_length=30, 
            tracking_method="both", 
            iou_threshold=0.85,
            dist_threshold = 5
        )

        # colour list for publishing clusters in different colours
        self.rgb_list, self.colour_list = create_colour_list()

        # set logging level
        self.get_logger().set_level(logging.DEBUG)

    def _oxts_callback(self, msg):
        """IMU subscriber callback. 
        Receives TwistStamped message from mock publisher to use in velocity calculations.

        http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/TwistStamped.html

        Args:
            msg (geometry_msgs.msg.TwistStamped): IMU sensor data from KITTI
        """
        self.tracker.twists_stamped.append(msg)
        self.get_logger().debug(f"TwistStamped message received! {msg}")

    def _tracker_callback(self, msg):
        """LiDAR subscriber callback. 
        Reads in the PCL2 message and converts it to numpy array for easier manipulation.

        Converts numpy array to pcl.PointCloud message to fit clustering using euclidean 
        clustering. Merges clusters based on side to side distance from vehicle, this ensures
        that things like walls are marged together rather than identified as multiple clusters.
        
        Colours the clusters and publishes them to `/clustered_pointclouds`. 

        Fits bounding boxes to the clusters by creating a `DetectedObjectArray`. Track objects
        by assigning IDs to the `DetectedObjectArray` using `Tracker` class. Publish the 
        bounding boxes and `DetectedObjectArray`. Creates bounding box markers for RViz.


        http://docs.ros.org/en/lunar/api/sensor_msgs/html/msg/PointCloud2.html

        Args:
            msg (sensor_msgs.msg.PointCloud2): pointcloud received from LiDAR
        """
        ## read in PCL2 message and convert to numpy array
        start = time.time()
        self.original_frame_id = msg.header.frame_id
        self.full_cloud = PCL2_2_numpy(msg, reflectance=False)
        self.cloud = self.full_cloud[:, :3] # ignore reflectance for clustering
        
        self.no_samples, self.no_axes = self.cloud.shape
        cloud_time = time.time() - start
        self.get_logger().debug(f"Took {cloud_time:.5f}s to receive pcl2 message: {self.no_samples, self.no_axes}")

        ## python3-pcl binding euclidean clustering
        ec_cloud = pcl.PointCloud() # create empty pcl.PointCloud to use C++ bindings to PCL 
        ec_cloud.from_array(self.cloud)
        ec_tree = ec_cloud.make_kdtree() # make kdtree from our tree
        start = time.time()
        self.euclidean_clustering_ec(ec_cloud, ec_tree)
        clustering_ec_time = time.time() - start
        self.get_logger().debug(f"PCL binding ec took {clustering_ec_time:.5f}s to find {len(self.clusters_ec)} clusters.")

        # merge clusters based on side to side distance from vehicle
        merged_clusters = self.merge_y_clusters()

        ## PUBLISHING
        # COLOURED VERSION
        # colouring the clouds
        self.colour_cluster_point_list = []
        for j, indices in enumerate(merged_clusters):
            for indice in indices:
                self.colour_cluster_point_list.append([
                    self.cloud[indice][0],
                    self.cloud[indice][1],
                    self.cloud[indice][2],
                    self.colour_list[j]
                ])
    # def _callback(self, msg):
    #     """Subscriber callback. Receives PCL2 message and converts it to points"""

    #     # create and save a pythonpcl pointcloud and its numpy representation from the ros message
    #     self.np_pointcloud, self.pointcloud = self.load_pointcloud_from_ros_msg(msg)

    #     # make kdtree from the pointcloud
    #     kd_tree = self.pointcloud.make_kdtree()

    #     # a list of indices representing each set of points in np_pointcloud that are in the same cluster
    #     np_pointcloud_cluster_indices = self.create_euclidean_cluster(
    #         self.pointcloud, kd_tree, config.cluster_tolerance, config.min_cluster_size, config.max_cluster_size)

    #     # a list representing a coloured version of the clusters in the pointcloud for visualisation
    #     coloured_clustered_points = self.create_coloured_pointcloud_clusters(np_pointcloud_cluster_indices)

    #     # convert to pcl.PointCloud_PointXYZRGB for visualisation in RViz
    #     coloured_clustered_pointcloud = pcl.PointCloud_PointXYZRGB()
    #     coloured_clustered_pointcloud.from_list(coloured_clustered_points)
    #     timestamp = self.get_clock().now().to_msg()

        # # convert the pcl to a ROS PCL2 message
        # pcl2_msg = pcl_to_ros(coloured_clustered_pointcloud,
        #                       timestamp, self.original_frame_id)

        # self._cloud_cluster_publisher.publish(pcl2_msg)

        # # fit bounding boxes to the clustered pointclouds
        # detected_objects = self.create_detected_objects(np_pointcloud_cluster_indices) 

        # track objects over time
        # start = time.time()
        # tracked_detected_objects = self.tracker.update(detected_objects, timestamp=msg.header.stamp)
        # self.get_logger().debug(f"Number of tracked objects: {len(tracked_detected_objects.detected_objects)}")

        # # create bounding boxes and ID labels for ROS visualisation
        # self.create_ros_markers(tracked_detected_objects)

        # tracking_time = time.time() - start
        # self.get_logger().debug(f"Tracking took {tracking_time:.5f}s")

        # # publish cluster, bounding boxes and detected objects
        # self._cloud_cluster_publisher.publish(pcl2_msg)
        # self._bounding_boxes_publisher.publish(self.markers)
        # self._detected_objects_publisher.publish(tracked_detected_objects)


    def create_coloured_pointcloud_clusters(self, np_pointcloud_cluster_indices):
        # colouring the clouds
        coloured_clustered_points = []
        for j, indices in enumerate(np_pointcloud_cluster_indices):
            for idx in indices:
                coloured_clustered_points.append([
                    self.np_pointcloud[idx][0],
                    self.np_pointcloud[idx][1],
                    self.np_pointcloud[idx][2],
                    self.colour_list[j]
                ])
        return coloured_clustered_points


    def load_pointcloud_from_ros_msg(self, msg):
        self.original_frame_id = msg.header.frame_id
        np_full_pointcloud = PCL2_2_numpy(msg, reflectance=False)
        np_pointcloud = np_full_pointcloud[:, :3] # ignore reflectance for clustering

        ## python3-pcl binding euclidean clustering
        pointcloud = pcl.PointCloud() # create empty pcl.PointCloud to use C++ bindings to PCL 
        pointcloud.from_array(np_pointcloud)
        
        return np_pointcloud, pointcloud


    def create_euclidean_cluster(self, pointcloud, kd_tree, cluster_tolerance, min_cluster_size, max_cluster_size):
        """
        Perform euclidean clustering with a given pcl.PointCloud() and kdtree

        Parameters
        ----------
        pointcloud : pcl.PointCloud()
            pcl version of pointcloud message received
        kd_tree : kdtree
            kdtree from pcl-python binding
        cluster_tolerance: float 
        min_cluster_size: int
            minimum size of a cluster
        max_cluster_size: int
            maximum size of a cluster
        """
        # make euclidean cluster extraction method
        ec = pointcloud.make_EuclideanClusterExtraction()
        # set parameters
        # ec.set_ClusterTolerance(self.cluster_tolerance)
        # ec.set_MinClusterSize(self.min_cluster_size)
        # ec.set_MaxClusterSize(self.max_cluster_size)
        # ec.set_SearchMethod(ec_tree)
        ec.set_ClusterTolerance(cluster_tolerance)
        ec.set_MinClusterSize(min_cluster_size + 1)
        ec.set_MaxClusterSize(max_cluster_size)
        ec.set_SearchMethod(kd_tree)

        # perform euclidean clustering and return indices
        return ec.Extract()


    def create_detected_objects(self, np_pointcloud_cluster_indices):
        """
        Create detected objects from the clusters by finding their centre points and dimensions. This 
        creates the constraints necessary to fit a bounding box later.
        
        Tutorial at PCL docs helps with make_MomentOfInertiaEstimation aspect
        https://pcl.readthedocs.io/projects/tutorials/en/master/moment_of_inertia.html#moment-of-inertia
        """
        self.merged_clusters = [] # list of clusters similar to self.clusters_ec but merged
        to_merge_ys = [] # y values of clusters that are marked to be merged
        to_merge_clusters = [] # cluster indices of clusters that are marked to be merged

        for cluster_idx, indices in enumerate(np_pointcloud_cluster_indices):
            cloud = self.np_pointcloud[list(indices)] # numpy array cloud
            # convert to pcl object
            bb_cloud = pcl.PointCloud()
            bb_cloud.from_array(cloud) 
            # create feature extractor for bounding box
            feature_extractor = bb_cloud.make_MomentOfInertiaEstimation()
            feature_extractor.compute()
            # oriented bounding box
            _, _, position_OBB, _ = feature_extractor.get_OBB()
            y = float(position_OBB[0,1])

            # append to list that will then be checked for merging
            if abs(y) > self.merge_y_thresh:
                to_merge_ys.append(y)
                to_merge_clusters.append(indices)
            # otherwise append to already 'merged' clusters
            else:
                self.merged_clusters.append(indices)
            
        # check through each cluster to merge to find which have similar y values
        while len(to_merge_ys) > 1:
            # get distance between each cluster
            y = to_merge_ys[0] # y value to compare to other clusters
            distances = [other_y - y for other_y in to_merge_ys[1:]]
            # iterate through the distances and merge those with small distance
            merged_cluster = to_merge_clusters[0]
            leftover_indices = []
            for i, dist in enumerate(distances):
                if abs(dist) < 2.0:
                    merged_cluster.extend(to_merge_clusters[i+1])
                else:
                    leftover_indices.append(i+1)
            # append merged cluster and remove those that have already been merged
            self.merged_clusters.append(merged_cluster)            
            to_merge_ys = [to_merge_ys[i] for i in leftover_indices]
            to_merge_clusters = [to_merge_clusters[i] for i in leftover_indices]
        # put any leftover clusters into the merged clusters
        for leftover_cluster in to_merge_clusters:
            self.merged_clusters.append(leftover_cluster)
        
        self.get_logger().info(f"Merged {len(self.clusters_ec)} clusters into {len(self.merged_clusters)}.")
        return self.merged_clusters


    def create_detected_objects(self):
        """
        Create `DetectedObjectArray` by fitting bounding boxes to clusters.
        
        Tutorial at PCL docs helps with make_MomentOfInertiaEstimation aspect
        https://pcl.readthedocs.io/projects/tutorials/en/master/moment_of_inertia.html#moment-of-inertia
        """
        objects = DetectedObjectArray()

        self.L_shape_fitter = LShapeFitter()
        for cluster_idx, indices in enumerate(self.merged_clusters):
            cloud = self.cloud[list(indices)] # numpy array cloud

            # create detected object
            detected_object = DetectedObject()
            detected_object.object_id = cluster_idx # dummy value until we track the objects
            detected_object.frame_id = self.original_frame_id

            # perform L-shaped fitting on cluster
            self.get_logger().debug(f"Cluster {cluster_idx}")
            centre, width, length, yaw = self.L_shape_fitter.fit_rectangle(cloud)
            min_z = min(float(point[2]) for point in cloud)
            max_z = max(float(point[2]) for point in cloud)
            height = max_z - min_z

            detected_object.pose.position.x = centre[0]
            detected_object.pose.position.y = centre[1]
            detected_object.pose.position.z = -(max_z - min_z) / 2 # coordinate frame defined negative
            # dimensions -> assuming want distance from face to face
            detected_object.dimensions.x = length
            detected_object.dimensions.y = width
            detected_object.dimensions.z = height
            # object and signal class -> unknown for now
            detected_object.object_class = detected_object.CLASS_UNKNOWN
            detected_object.signal_state = detected_object.SIGNAL_UNKNOWN

            quat = euler2quat(0., 0., yaw)
            # pose -> assume of center point
            detected_object.pose.orientation.w = quat[0]
            detected_object.pose.orientation.x = quat[1]
            detected_object.pose.orientation.y = quat[2]
            detected_object.pose.orientation.z = quat[3]


            # perform rule based filtering for types of objects we want to track
            x = centre[0]
            y = centre[1]
            z = (max_z - min_z) / 2
            real_object = self.check_real_object(height, width, length, x, y, z)   

            # TODO change to commented section below once not debugging
            # assign objects that are real a signal state to then change bounding box colour for visualisation in RViz
            if real_object:
                detected_object.signal_state = detected_object.SIGNAL_GREEN
            objects.detected_objects.append(detected_object)

            # TODO this is the commented section
            # if real_object:
            #     objects.detected_objects.append(detected_object)

        return objects


    def get_L_shape_bb(self, cloud):
        """Finds L shaped bounding box based on algorithm in paper referenced below
        and code implemented in C++ found in Autoware AI.

        3D-LIDAR Multi Object Tracking for Autonomous Driving - A.S. Abdul Rachman
        https://github.com/Autoware-AI/core_perception/blob/master/lidar_naive_l_shape_detect/nodes/lidar_naive_l_shape_detect/lidar_naive_l_shape_detect.cpp

        Args:
            cloud (np.ndarray): numpy array representing the points in a cluster that a bounding box 
                should be fitted to
        """
        # thresholds for L shaped fitting
        _no_random_points = 80
        _slope_dist_thresh = 2
        _num_points_thresh = 10


        # initialise minimum and maximum points 
        # works by knowing that if all points are in positive quadrant
        # then y/x will produce largest number for point at large y and small x
        # and will produce smallest number for point at small y and large x
        min_point = [0,0] # x1 in paper
        max_point = [0,0] # x2 in paper
        min_slope = 1000 # preset to some large number
        max_slope = -1000 # preset to some small number
        for point in cloud:
            # store values from current point for calculation
            x_point = point[0]
            y_point = point[1]

            # calculate slope ?? line 233 autoware AI
            delta_m = y_point / x_point
            if delta_m < min_slope:
                min_slope = delta_m
                min_point = [x_point, y_point]
            if delta_m > max_slope:
                max_slope = delta_m
                max_point = [x_point, y_point]

            # L-shape fitting parameters
            dist_vec = [max_point[0]-min_point[0], max_point[1] - min_point[1]]
            slope_dist = np.sqrt(dist_vec[0]**2 + dist_vec[1]**2)
            slope = dist_vec[1] / dist_vec[0]

            if slope_dist > _slope_dist_thresh and len(cloud) > _num_points_thresh:
                max_dist = 0
                orthogonal_point = [0,0]

    def check_real_object(self, height, width, length, x,y,z):
        """Rule based filtering to determine whether a bounding box fitted is an object
        we want to track or not. Checks height, width, length and ratio of length to 
        width.

        Args:
            height (float): height of object
            width (float): width of object
            length (float): length of object
            x (float): x coordinate of object
            y (float): y coordinate of object
            z (float): z coordinate of object

        Returns:
            bool: whether object should be tracked or not
        """
        # parameters
        min_height = 0.8
        max_height = 3.5
        min_width = 0.5
        max_width = 3.
        min_length = 0.5
        max_length = 7.
        min_ratio = 1.3
        max_ratio = 5.
        min_l_for_ratio = 3

        top_area = width * length
        ratio_l_w = length/width

        if not(min_height <= height <= max_height):
            self.get_logger().info(f"Discarding item, not in height boundaries: {height:.3f} ({x:.3f},{y:.3f},{z:.3f})")
            return False
        if not(min_width <= width <= max_width):
            self.get_logger().info(f"Discarding item, not in width boundaries: {width:.3f} ({x:.3f},{y:.3f},{z:.3f})")
            return False
        if not(min_length <= length <= max_length):
            self.get_logger().info(f"Discarding item, not in length boundaries: {length:.3f} ({x:.3f},{y:.3f},{z:.3f})")
            return False
        if length > min_l_for_ratio and not(min_ratio <= ratio_l_w <= max_ratio):
            self.get_logger().info(f"Discarding item, not in ratio boundaries: {ratio_l_w:.3f} ({x:.3f},{y:.3f},{z:.3f})")
            return False
        return True

    def create_ros_markers(self, tracked_detected_objects):
        """Add bounding box markers to tracked detected objects for visualisation in RViz.
        Uses `MarkerArray` to create Rectangular Prisms.

        Args:
            tracked_detected_objects (DetectedObjectArray): objects that have been tracked from a frame
        """
        self.markers = MarkerArray() # list of markers for visualisations of boxes/IDs
        
        for d_o in tracked_detected_objects.detected_objects:
            # create number that shows ID of the detected object
            id_marker = Marker()
            id_marker.ns = 'object_id'
            id_marker.id = d_o.object_id
            id_marker.header.frame_id = self.original_frame_id
            id_marker.type = Marker.TEXT_VIEW_FACING
            id_marker.action = id_marker.ADD
            id_marker.pose.position.x = d_o.pose.position.x
            id_marker.pose.position.y = d_o.pose.position.y + 0.5
            id_marker.pose.position.z = d_o.pose.position.z
            id_marker.color.a = 1.
            id_marker.color.r = 1.
            id_marker.color.g = 1.
            id_marker.color.b = 1.
            id_marker.scale.x = 1.
            id_marker.scale.y = 0.8
            id_marker.scale.z = 0.5
            id_marker.text = f"{d_o.object_id}"
            self.markers.markers.append(id_marker)

            # create bounding boxes for visualisation
            bounding_box_marker = Marker()
            bounding_box_marker.ns = 'bounding_boxes'
            bounding_box_marker.id = d_o.object_id
            bounding_box_marker.header.frame_id = self.original_frame_id
            bounding_box_marker.type = Marker.CUBE
            bounding_box_marker.action = Marker.ADD
            bounding_box_marker.color.a = 0.5
            if d_o.signal_state == d_o.SIGNAL_GREEN:
                bounding_box_marker.color.r = 0.
            else:
                self.get_logger().debug(f"Discarded at pos: ({d_o.pose.position.x}, {d_o.pose.position.y})")
                bounding_box_marker.color.r = 255.
            bounding_box_marker.color.g = 255.
            bounding_box_marker.color.b = 255.
            bounding_box_marker.scale.x = d_o.dimensions.x
            bounding_box_marker.scale.y = d_o.dimensions.y
            bounding_box_marker.scale.z = d_o.dimensions.z
            bounding_box_marker.pose = d_o.pose
            self.markers.markers.append(bounding_box_marker)

        try:
            # delete bounding boxes and IDs that aren't present in this frame but were in previous
            for delete_id in self.tracker.deleted_ids:
                del_id_marker = Marker()
                del_id_marker.ns = 'object_id'
                del_id_marker.header.frame_id = self.original_frame_id
                del_id_marker.id = delete_id
                del_id_marker.action = Marker.DELETE
                self.markers.markers.append(del_id_marker)
                del_bb_marker = Marker()
                del_bb_marker.ns = 'bounding_boxes'
                del_bb_marker.header.frame_id = self.original_frame_id
                del_bb_marker.id = delete_id
                del_bb_marker.action = Marker.DELETE
                self.markers.markers.append(del_bb_marker)
        except AttributeError: # if only 1 frame there are no bounding boxes to delete
            pass

    

def main(args=None):
    rclpy.init(args=args)

    pcl2_subscriber = PCL2Subscriber()
    
    try:
        rclpy.spin(pcl2_subscriber)
    except KeyboardInterrupt:
        pcl2_subscriber.get_logger().debug("Keyboard interrupt")

    # destroy node explicity
    pcl2_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()