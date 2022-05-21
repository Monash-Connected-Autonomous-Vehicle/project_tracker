#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import logging
import time

import numpy as np

from sensor_msgs.msg import PointCloud2 as PCL2
from geometry_msgs.msg import TwistStamped
from visualization_msgs.msg import MarkerArray, Marker

from project_tracker.utils import PCL2_2_numpy, create_colour_list, pcl_to_ros
from project_tracker.tracking import Tracker

from mcav_interfaces.msg import DetectedObject, DetectedObjectArray

import pcl


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

        # publishers
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

        # create tracker for identifying and following objects over time
        # self.tracker = Tracker(max_frames_before_forget=2, max_frames_length=30, tracking_method="centre_distance", dist_threshold=5)
        # self.tracker = Tracker(max_frames_before_forget=2, max_frames_length=30, tracking_method="iou", iou_threshold=0.85)
        self.tracker = Tracker(
            max_frames_before_forget=2, max_frames_length=30, tracking_method="both", 
            iou_threshold=0.85, dist_threshold = 5
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
        # convert to pcl.PointCloud_PointXYZRGB for visualisation in RViz
        cluster_colour_cloud = pcl.PointCloud_PointXYZRGB()
        cluster_colour_cloud.from_list(self.colour_cluster_point_list)
        timestamp = self.get_clock().now().to_msg()
        pcl2_msg = pcl_to_ros(cluster_colour_cloud, timestamp, self.original_frame_id) # convert the pcl to a ROS PCL2 message
        self._cloud_cluster_publisher.publish(pcl2_msg)

        # create DetectedObjectArray
        start = time.time()
        detected_objects = self.create_detected_objects() 

        # track objects over time
        tracked_detected_objects = self.tracker.update(detected_objects, timestamp=msg.header.stamp)
        self.get_logger().debug(f"Number of tracked objects: {len(tracked_detected_objects.detected_objects)}")

        # create bounding boxes and ID labels
        self.create_bounding_boxes(tracked_detected_objects)

        tracking_time = time.time() - start
        self.get_logger().debug(f"Tracking took {tracking_time:.5f}s")

        # publish bounding boxes and detected objects
        self._bounding_boxes_publisher.publish(self.markers)
        self._detected_objects_publisher.publish(tracked_detected_objects)

    def euclidean_clustering_ec(self, ec_cloud, ec_tree):
        """
        Perform euclidean clustering with a given pcl.PointCloud and kdtree

        Args:
            ec_cloud (pcl.PointCloud): pcl version of pointcloud message received
            ec_tree(pcl.kdtree): kdtree from pcl-python binding created with pcl.PointCloud.make_kdtree()
        """
        # make euclidean cluster extraction method
        ec = ec_cloud.make_EuclideanClusterExtraction()
        # set parameters
        ec.set_ClusterTolerance(self.cluster_tolerance)
        ec.set_MinClusterSize(self.min_cluster_size)
        ec.set_MaxClusterSize(self.max_cluster_size)
        ec.set_SearchMethod(ec_tree)
        # perform euclidean clustering and return indices
        self.clusters_ec = ec.Extract()
        return

    def merge_y_clusters(self):
        """Merge clusters if they have a large absolute y value (side to side). Large absolute y 
        value indicates that the cluster is probably a wall. Don't want to fit multiple clusters
        to wall.

        Returns:
            list: clusters from euclidean clustering that have been merged based on y value
        """
        self.merged_clusters = [] # list of clusters similar to self.clusters_ec but merged
        to_merge_ys = [] # y values of clusters that are marked to be merged
        to_merge_clusters = [] # cluster indices of clusters that are marked to be merged

        # iterate through all of the current clusters and note where large absolute y values occur
        for indices in self.clusters_ec:
            # extract centroids
            cloud = self.cloud[list(indices)] # numpy array cloud
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
            if abs(y) > 7.:
                to_merge_ys.append(y)
                to_merge_clusters.append(indices)
            # otherwise append to already 'merged' clusters
            else:
                self.merged_clusters.append(indices)
            
        # check through each cluster to merge to find which have similar y values
        while len(to_merge_ys) > 1:
            # get distance between each cluster
            y = to_merge_ys[0] # y value to compare to other clusters
            distances = [comp - y for comp in to_merge_ys[1:]]
            # iterate through the distances and merge those with small distance
            merged_cluster = []
            leftover_indices = []
            for i, dist in enumerate(distances):
                if abs(dist) < 2.0:
                    merged_cluster.extend(to_merge_clusters[i])
                else:
                    leftover_indices.append(i+1)
            # append merged cluster and remove those that have already been merged
            self.merged_clusters.append(merged_cluster)            
            to_merge_ys = [to_merge_ys[i] for i in leftover_indices]
            to_merge_clusters = [to_merge_clusters[i] for i in leftover_indices]
        # put any leftover clusters into the merged clusters
        for leftover_cluster in to_merge_clusters:
            self.merged_clusters.append(leftover_cluster)
        
        self.get_logger().info(f"Merged {len(self.clusters_ec)-len(self.merged_clusters)} clusters.")
        return self.merged_clusters


    def create_detected_objects(self):
        """
        Create `DetectedObjectArray` from the clusters by finding their centre points and dimensions. This 
        creates the constraints necessary to fit a bounding box later.

        TODO: replace with L-shape fitting
        
        Tutorial at PCL docs helps with make_MomentOfInertiaEstimation aspect
        https://pcl.readthedocs.io/projects/tutorials/en/master/moment_of_inertia.html#moment-of-inertia
        """
        objects = DetectedObjectArray()

        for cluster_idx, indices in enumerate(self.merged_clusters):
            cloud = self.cloud[list(indices)] # numpy array cloud
            # convert to pcl object
            bb_cloud = pcl.PointCloud()
            bb_cloud.from_array(cloud) 

            # create feature extractor for bounding box
            feature_extractor = bb_cloud.make_MomentOfInertiaEstimation()
            feature_extractor.compute()
            # oriented bounding box
            [min_point_OBB, max_point_OBB, position_OBB,
                rotational_matrix_OBB] = feature_extractor.get_OBB()

            # create detected object
            detected_object = DetectedObject()
            detected_object.object_id = cluster_idx # dummy value until we track the objects
            detected_object.frame_id = self.original_frame_id


            ### COMMENTED OUT AS WE ARE NOT ESTIMATING YAW HERE ANYMORE
            ### WE ARE GOING TO ESTIMATE YAW BASED ON VELOCITY VECTORS

            # # convert rotational matrix to quaternion for use in pose
            # roll, pitch, yaw = mat2euler(rotational_matrix_OBB)
            # while not(-10. < yaw*180/np.pi < 10.):
            #     yaw -= np.sign(yaw) * 0.15
            # quat = euler2quat(0., 0., yaw)
            # # pose -> assume of center point
            # detected_object.pose.orientation.w = quat[0]
            # detected_object.pose.orientation.x = quat[1]
            # detected_object.pose.orientation.y = quat[2]
            # detected_object.pose.orientation.z = quat[3]
            # # orientation -> restricted to rotate only around the z axis i.e. flat to ground plane
            # mag = sqrt(quat[0]**2 + quat[3]**2)
            # detected_object.pose.orientation.w = float(quat[0]/mag)
            # detected_object.pose.orientation.x = 0. #float(quat[1])
            # detected_object.pose.orientation.y = 0. #float(quat[2]/mag)
            # detected_object.pose.orientation.z = float(quat[2]/mag)#float(quat[3])


            detected_object.pose.position.x = float(position_OBB[0,0])
            detected_object.pose.position.y = float(position_OBB[0,1])
            detected_object.pose.position.z = float(position_OBB[0,2]) 
            # dimensions -> assuming want distance from face to face
            detected_object.dimensions.x = 2 * float(max_point_OBB[0,0])
            detected_object.dimensions.y = 2 * float(max_point_OBB[0,1])
            detected_object.dimensions.z = 2 * float(max_point_OBB[0,2])
            # object and signal class -> unknown for now
            detected_object.object_class = detected_object.CLASS_UNKNOWN
            detected_object.signal_state = detected_object.SIGNAL_UNKNOWN


            # perform rule based filtering for types of objects we want to track
            object_height = 2 * float(max_point_OBB[0,2])
            object_width = 2 * float(max_point_OBB[0,1])
            object_length = 2 * float(max_point_OBB[0,0])
            
            x = float(position_OBB[0,0])
            y = float(position_OBB[0,1])
            z = float(position_OBB[0,2]) 
            real_object = self.check_real_object(object_height, object_width, object_length,
            x, y, z)   

            if real_object:
                objects.detected_objects.append(detected_object)


        return objects

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

    def create_bounding_boxes(self, tracked_detected_objects):
        """Add bounding boxes to tracked detected objects for visualisation in RViz.
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