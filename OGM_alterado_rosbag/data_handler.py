#!/usr/bin/env python3
import rospy
import rosbag
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from message_filters import ApproximateTimeSynchronizer, Subscriber

class Turtlebot3SensorData:
    def __init__(self, config, use_bag=False, bag_path=None):
        self.config = config
        self.use_bag = use_bag
        self.bag_path = bag_path
        self.bag_iterator = None  # Internal generator if in rosbag mode
        self.current_pose = None
        self.current_scan = None
        self.last_scan_time = None
        self.last_pose_time = None
        self.scan_range_max = None
        self.scan_range_min = None 
        self.scan_angle_inc = None  
        self.scan_angle_min = None 
        self.scan_angle_max = None
 
        if not self.use_bag:
            self._setup_live_ros()
        else:
            self._setup_bag_mode()
    
    def _setup_live_ros(self):
        rospy.init_node('turtlebot3_data_handler', anonymous=True)

        scan_sub = Subscriber("/scan", LaserScan)
        odom_sub = Subscriber("/odom", Odometry)

        self.ts = ApproximateTimeSynchronizer(
            [scan_sub, odom_sub],
            queue_size=10,
            slop=0.05
        )
        self.ts.registerCallback(self.sync_callback)

    def _setup_bag_mode(self):
        rospy.init_node('mapper_node', anonymous=True, disable_signals=True)
        if not self.bag_path:
            raise ValueError("Bag path must be provided if use_bag is True")

        self.bag = rosbag.Bag(self.bag_path)
        self.scan_msg = None
        self.odom_msg = None
        self.bag_iterator = self._bag_generator()

    def _bag_generator(self):
        for topic, msg, t in self.bag.read_messages(topics=["/scan", "/odom"]):
            if topic == "/scan":
                self.scan_msg = msg
            elif topic == "/odom":
                self.odom_msg = msg

            if self.scan_msg and self.odom_msg:
                time_diff = abs((self.scan_msg.header.stamp - self.odom_msg.header.stamp).to_sec())
                if time_diff < 0.1:
                    self.sync_callback(self.scan_msg, self.odom_msg)
                    yield  # Yield control so `get_latest_data()` can be called
                    self.scan_msg = None
                    self.odom_msg = None

        
    def sync_callback(self, scan_msg, odom_msg):
        """Synchronized callback for scan and odometry data"""
        # Process scan
        ranges = np.array(scan_msg.ranges)
        ranges[np.isinf(ranges)] = scan_msg.range_max
        ranges = np.clip(ranges, scan_msg.range_min, scan_msg.range_max)
        self.current_scan = ranges
        self.last_scan_time = scan_msg.header.stamp
        self.scan_range_max = scan_msg.range_max
        self.scan_range_min = scan_msg.range_min
        self.scan_angle_inc = scan_msg.angle_increment
        self.scan_angle_min = scan_msg.angle_min
        self.scan_angle_max = scan_msg.angle_max

        # Process odometry
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        orientation = odom_msg.pose.pose.orientation
        (_, _, theta) = euler_from_quaternion([orientation.x, orientation.y, 
                                              orientation.z, orientation.w])
        if theta < 0: 
            theta = 2 * np.pi + theta
        self.current_pose = np.array([x, y, theta])
        self.last_pose_time = odom_msg.header.stamp
        
    def get_latest_data(self):
        """
        Get the most recent synchronized sensor data.
        
        Returns:
            tuple: (pose, scan, max_range, min_range, angle_inc, angle_min, angle_max)
        """
        if self.use_bag:
            try:
                next(self.bag_iterator)
            except StopIteration:
                return None, None, None, None, None, None, None

        if (self.current_pose is not None and 
            self.current_scan is not None and
            abs((self.last_pose_time - self.last_scan_time).to_sec()) < 0.1):
            return (self.current_pose, self.current_scan,
                    self.scan_range_max, self.scan_range_min,
                    self.scan_angle_inc, self.scan_angle_min, self.scan_angle_max)

        return None, None, None, None, None, None, None