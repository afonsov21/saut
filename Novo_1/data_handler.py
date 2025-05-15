#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from message_filters import ApproximateTimeSynchronizer, Subscriber

class Turtlebot3SensorData:
    def __init__(self, config):
        self.config = config
        self.current_pose = None
        self.current_scan = None
        self.last_scan_time = None
        self.last_pose_time = None
        
        # ROS Initialization
        rospy.init_node('turtlebot3_data_handler', anonymous=True)
        
        # Setup synchronized subscribers
        scan_sub = Subscriber(config['ros']['scan_topic'], LaserScan)
        odom_sub = Subscriber(config['ros']['odom_topic'], Odometry)
        
        self.ts = ApproximateTimeSynchronizer(
            [scan_sub, odom_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)
        
    def sync_callback(self, scan_msg, odom_msg):
        """Synchronized callback for scan and odometry data"""
        # Process scan
        ranges = np.array(scan_msg.ranges)
        ranges[np.isinf(ranges)] = scan_msg.range_max
        ranges = np.clip(ranges, scan_msg.range_min, self.config['laser']['max_range'])
        self.current_scan = ranges
        self.last_scan_time = scan_msg.header.stamp
        
        # Process odometry
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        orientation = odom_msg.pose.pose.orientation
        (_, _, theta) = euler_from_quaternion([orientation.x, orientation.y, 
                                              orientation.z, orientation.w])
        self.current_pose = np.array([x, y, theta])
        self.last_pose_time = odom_msg.header.stamp
        
    def get_latest_data(self):
        """Returns the most recent synchronized pose and scan"""
        if (self.current_pose is not None and 
            self.current_scan is not None and
            abs((self.last_pose_time - self.last_scan_time).to_sec()) < 0.1):
            return self.current_pose, self.current_scan
        return None, None