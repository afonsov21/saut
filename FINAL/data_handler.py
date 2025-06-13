# ========== data_handler.py ==========
# This module handles input from either a live ROS topic or a ROS bag file.
# It synchronizes LIDAR and odometry data and extracts robot pose and scan info.

#!/usr/bin/env python3

# ===== Imports =====
import rospy
import rosbag
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from message_filters import ApproximateTimeSynchronizer, Subscriber

class Turtlebot3SensorData:
    def __init__(self, config, use_bag=False, bag_path=None):
        """
        Initialize the data handler for live mode or bag mode.
        Args:
            config: configuration dictionary
            use_bag: whether to use rosbag mode
            bag_path: path to rosbag file (if use_bag is True)
        """
        self.config = config
        self.use_bag = use_bag
        self.bag_path = bag_path
        self.bag_iterator = None

        # Latest synchronized data
        self.current_pose = None
        self.current_scan = None
        self.last_scan_time = None
        self.last_pose_time = None

        # Scan parameters
        self.scan_range_max = None
        self.scan_range_min = None 
        self.scan_angle_inc = None  
        self.scan_angle_min = None 
        self.scan_angle_max = None

        # Setup ROS or bag data source
        if not self.use_bag:
            self._setup_live_ros()
        else:
            self._setup_bag_mode()

    def _setup_live_ros(self):
        """Set up live ROS subscribers using time synchronizer."""
        rospy.init_node('turtlebot3_data_handler', anonymous=True)

        # Subscribe to scan and odometry topics
        scan_sub = Subscriber("/scan", LaserScan)
        odom_sub = Subscriber("/odom", Odometry)

        # Synchronize messages based on timestamp closeness
        self.ts = ApproximateTimeSynchronizer(
            [scan_sub, odom_sub],
            queue_size=10,
            slop=0.05
        )
        self.ts.registerCallback(self.sync_callback)

    def _setup_bag_mode(self):
        """Initialize rosbag reading and scan/odom synchronization."""
        rospy.init_node('mapper_node', anonymous=True, disable_signals=True)
        if not self.bag_path:
            raise ValueError("Bag path must be provided if use_bag is True")

        self.bag = rosbag.Bag(self.bag_path)
        self.scan_msg = None
        self.odom_msg = None
        self.bag_iterator = self._bag_generator()

    def _bag_generator(self):
        """
        Generator that reads from rosbag and yields when both
        LIDAR and odometry messages are synchronized.
        """
        for topic, msg, _ in self.bag.read_messages(topics=["/scan", "/odom"]):
            if topic == "/scan":
                self.scan_msg = msg
            elif topic == "/odom":
                self.odom_msg = msg

            if self.scan_msg and self.odom_msg:
                time_diff = abs((self.scan_msg.header.stamp - self.odom_msg.header.stamp).to_sec())
                if time_diff < 0.1:
                    self.sync_callback(self.scan_msg, self.odom_msg)
                    yield  # Pause here to let `get_latest_data()` use the new data
                    self.scan_msg = None
                    self.odom_msg = None

    def sync_callback(self, scan_msg, odom_msg):
        """
        Callback triggered when a synchronized scan and odometry message pair is received.
        Processes and stores the latest pose and scan.
        """
        # === Process scan ===
        ranges = np.array(scan_msg.ranges)

        # Replace infs with max range and clip
        ranges[np.isinf(ranges)] = scan_msg.range_max
        ranges = np.clip(ranges, scan_msg.range_min, scan_msg.range_max)

        self.current_scan = ranges
        self.last_scan_time = scan_msg.header.stamp

        # Save scan metadata
        self.scan_range_max = scan_msg.range_max
        self.scan_range_min = scan_msg.range_min
        self.scan_angle_inc = scan_msg.angle_increment
        self.scan_angle_min = scan_msg.angle_min
        self.scan_angle_max = scan_msg.angle_max

        # === Process odometry ===
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y

        # Convert quaternion to yaw angle
        orientation = odom_msg.pose.pose.orientation
        (_, _, theta) = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

        # Normalize theta to [0, 2π]
        if theta < 0: 
            theta += 2 * np.pi

        self.current_pose = np.array([x, y, theta])
        self.last_pose_time = odom_msg.header.stamp

    def get_latest_data(self):
        """
        Return the most recent synchronized scan and pose.
        Returns:
            tuple: (pose, scan, max_range, min_range, angle_inc, angle_min, angle_max)
        """
        # In bag mode, advance the generator
        if self.use_bag:
            try:
                next(self.bag_iterator)
            except StopIteration:
                return None, None, None, None, None, None, None

        # Check for valid, time-synchronized data
        if (self.current_pose is not None and 
            self.current_scan is not None and
            abs((self.last_pose_time - self.last_scan_time).to_sec()) < 0.05):
            return (self.current_pose, self.current_scan,
                    self.scan_range_max, self.scan_range_min,
                    self.scan_angle_inc, self.scan_angle_min, self.scan_angle_max)

        # Data not ready or out of sync
        return None, None, None, None, None, None, None
