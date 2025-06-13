# ========== main.py ==========
# Entry point for running occupancy grid mapping.
# Loads config, initializes components, and starts the update loop.

#!/usr/bin/env python3

# ===== Imports =====
import rospy                          # ROS Python API
import yaml                           # For reading YAML configuration
import numpy as np
from pathlib import Path              # To manage file paths across OSes
from threading import Lock            # To avoid race conditions with shared data

# Project-specific modules
from data_handler import Turtlebot3SensorData    # LIDAR and odometry handler
from occupancy_grid_mapping import GridMapper    # Core mapping logic
from live_map import LiveMapVisualizer           # Map + LIDAR visualizer

# ===== Main Application Class =====
class OccupancyGridMapping:
    def __init__(self, config_path):
        """
        Initialize the full occupancy grid mapping system.
        Loads configuration and sets up all subsystems.
        """
        # Load configuration file
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load sensor interface: live or bag mode
        self.data_handler = Turtlebot3SensorData(
            self.config, use_bag=use_bag, bag_path=bag_path
        )

        # Initialize map and visualizer
        self.grid_mapper = GridMapper(self.config)
        self.visualizer = LiveMapVisualizer(self.config)

        # Thread lock to protect shared map access
        self.lock = Lock()

        # History of robot poses for drawing trail
        self.pose_history = []

        # Run flag (will be cleared on shutdown)
        self.running = True

    def run(self):
        """
        Main ROS loop: fetch data, update map, and redraw visualization.
        """
        rate = rospy.Rate(self.config['ros']['rate'])  # loop frequency in Hz

        while not rospy.is_shutdown() and self.running:
            # Retrieve latest synchronized pose and scan
            pose, scan, max_range, min_range, angle_inc, angle_min, angle_max = \
                self.data_handler.get_latest_data()

            # Process and visualize if data is valid
            if pose is not None and scan is not None:
                with self.lock:
                    # Update internal occupancy grid
                    self.grid_mapper.update_map(
                        pose, scan, max_range, min_range,
                        angle_inc, angle_min, angle_max
                    )

                    # Save pose for path trail
                    self.pose_history.append(pose)

                    # Refresh visual output
                    self.visualizer.update(
                        pose, scan, self.grid_mapper,
                        min_range, max_range,
                        angle_inc, angle_min, angle_max,
                        self.pose_history
                    )

            # Publish map as a ROS message
            self.grid_mapper.publish_ros_map()

            # Wait for next loop iteration
            rate.sleep()

    def save_map(self, filename):
        """
        Save the current map to disk (thread-safe).
        Args:
            filename: name for saved .npz map file
        """
        with self.lock:
            self.grid_mapper.save_map(filename)

    def shutdown(self):
        """Signal the loop to stop on shutdown."""
        self.running = False

# ===== Main Entry Point =====
if __name__ == '__main__':
    try:
        # Load path to configuration file
        config_path = Path(__file__).parent / 'config.yaml'

        # Configuration: set to True if running from a rosbag
        use_bag = True
        bag_path = str(Path(__file__).parent / 'elevador.bag')

        # Instantiate and run the mapper
        mapper = OccupancyGridMapping(config_path)

        # Clean shutdown on Ctrl+C
        rospy.on_shutdown(mapper.shutdown)

        # Start main loop
        mapper.run()

    except rospy.ROSInterruptException:
        # Graceful exit on ROS interrupt
        pass
