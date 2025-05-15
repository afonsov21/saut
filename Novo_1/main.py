#!/usr/bin/env python3
import rospy
import yaml
import numpy as np
from pathlib import Path
from threading import Lock
from data_handler import Turtlebot3SensorData
from occupancy_grid_mapping import GridMapper
from live_map import LiveMapVisualizer

class OccupancyGridMapping:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.data_handler = Turtlebot3SensorData(self.config)
        self.grid_mapper = GridMapper(self.config)
        self.visualizer = LiveMapVisualizer(self.config)
        self.lock = Lock()
        
        # Tracking variables
        self.pose_history = []
        self.running = True
        
    def run(self):
        rate = rospy.Rate(self.config['ros']['rate'])
        while not rospy.is_shutdown() and self.running:
            # Get synchronized data
            pose, scan = self.data_handler.get_latest_data()
            
            if pose is not None and scan is not None:
                with self.lock:
                    # Update map
                    self.grid_mapper.update_map(pose, scan)
                    
                    # Update visualization
                    self.pose_history.append(pose)
                    if len(self.pose_history) > self.config['plot']['trail_length']:
                        self.pose_history.pop(0)
                    
                    self.visualizer.update(
                        pose, 
                        scan, 
                        self.grid_mapper, 
                        self.pose_history
                    )
            
            rate.sleep()
    
    def save_map(self, filename):
        with self.lock:
            self.grid_mapper.save_map(filename)
    
    def shutdown(self):
        self.running = False

if __name__ == '__main__':
    try:
        config_path = Path(__file__).parent / 'config.yaml'
        mapper = OccupancyGridMapping(config_path)
        rospy.on_shutdown(mapper.shutdown)
        mapper.run()
    except rospy.ROSInterruptException:
        pass