#!/usr/bin/env python3
import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler

class MicroSimulator:
    def __init__(self):
        rospy.init_node('micro_simulator')
        
        # Publishers
        self.scan_pub = rospy.Publisher('/scan', LaserScan, queue_size=10)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        
        # Robot state
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        
        # Simulation parameters
        self.rate = rospy.Rate(10)  # 10Hz
        self.sim_time = 0
        
        # Replace self.walls with this corridor+rooms layout:
        self.walls = [
            # Main corridor (long horizontal walls)
            (-10, -1, 10, -1),  # Bottom corridor wall
            (-10, 1, 10, 1),     # Top corridor wall
            
            # Rooms on bottom side
            (-8, -1, -8, -5),    # Left room left wall
            (-8, -5, -5, -5),    # Left room bottom wall
            (-5, -5, -5, -1),    # Left room right wall
            
            (2, -1, 2, -4),      # Center room left wall
            (2, -4, 6, -4),      # Center room bottom wall
            (6, -4, 6, -1),      # Center room right wall
            
            # Rooms on top side
            (-7, 1, -7, 4),      # Top-left room left wall
            (-7, 4, -3, 4),      # Top-left room top wall
            (-3, 4, -3, 1),      # Top-left room right wall
            
            (4, 1, 4, 3),        # Top-right room left wall
            (4, 3, 8, 3),        # Top-right room top wall
            (8, 3, 8, 1)         # Top-right room right wall
        ]

        # Change robot path to traverse corridor
        self.linear_speed = 0.5
        self.corridor_length = 20
    
    def raycast(self, angle):
        """Simple raycasting against walls"""
        x1, y1 = self.x, self.y
        x2 = x1 + 10 * math.cos(angle + self.yaw)
        y2 = y1 + 10 * math.sin(angle + self.yaw)
        
        closest_dist = 3.5  # max range
        
        for wall in self.walls:
            x3, y3, x4, y4 = wall
            
            # Line intersection math
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if denom == 0:
                continue
                
            t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
            u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
            
            if 0 <= t <= 1 and 0 <= u <= 1:
                dist = math.sqrt((x1 - (x1+t*(x2-x1)))**2 + 
                              (y1 - (y1+t*(y2-y1)))**2)
                if dist < closest_dist:
                    closest_dist = dist
        
        return closest_dist + np.random.normal(0, 0.02)  # Add small noise

    def generate_scan(self):
        scan = LaserScan()
        scan.header.stamp = rospy.Time.now()
        scan.header.frame_id = "base_link"
        scan.angle_min = -math.pi/2
        scan.angle_max = math.pi/2
        scan.angle_increment = math.pi / 180  # 1 degree resolution
        scan.range_min = 0.1
        scan.range_max = 3.5
        scan.ranges = [self.raycast(scan.angle_min + i*scan.angle_increment) 
                      for i in range(181)]  # 181 readings (-90° to +90°)
        return scan

    def generate_odom(self):
        self.sim_time += 0.1
        
        # Straight path down corridor with pauses at rooms
        if self.sim_time < 10:  # First half of corridor
            self.x = -8 + self.linear_speed * self.sim_time
            self.y = 0
            self.yaw = 0
        elif 10 <= self.sim_time < 12:  # Pause at first room
            self.x = 2
            self.y = 0
            self.yaw = -math.pi/2  # Look into room
        elif 12 <= self.sim_time < 22:  # Second half
            self.x = 2 + self.linear_speed * (self.sim_time-12)
            self.y = 0
            self.yaw = 0
        else:  # Loop
            self.sim_time = 0

        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose = Pose(
            Point(self.x, self.y, 0),
            Quaternion(*quaternion_from_euler(0, 0, self.yaw)))
        return odom

    def run(self):
        while not rospy.is_shutdown():
            # Publish data
            self.scan_pub.publish(self.generate_scan())
            self.odom_pub.publish(self.generate_odom())
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        sim = MicroSimulator()
        sim.run()
    except rospy.ROSInterruptException:
        pass