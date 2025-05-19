#!/usr/bin/env python3
import rospy
import math
import numpy as np
import matplotlib.pyplot as plt
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
        self.x = 2
        self.y = 2
        self.yaw = 0.0
        
        # Simulation parameters
        self.rate = rospy.Rate(10)  # 10Hz
        self.sim_time = 0
        
        self.trajectory = []
        self.save_interval = 10  # Save every 10 steps
        self.step = 0   
        
        self.waypoints = [
            (2, 2), (8, 2), (8, 8), (2, 8), (2, 2)  # Full square loop
        ]
        self.current_index = 0
        self.reached_goal = False

        
        # Replace self.walls with this corridor+rooms layout:
        self.walls = [
            # Bottom corridor wall (with gaps at 3 and 6)
            (1, 1, 3, 1), (4, 1, 6, 1), (7, 1, 9, 1),
            
            (3, 3, 7, 3),
            
            # Top corridor wall (with gaps at 3 and 6)
            (1, 9, 3, 9), (4, 9, 6, 9), (7, 9, 9, 9),
            
            (3, 7, 7, 7),
            
            # Left corridor wall (with gaps at y=3 and y=6)
            (1, 1, 1, 3), (1, 4, 1, 6), (1, 7, 1, 9),
            
            (3, 3, 3, 7),
            
            # Right corridor wall (with gaps at y=3 and y=6)
            (9, 1, 9, 3), (9, 4, 9, 6), (9, 7, 9, 9),
            
            (7, 3, 7, 7),

            # Rooms outside bottom corridor
            (2, 0, 2, 1), (5, 0, 5, 1), (2, 0, 5, 0),  # Room 1
            (5, 0, 5, 1), (8, 0, 8, 1), (5, 0, 8, 0),  # Room 2
           
            # Rooms outside top corridor
            (2, 9, 2, 10), (5, 9, 5, 10), (2, 10, 5, 10),
            (5, 9, 5, 10), (8, 9, 8, 10), (5, 10, 8, 10),

            # Rooms outside left corridor
            (0, 2, 1, 2), (0, 5, 1, 5), (0, 2, 0, 5),
            (0, 5, 1, 5), (0, 8, 1, 8), (0, 5, 0, 8),

            # Rooms outside right corridor
            (9, 2, 10, 2), (9, 5, 10, 5), (10, 2, 10, 5),
            (9, 5, 10, 5), (9, 8, 10, 8), (10, 5, 10, 8),
        ]
        
        self.entrances = [
            (3.5, 2),  # bottom corridor gap at x=3
            (6.5, 2),
            (8, 3.5),  # top corridor gap at x=3
            (8, 6.5),
            (6.5, 8),
            (3.5, 8),
            (2, 6.5),
            (2, 3.5),
        ]
        self.looking = False
        self.look_steps = 0
        self.max_look_steps = 20  # How many steps to rotate in place
        self.look_angle_per_step = (math.pi / 20)  # rotate ~180 degrees over 20 steps
        self.rotating_to_entrance = False
        self.rotating_back = False
        self.original_yaw = 0.0
        self.target_entrance_yaw = 0.0


        # Change robot path to traverse corridor
        self.linear_speed = 0.2
        self.angular_speed = math.pi/4
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
    
    def is_near_entrance(self):
        for ex, ey in self.entrances:
            if math.hypot(self.x - ex, self.y - ey) < 0.2:
                return True
        return False

    @staticmethod
    def angle_diff(a, b):
        """Compute smallest difference between two angles"""
        diff = a - b
        while diff > math.pi:
            diff -= 2*math.pi
        while diff < -math.pi:
            diff += 2*math.pi
        return diff

    def generate_odom(self):
        self.sim_time += 0.1

        if self.reached_goal:
            rospy.signal_shutdown("Finished waypoint loop")
            odom = Odometry()
            odom.header.stamp = rospy.Time.now()
            odom.header.frame_id = "odom"
            odom.child_frame_id = "base_link"
            odom.pose.pose = Pose(Point(0,0,0), Quaternion(0,0,0,1))
            return odom

        # If currently rotating at entrance
        if self.looking:
            # Rotate in place
            self.yaw += self.look_angle_per_step
            self.look_steps += 1

            if self.look_steps >= self.max_look_steps:
                self.looking = False
                self.look_steps = 0

            odom = Odometry()
            odom.header.stamp = rospy.Time.now()
            odom.header.frame_id = "odom"
            odom.child_frame_id = "base_link"
            odom.pose.pose = Pose(
                Point(self.x, self.y, 0),
                Quaternion(*quaternion_from_euler(0, 0, self.yaw))
            )
            return odom

        # Check if near entrance
        '''
        if self.is_near_entrance():
            self.looking = False #Suposto ser True
            self.look_steps = 0
            return self.generate_odom()  # Enter look mode immediately
        '''
        # Get current and next waypoint
        x_goal, y_goal = self.waypoints[self.current_index + 1]
        dx = x_goal - self.x
        dy = y_goal - self.y
        distance = math.hypot(dx, dy)
        
        # If close to the next waypoint, advance to next
        if distance < 0.05:
            self.current_index += 1
            if self.current_index >= len(self.waypoints) - 1:
                self.reached_goal = True
            return self.generate_odom()

        target_yaw = math.atan2(dy, dx)

        # Smooth yaw turning
        max_yaw_change = self.angular_speed * 0.1  # max angular speed * dt
        yaw_error = self.angle_diff(target_yaw, self.yaw)
        if abs(yaw_error) > max_yaw_change:
            self.yaw += max_yaw_change if yaw_error > 0 else -max_yaw_change
        else:
            self.yaw = target_yaw

        # Move forward only if facing roughly target direction
        if abs(self.angle_diff(target_yaw, self.yaw)) < math.pi / 18:  # 10 degrees tolerance
            move_dist = min(self.linear_speed * 0.1, distance)
            self.x += move_dist * math.cos(self.yaw)
            self.y += move_dist * math.sin(self.yaw)

        # Create odom message
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose = Pose(
            Point(self.x, self.y, 0),
            Quaternion(*quaternion_from_euler(0, 0, self.yaw))
        )

        self.trajectory.append((self.x, self.y))
        self.step += 1
        if self.step % self.save_interval == 0:
            self.plot_real_map()

        return odom


    def plot_real_map(self):
        plt.figure(figsize=(10, 6))
        for wall in self.walls:
            x1, y1, x2, y2 = wall
            plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

        traj = np.array(self.trajectory)
        if traj.size > 0:
            plt.plot(traj[:, 0], traj[:, 1], 'r-', label='Robot Path')

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Real Map with Robot Path")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.savefig("/mnt/c/Users/Pedro Coelho/Desktop/SAut_local/Novo_1/real_map.png")
        plt.close()
      

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