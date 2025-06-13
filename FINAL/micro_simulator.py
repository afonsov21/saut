# ========== micro_simulator.py ==========
# A lightweight ROS-based simulator that mimics a robot navigating through a corridor.
# Publishes fake LIDAR and odometry data based on geometric raycasting.

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
        """Initialize simulator state and ROS publishers."""
        rospy.init_node('micro_simulator')

        # ROS publishers for scan and odometry
        self.scan_pub = rospy.Publisher('/scan', LaserScan, queue_size=10)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)

        # Robot pose (meters and radians)
        self.x = 2
        self.y = 2
        self.yaw = 0.0  # orientation

        # Timing and simulation control
        self.rate = rospy.Rate(10)  # 10 Hz
        self.sim_time = 0
        self.step = 0
        self.save_interval = 10
        self.trajectory = []

        # Waypoints for robot navigation
        self.waypoints = [(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)]
        self.current_index = 0
        self.reached_goal = False

        # Speed settings
        self.linear_speed = 0.2           # m/s
        self.angular_speed = math.pi / 4  # rad/s

        # Corridor and room layout (wall segments)
        self.walls = [
            (1, 1, 3, 1), (4, 1, 6, 1), (7, 1, 9, 1),
            (3, 3, 7, 3),
            (1, 9, 3, 9), (4, 9, 6, 9), (7, 9, 9, 9),
            (3, 7, 7, 7),
            (1, 1, 1, 3), (1, 4, 1, 6), (1, 7, 1, 9),
            (3, 3, 3, 7),
            (9, 1, 9, 3), (9, 4, 9, 6), (9, 7, 9, 9),
            (7, 3, 7, 7),
            (2, 0, 5, 0), (5, 0, 8, 0),
            (2, 10, 5, 10), (5, 10, 8, 10),
            (0, 2, 0, 5), (0, 5, 0, 8),
            (10, 2, 10, 5), (10, 5, 10, 8)
        ]

        # Entrance positions (for potential interaction)
        self.entrances = [
            (3.5, 2), (6.5, 2), (8, 3.5), (8, 6.5),
            (6.5, 8), (3.5, 8), (2, 6.5), (2, 3.5)
        ]

        # Looking logic (disabled for now)
        self.looking = False
        self.look_steps = 0
        self.max_look_steps = 20
        self.look_angle_per_step = math.pi / 20
        self.rotating_to_entrance = False
        self.rotating_back = False
        self.original_yaw = 0.0
        self.target_entrance_yaw = 0.0

    def raycast(self, angle):
        """
        Perform simple raycasting from current robot pose in given angle.
        Returns:
            Simulated distance to nearest wall (with Gaussian noise).
        """
        x1, y1 = self.x, self.y
        x2 = x1 + 10 * math.cos(angle + self.yaw)
        y2 = y1 + 10 * math.sin(angle + self.yaw)

        closest_dist = 3.5  # LIDAR max range

        for wall in self.walls:
            x3, y3, x4, y4 = wall

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                continue  # parallel lines

            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

            if 0 <= t <= 1 and 0 <= u <= 1:
                intersect_x = x1 + t * (x2 - x1)
                intersect_y = y1 + t * (y2 - y1)
                dist = math.hypot(x1 - intersect_x, y1 - intersect_y)
                closest_dist = min(closest_dist, dist)

        return closest_dist + np.random.normal(0, 0.02)

    def generate_scan(self):
        """Generate and return a simulated LaserScan message."""
        scan = LaserScan()
        scan.header.stamp = rospy.Time.now()
        scan.header.frame_id = "base_link"
        scan.angle_min = -math.pi / 2
        scan.angle_max = math.pi / 2
        scan.angle_increment = math.pi / 180
        scan.range_min = 0.1
        scan.range_max = 3.5
        scan.ranges = [self.raycast(scan.angle_min + i * scan.angle_increment)
                       for i in range(181)]
        return scan

    def angle_diff(self, a, b):
        """Compute shortest angular difference between angles a and b."""
        diff = a - b
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def generate_odom(self):
        """Update robot pose and return simulated Odometry message."""
        self.sim_time += 0.1

        # End of path
        if self.reached_goal:
            rospy.signal_shutdown("Finished waypoint loop")
            return self._create_odom(self.x, self.y, 0)

        # Get next target waypoint
        x_goal, y_goal = self.waypoints[self.current_index + 1]
        dx = x_goal - self.x
        dy = y_goal - self.y
        distance = math.hypot(dx, dy)

        # If we're at the waypoint, advance to next
        if distance < 0.05:
            self.current_index += 1
            if self.current_index >= len(self.waypoints) - 1:
                self.reached_goal = True
            return self._create_odom(self.x, self.y, self.yaw)

        # Compute heading to goal
        target_yaw = math.atan2(dy, dx)
        yaw_error = self.angle_diff(target_yaw, self.yaw)

        # Smooth rotation
        max_yaw_change = self.angular_speed * 0.1
        if abs(yaw_error) > max_yaw_change:
            self.yaw += max_yaw_change if yaw_error > 0 else -max_yaw_change
        else:
            self.yaw = target_yaw

        # Move forward if facing close to goal
        if abs(yaw_error) < math.radians(10):
            move_dist = min(self.linear_speed * 0.1, distance)
            self.x += move_dist * math.cos(self.yaw)
            self.y += move_dist * math.sin(self.yaw)

        # Save trajectory for plot
        self.trajectory.append((self.x, self.y))
        self.step += 1
        if self.step % self.save_interval == 0:
            self.plot_real_map()

        return self._create_odom(self.x, self.y, self.yaw)

    def _create_odom(self, x, y, yaw):
        """Helper to build an Odometry message."""
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose = Pose(
            Point(x, y, 0),
            Quaternion(*quaternion_from_euler(0, 0, yaw))
        )
        return odom

    def plot_real_map(self):
        """Plot the real map layout and the robot trajectory."""
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
        plt.savefig("/mnt/c/Users/Pedro Coelho/Desktop/SAut_local/Novo_3/real_map.png")
        plt.close()

    def run(self):
        """Main loop: publish scan and odom messages continuously."""
        while not rospy.is_shutdown():
            self.scan_pub.publish(self.generate_scan())
            self.odom_pub.publish(self.generate_odom())
            self.rate.sleep()

# ===== Main Entry Point =====
if __name__ == '__main__':
    try:
        sim = MicroSimulator()
        sim.run()
    except rospy.ROSInterruptException:
        pass
