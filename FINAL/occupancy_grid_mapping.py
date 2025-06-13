# ========== occupancy_grid_mapping.py ==========
# Core mapping logic for building an occupancy grid map from synchronized LIDAR and odometry data.
# Uses a log-odds formulation to incrementally update belief in occupancy of each cell.

# ===== Imports =====
import numpy as np
import math
from collections import defaultdict, deque
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion
import rospy

# ===== Main Mapper Class =====
class GridMapper:
    def __init__(self, config):
        """
        Initialize the occupancy grid mapper.
        Loads configuration and sets up internal log-odds map storage.
        """
        self.config = config

        # Map dimensions in world coordinates (meters)
        self.x_min = config['map']['x_min']
        self.x_max = config['map']['x_max']
        self.y_min = config['map']['y_min']
        self.y_max = config['map']['y_max']
        self.resolution = config['map']['resolution']  # meters per cell

        # This will later be updated to the upper-left corner of the map in grid coordinates
        self.map_origin = (0, 0)

        # Probabilities for occupancy, free space, and unknown
        self.prob_occ = config['map']['prob_occ']
        self.prob_free = config['map']['prob_free']
        self.prob_unk = config['map']['prob_unknown']
        self.log_min = config['map']['log_odds_min']
        self.log_max = config['map']['log_odds_max']

        # Use a dictionary to store only the visited cells (sparse map)
        self.log_odds_map = defaultdict(lambda: np.log(0.5 / (1 - 0.5)))  # log-odds of 0.5 = 0

    def log_odds_update(self, prob):
        """Convert probability (0 to 1) to log-odds representation."""
        return np.log(prob / (1 - prob))

    def prob_update(self, log_val):
        """Convert log-odds back to bounded probability (0 to 1)."""
        return 1 - 1 / (1 + np.exp(np.clip(log_val, self.log_min, self.log_max)))

    def world_to_map(self, world_coords):
        """
        Convert world coordinates (meters) to grid indices.
        Args:
            world_coords: tuple (x, y) in meters
        Returns:
            tuple (i, j): grid row and column
        """
        x, y = world_coords
        i = int((y - self.y_min) / self.resolution)
        j = int((x - self.x_min) / self.resolution)
        return (i, j)

    def map_to_world(self, map_coords):
        """
        Convert grid indices (i, j) to world coordinates (x, y).
        Args:
            map_coords: tuple (i, j)
        Returns:
            tuple (x, y) in meters
        """
        i, j = map_coords
        x = j * self.resolution + self.x_min
        y = i * self.resolution + self.y_min
        return (x, y)

    def update_map(self, pose, scan, max_range, min_range, angle_inc, angle_min, angle_max):
        """
        Update the occupancy grid with a new LIDAR scan.
        Args:
            pose: robot's pose (x, y, theta)
            scan: array of distance measurements
            max_range, min_range: LIDAR distance limits
            angle_inc: angle increment between rays
            angle_min, angle_max: min and max angles of scan
        """
        pose_px = self.world_to_map(pose[:2])
        x, y, theta = pose
        angle = angle_min

        for dist in scan:
            # Sanitize distance
            if math.isnan(dist):
                dist = max_range
            dist = np.clip(dist, min_range, max_range)

            # Compute the global direction of the ray
            angle += angle_inc
            global_angle = theta + angle

            # Compute the hit point in world coordinates
            hit_x = x + dist * math.cos(global_angle)
            hit_y = y + dist * math.sin(global_angle)
            hit_px = self.world_to_map((hit_x, hit_y))

            # Get all cells along the ray from robot to hit (excluding the hit cell)
            free_cells = self.bresenham(pose_px, hit_px)[:-1]

            # If the measurement was not max range, mark the end cell as occupied
            if dist < max_range:
                # Weight inversely proportional to distance
                weight = 1.0 - ((dist - min_range) / (max_range - min_range))
                weight = np.clip(weight, 0.0, 1.0)

                self.log_odds_map[hit_px] += weight * (
                    self.log_odds_update(self.prob_occ) - self.log_odds_update(self.prob_unk)
                )

            # Mark all cells before the hit as free
            for cell in free_cells:
                self.log_odds_map[cell] += (
                    self.log_odds_update(self.prob_free) - self.log_odds_update(self.prob_unk)
                )

    def get_probability_map(self):
        """
        Convert the internal log-odds representation into a dense probability grid.
        Returns:
            2D NumPy array of occupancy probabilities.
        """
        if not self.log_odds_map:
            return np.zeros((1, 1))

        # Determine map size from visited cells
        keys = np.array(list(self.log_odds_map.keys()))
        i_min, j_min = np.min(keys, axis=0)
        i_max, j_max = np.max(keys, axis=0)
        height = i_max - i_min + 1
        width = j_max - j_min + 1

        # Initialize full probability map to 0.5 (unknown)
        prob_map = np.full((height, width), 0.5, dtype=np.float32)

        # Fill in visited cells with updated probabilities
        for (i, j), log_val in self.log_odds_map.items():
            p = self.prob_update(log_val)
            if p <= 0.05:
                p = 0.0
            elif p >= 0.95:
                p = 1.0
            else:
                p = 0.5
            prob_map[i - i_min, j - j_min] = p

        # Save the map origin for correct alignment
        self.map_origin = (i_min, j_min)
        return prob_map

    def bresenham(self, start, end):
        """
        Bresenham's Line Algorithm — get discrete grid cells between two points.
        Args:
            start: (i0, j0)
            end: (i1, j1)
        Returns:
            List of (i, j) cells along the line
        """
        x0, y0 = start
        x1, y1 = end
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return cells

    def publish_ros_map(self):
        """
        Publish the current map as a ROS OccupancyGrid message on topic 'my_map'.
        """
        prob_map = self.get_probability_map()
        origin_i, origin_j = self.map_origin
        height, width = prob_map.shape

        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        # Set grid info
        msg.info.resolution = self.resolution
        msg.info.width = width
        msg.info.height = height

        # Set map origin in world coordinates
        origin_world = self.map_to_world((origin_i, origin_j))
        msg.info.origin = Pose(
            position=Point(origin_world[0], origin_world[1], 0.0),
            orientation=Quaternion(0, 0, 0, 1)
        )

        # Convert each probability to ROS occupancy format (0–100 or -1 for unknown)
        msg.data = []
        for i in range(height):
            for j in range(width):
                p = prob_map[i, j]
                if p == 0.5:
                    msg.data.append(-1)  # unknown
                else:
                    msg.data.append(int(p * 100))  # 0–100

        # Create publisher only once
        if not hasattr(self, 'map_pub'):
            self.map_pub = rospy.Publisher('my_map', OccupancyGrid, queue_size=1, latch=True)
        self.map_pub.publish(msg)
