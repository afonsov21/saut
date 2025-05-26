import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, binary_erosion, binary_dilation
from collections import deque
import numpy as np
from collections import deque, defaultdict
from datetime import datetime
import math
from scipy.ndimage import median_filter
import rospy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion

class GridMapper:
    def __init__(self, config):
        self.config = config
        self.x_min = config['map']['x_min']
        self.x_max = config['map']['x_max']
        self.y_min = config['map']['y_min']
        self.y_max = config['map']['y_max']
        self.resolution = config['map']['resolution']
        #self.size_meters = np.array(config['map']['size'])
        #self.origin = np.array(config['map']['origin']) * self.resolution
        #self.size_pixels = (self.size_meters * self.resolution).astype(int)
        self.map_origin = (0,0)
        
        # Log-odds map
        self.log_odds_map = defaultdict(lambda:  np.log(0.5 / (1 - 0.5)))
        
        # Constants
        self.prob_occ = config['map']['prob_occ']
        self.prob_free = config['map']['prob_free']
        self.prob_unk = config['map']['prob_unknown']
        self.log_min = config['map']['log_odds_min']
        self.log_max = config['map']['log_odds_max']
        
        
        self.hit_threshold_margin =  0.05
        self.free_decay = 0.2
        self.hit_counts = defaultdict(int)
        self.miss_counts = defaultdict(int)
        
        # Odometry smoothing buffer
        self.pose_buffer = deque(maxlen=5)
    
    def log_odds_update(self, prob):
        """Log odds ratio update"""
        return np.log(prob / (1 - prob))
    
    def prob_update(self, log_val):
        """probability update"""
        return 1 - 1 / (1 + np.exp(np.clip(log_val, self.log_min, self.log_max)))

        
    def world_to_map(self, world_coords):
        """
        Convert world coordinates (meters) to map coordinates (pixels)
        
        Args:
            world_coords: (x,y) tuple in meters (world frame)
            
        Returns:
            (i,j) tuple of grid cell indices
        """
        # Convert to continuous map coordinates first (meters in map frame)
        map_x = world_coords[0] 
        map_y = world_coords[1] 
        
        # Convert to discrete grid cells
        i = int((map_y - self.y_min) / self.resolution) #row
        j = int((map_x - self.x_min) / self.resolution) #col
        
    
        return (i, j)
    
    def map_to_world(self, map_coords):
        """
        Convert map coordinates (pixels) to world coordinates (meters)
        
        Args:
            map_coords: (i,j) tuple of grid cell indices
            
        Returns:
            (x,y) tuple in meters (world frame)
        """
        # Convert to continuous map coordinates first
        map_x = map_coords[1] * self.resolution 
        map_y = map_coords[0] * self.resolution 
        
        # Convert to world coordinates
        world_x = map_x + self.x_min
        world_y = map_y + self.y_min
        
        return (world_x, world_y)
    
    def update_map(self, pose, scan, max_range, min_range, angle_inc, angle_min, angle_max):
        # Smooth pose with moving average
        # Smooth pose using buffer
        #self.pose_buffer.append(pose)
        #smoothed_pose = np.mean(self.pose_buffer, axis=0) if self.pose_buffer else pose
        #pose_px = self.world_to_map(smoothed_pose[:2])
        #pose = smoothed_pose
        
        pose_px = self.world_to_map(pose[:2])
        x, y, theta = pose
        
        ### Process each scan ###
        angle_increment = angle_inc
        angle = angle_min
        for i, dist in enumerate(scan):
            if dist <= min_range:
                dist = min_range
            
            if dist >= max_range:
                dist = max_range

            if math.isnan(dist) : dist = max_range
            
            angle += angle_increment
            ### --- ###
            
            global_angle = theta + angle

            hit_x = x + dist * np.cos(global_angle)
            hit_y = y + dist * np.sin(global_angle)
            hit_px = self.world_to_map([hit_x, hit_y])

             # Get cells along the ray
            free_cells = self.bresenham(pose_px, hit_px)[:-1]

            # Update logic
            if dist < max_range:
                self.log_odds_map[hit_px] += self.log_odds_update(self.prob_occ) + self.log_odds_update(self.prob_unk)

            for cell in free_cells:
                self.log_odds_map[cell] += self.log_odds_update(self.prob_free) + self.log_odds_update(self.prob_unk)
    '''
    def _update_cell(self, cell, occupied, distance, max_range=False):
        """Weight updates by distance and sensor reliability."""
        if occupied:
            self.hit_counts[cell] += 1
            if self.hit_counts[cell] >= 3:  # Require 3 hits to confirm occupation
                # Stronger update for closer hits
                #weight = np.exp(-distance / 5.0)  # Decay with distance
                self.log_odds_map[cell] = np.clip(
                    self.log_odds_map[cell] +  self.log_odds_update(True),
                    self.log_min, self.log_max
                )
        else:
            self.miss_counts[cell] += 1
            if self.miss_counts[cell] >= 2:  # Require 2 misses to confirm free
                # Weaker update for max-range free cells
                #weight = 0.3 if max_range else np.exp(-distance / 2.0)
                self.log_odds_map[cell] = np.clip(
                    self.log_odds_map[cell] +  self.log_odds_update(False),
                    self.log_min, self.log_max
                )
    '''
    def get_probability_map(self):
        if not self.log_odds_map:
            return np.zeros((1, 1))

        keys = np.array(list(self.log_odds_map.keys()))
        i_min, j_min = np.min(keys, axis=0)
        i_max, j_max = np.max(keys, axis=0)

        height = i_max - i_min + 1
        width = j_max - j_min + 1

        prob_map = np.full((height, width), 0.5, dtype=np.float32)

        for (i, j), log_val in self.log_odds_map.items():
            p = self.prob_update(log_val)
            if(p <= 0.05): p = 0.0
            if(p >= 0.95): p = 1.0
            if(p > 0.05 and p < 0.95): p = 0.5
            prob_map[i - i_min, j - j_min] = p
            

        self.map_origin = (i_min, j_min)
        return prob_map
        
    @staticmethod
    def bresenham(start, end):
        """Bresenham's line algorithm"""
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
        prob_map = self.get_probability_map()
        origin_i, origin_j = self.map_origin
        height, width = prob_map.shape

        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"

        msg.info.resolution = self.resolution
        msg.info.width = width
        msg.info.height = height

        # Origem do mapa em coordenadas do mundo
        origin_world = self.map_to_world((origin_i, origin_j))
        msg.info.origin = Pose()
        msg.info.origin.position = Point(origin_world[0], origin_world[1], 0.0)
        msg.info.origin.orientation = Quaternion(0, 0, 0, 1)

        # Conversão para int8
        flat_data = []
        for i in range(height):
            for j in range(width):
                p = prob_map[i, j]
                if p == 0.5:
                    flat_data.append(-1)  # unknown
                else:
                    flat_data.append(int(p * 100))

        msg.data = flat_data

        # Publica no tópico /map
        if not hasattr(self, 'map_pub'):
            self.map_pub = rospy.Publisher('my_map', OccupancyGrid, queue_size=1, latch=True)
        self.map_pub.publish(msg)