import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from collections import deque

class GridMapper:
    def __init__(self, config):
        self.config = config
        self.resolution = config['map']['resolution']
        self.size_meters = np.array(config['map']['size'])
        self.origin = np.array(config['map']['origin']) * self.resolution
        self.size_pixels = (self.size_meters * self.resolution).astype(int)
        
        # Log-odds map
        self.log_odds_map = np.zeros(self.size_pixels, dtype=np.float32)
        self.log_odds_map.fill(config['map']['log_odds_min'])
        
        # Constants
        self.prob_hit = config['map']['prob_occ']
        self.log_min = config['map']['log_odds_min']
        self.log_max = config['map']['log_odds_max']
        self.laser_max_range = config['laser']['max_range'] * self.resolution
        self.hit_threshold_margin = config['laser'].get('hit_threshold_margin', 0.05)
        self.free_decay = config['map'].get('free_decay', 0.3)

        # Odometry smoothing buffer
        self.pose_buffer = deque(maxlen=5)
        
    def world_to_map(self, world_coords):
        """
        Convert world coordinates (meters) to map coordinates (pixels)
        
        Args:
            world_coords: (x,y) tuple in meters (world frame)
            
        Returns:
            (i,j) tuple of grid cell indices
        """
        # Convert to continuous map coordinates first (meters in map frame)
        map_x = world_coords[0] + self.origin[0] / self.resolution
        map_y = world_coords[1] + self.origin[1] / self.resolution
        
        # Convert to discrete grid cells
        i = int(np.floor(map_x * self.resolution))
        j = int(np.floor(map_y * self.resolution))
        
        # Clip to map bounds
        i = np.clip(i, 0, self.size_pixels[0] - 1)
        j = np.clip(j, 0, self.size_pixels[1] - 1)
        
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
        map_x = map_coords[0] / self.resolution
        map_y = map_coords[1] / self.resolution
        
        # Convert to world coordinates
        world_x = map_x - self.origin[0] / self.resolution
        world_y = map_y - self.origin[1] / self.resolution
        
        return (world_x, world_y)
    
    def update_map(self, pose, scan):
        # Smooth pose with moving average
        self.pose_buffer.append(pose)
        smoothed_pose = np.mean(self.pose_buffer, axis=0)

        pose_px = self.world_to_map(smoothed_pose[:2])
        angle_min = self.config['laser']['min_angle']
        angle_max = self.config['laser']['max_angle']
        angle_increment = (angle_max - angle_min) / len(scan)

        for i, dist in enumerate(scan):
            if dist <= 0 or dist >= self.config['laser']['max_range']:
                continue

            # Skip noisy spikes
            if i > 0 and abs(scan[i] - scan[i - 1]) > 1.0:
                continue

            angle = angle_min + i * angle_increment
            global_angle = smoothed_pose[2] + angle

            hit_x = smoothed_pose[0] + dist * np.cos(global_angle)
            hit_y = smoothed_pose[1] + dist * np.sin(global_angle)
            hit_px = self.world_to_map([hit_x, hit_y])

            free_cells = self.bresenham(pose_px, hit_px)

            # Update free cells with decay factor
            for cell in free_cells[:-1]:
                i_cell, j_cell = cell
                if 0 <= i_cell < self.size_pixels[0] and 0 <= j_cell < self.size_pixels[1]:
                    self.log_odds_map[i_cell, j_cell] = max(
                        self.log_odds_map[i_cell, j_cell] + self.free_decay * self.log_odds_update(False),
                        self.log_min
                    )

            # Only update hit cell if beam likely hit a real obstacle
            if dist < (self.config['laser']['max_range'] - self.hit_threshold_margin):
                i_hit, j_hit = hit_px
                if 0 <= i_hit < self.size_pixels[0] and 0 <= j_hit < self.size_pixels[1]:
                    self.log_odds_map[i_hit, j_hit] = min(
                        self.log_odds_map[i_hit, j_hit] + self.log_odds_update(True),
                        self.log_max
                    )

    
    def get_probability_map(self, inflate_radius=0):
        """Get probability map with optional obstacle inflation"""
        prob_map = 1 - 1 / (1 + np.exp(np.clip(self.log_odds_map, self.log_min, self.log_max)))
        if inflate_radius > 0:
            prob_map = binary_dilation(prob_map > 0.5, iterations=inflate_radius).astype(float)
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
    
    def log_odds_update(self, occupied):
        """Log odds ratio update"""
        return np.log(self.prob_hit / (1 - self.prob_hit)) if occupied else np.log((1 - self.prob_hit) / self.prob_hit)