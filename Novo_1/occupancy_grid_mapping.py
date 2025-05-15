import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

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
    
    def world_to_continuous_map(self, world_coords):
        """
        Convert world coordinates to continuous map coordinates
        
        Args:
            world_coords: (x,y) tuple in meters (world frame)
            
        Returns:
            (x,y) tuple in meters (map frame)
        """
        return (
            world_coords[0] + self.origin[0] / self.resolution,
            world_coords[1] + self.origin[1] / self.resolution
        )
    
    def continuous_map_to_world(self, map_coords):
        """
        Convert continuous map coordinates to world coordinates
        
        Args:
            map_coords: (x,y) tuple in meters (map frame)
            
        Returns:
            (x,y) tuple in meters (world frame)
        """
        return (
            map_coords[0] - self.origin[0] / self.resolution,
            map_coords[1] - self.origin[1] / self.resolution
        )
    
    def update_map(self, pose, scan):
        """Update map with new pose and scan data"""
        # Convert pose to map coordinates
        pose_px = self.world_to_map(pose[:2])
        angle_min = self.config['laser']['min_angle']
        angle_increment = (self.config['laser']['max_angle'] - angle_min) / len(scan)
        
        # Process each beam
        for i, dist in enumerate(scan):
            if dist <= 0 or dist > self.config['laser']['max_range']:
                continue
                
            angle = angle_min + i * angle_increment
            global_angle = pose[2] + angle
            
            # Calculate hit point in world coordinates
            hit_x = pose[0] + dist * np.cos(global_angle)
            hit_y = pose[1] + dist * np.sin(global_angle)
            hit_px = self.world_to_map([hit_x, hit_y])
            
            # Raytrace from pose to hit point
            free_cells = self.bresenham(pose_px, hit_px)
            
            # Update log odds
            for cell in free_cells[:-1]:  # Free cells
                if 0 <= cell[0] < self.size_pixels[0] and 0 <= cell[1] < self.size_pixels[1]:
                    self.log_odds_map[cell[0], cell[1]] = max(
                        self.log_odds_map[cell[0], cell[1]] + self.log_odds_update(False),
                        self.log_min
                    )
            
            # Update hit cell
            if 0 <= hit_px[0] < self.size_pixels[0] and 0 <= hit_px[1] < self.size_pixels[1]:
                self.log_odds_map[hit_px[0], hit_px[1]] = min(
                    self.log_odds_map[hit_px[0], hit_px[1]] + self.log_odds_update(True),
                    self.log_max
                )
    
    def get_probability_map(self, inflate_radius=0):
        """Get probability map with optional obstacle inflation"""
        prob_map = 1 - 1 / (1 + np.exp(np.clip(self.log_odds_map, self.log_min, self.log_max)))
        if inflate_radius > 0:
            prob_map = binary_dilation(prob_map > 0.5, iterations=inflate_radius).astype(float)
        return prob_map
    
    def save_map(self, filename):
        """Save map to file"""
        np.savez_compressed(
            filename,
            log_odds_map=self.log_odds_map,
            resolution=self.resolution,
            origin=self.origin,
            config=self.config
        )
    
    @classmethod
    def load_map(cls, filename):
        """Load map from file"""
        data = np.load(filename, allow_pickle=True)
        mapper = cls(data['config'].item())
        mapper.log_odds_map = data['log_odds_map']
        return mapper
    
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