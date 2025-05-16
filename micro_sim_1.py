import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi, log, exp, atan2, sqrt
import random

class OccupancyGridMapping:
    def __init__(self, map_size=10, resolution=0.1):
        # Sensor parameters (RPLIDAR A1 specs)
        self.max_range = 5.5  # meters
        self.angular_res = 1 * (pi/180)  # 1 degree in radians
        self.beam_width = 1 * (pi/180)  # 1 degree
        self.obstacle_thickness = 0.1  # meters
        
        # Map parameters
        self.map_size = map_size
        self.resolution = resolution
        self.grid_size = int(map_size / resolution)
        self.logodds_map = np.zeros((self.grid_size, self.grid_size))
        
        # Log-odds parameters
        self.l_occ = log(0.7/(1-0.7))  # Occupied
        self.l_free = log(0.3/(1-0.3))  # Free
        self.l_0 = log(0.5/(1-0.5))     # Prior
        
        # Visualization
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.canvas.manager.set_window_title('Occupancy Grid Mapping Simulator')
        
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        return (int(x/self.resolution), int(y/self.resolution))
    
    def grid_to_world(self, i, j):
        """Convert grid indices to world coordinates"""
        return (i*self.resolution, j*self.resolution)
    
    def check_boundaries(self, x, y):
        """Check if position is within map boundaries"""
        return (0 <= x < self.map_size) and (0 <= y < self.map_size)
    
    def check_collision(self, x, y, env):
        """Check if position collides with an obstacle"""
        i, j = self.world_to_grid(x, y)
        if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
            return env[i, j] == 1
        return False
    
    def enforce_boundaries(self, x, y):
        """Enforce map boundaries by clamping values"""
        x = max(0, min(x, self.map_size - self.resolution))
        y = max(0, min(y, self.map_size - self.resolution))
        return x, y
    
    def create_environment(self):
        """Create a simple test environment"""
        env = np.zeros((self.grid_size, self.grid_size))
        # Add walls
        env[10:30, 50] = 1  # Vertical wall
        env[75, 25:75] = 1  # Horizontal wall
        # Add some obstacles
        env[40:60, 30] = 1  # Additional vertical obstacle
        env[45, 20:40] = 1  # Additional horizontal obstacle
        return env
    
    def inverse_sensor_model(self, robot_pos, angle, measurement, cell_pos):
        """Calculate log-odds update for a grid cell"""
        dx = cell_pos[0] - robot_pos[0]
        dy = cell_pos[1] - robot_pos[1]
        r = sqrt(dx**2 + dy**2)
        phi = atan2(dy, dx) - robot_pos[2]
        
        # Normalize angle
        phi = (phi + pi) % (2*pi) - pi
        
        # Check if cell is in sensor FOV
        if abs(phi - angle) > self.beam_width/2:
            return self.l_0
        
        # Apply inverse sensor model
        if r > min(self.max_range, measurement + self.obstacle_thickness/2):
            return self.l_0
        
        if measurement < self.max_range and abs(r - measurement) < self.obstacle_thickness/2:
            return self.l_occ - self.l_0
        
        if r <= measurement:
            return self.l_free - self.l_0
        
        return 0
    
    def update_map(self, robot_pos, measurements):
        """Update the occupancy grid with new sensor data"""
        for angle, dist in measurements:
            # Get cells along the beam
            cells = self.get_beam_cells(robot_pos, angle, dist)
            
            for (i, j) in cells:
                cell_center = self.grid_to_world(i + 0.5, j + 0.5)
                update = self.inverse_sensor_model(robot_pos, angle, dist, cell_center)
                self.logodds_map[i, j] += update
    
    def get_beam_cells(self, robot_pos, angle, dist):
        """Bresenham's line algorithm for beam cells"""
        x0, y0 = self.world_to_grid(robot_pos[0], robot_pos[1])
        end_x = robot_pos[0] + dist * cos(robot_pos[2] + angle)
        end_y = robot_pos[1] + dist * sin(robot_pos[2] + angle)
        x1, y1 = self.world_to_grid(end_x, end_y)
        
        cells = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        
        while True:
            if 0 <= x0 < self.grid_size and 0 <= y0 < self.grid_size:
                cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy
        
        return cells
    
    def simulate_sensor(self, robot_pos, env):
        """Simulate LIDAR measurements"""
        measurements = []
        for angle in np.arange(0, 2*pi, self.angular_res):
            dist = self.max_range
            for r in np.linspace(0, self.max_range, int(self.max_range/0.05)):
                x = robot_pos[0] + r * cos(robot_pos[2] + angle)
                y = robot_pos[1] + r * sin(robot_pos[2] + angle)
                
                # Check if we hit the map boundary first
                if not self.check_boundaries(x, y):
                    dist = r * random.gauss(1.0, 0.05)  # Add noise
                    break
                    
                i, j = self.world_to_grid(x, y)
                if 0 <= i < self.grid_size and 0 <= j < self.grid_size and env[i, j] == 1:
                    dist = r * random.gauss(1.0, 0.05)  # Add noise
                    break
            measurements.append((angle, dist))
        return measurements
    
    def visualize(self, env, robot_pos, step):
        """Update visualization with inverted grayscale colormap"""
        self.ax1.clear()
        self.ax2.clear()
        
        # Real environment (black and white)
        self.ax1.imshow(env, cmap='binary', vmin=0, vmax=1, origin='lower')
        self.ax1.set_title(f"Real Environment (Step {step})")
        rx, ry = self.world_to_grid(robot_pos[0], robot_pos[1])
        self.ax1.plot(ry, rx, 'ro')  # Red dot for robot
        
        # Occupancy grid (INVERTED grayscale: white=free, black=occuppied)
        prob_map = 1/(1 + np.exp(-self.logodds_map))  # Probability of occupancy
        self.ax2.imshow(prob_map, cmap='binary', vmin=0, vmax=1, origin='lower')
        self.ax2.set_title("Occupancy Grid Map (White=Free)")
        self.ax2.plot(ry, rx, 'ro')  # Red dot for robot
        plt.tight_layout()
        plt.pause(0.05)
    
    def run_simulation(self, steps=1000):
        """Main simulation loop"""
        env = self.create_environment()
        robot_pos = np.array([1.0, 1.0, 0.0])  # x, y, theta
        
        for step in range(steps):
            # Get sensor measurements
            measurements = self.simulate_sensor(robot_pos, env)
            
            # Update map
            self.update_map(robot_pos, measurements)
            
            # Simple movement with collision detection
            new_x = robot_pos[0] + 0.1 * cos(robot_pos[2])
            new_y = robot_pos[1] + 0.1 * sin(robot_pos[2])
            
            # Check for collisions with boundaries or obstacles
            if not self.check_boundaries(new_x, new_y) or self.check_collision(new_x, new_y, env):
                robot_pos[2] += pi/2  # Turn 90 degrees when hitting an obstacle
                continue
            
            robot_pos[0] = new_x
            robot_pos[1] = new_y
            robot_pos[2] += random.uniform(-0.1, 0.1)
            
            # Visualize
            self.visualize(env, robot_pos, step)
        
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    mapper = OccupancyGridMapping(map_size=10, resolution=0.1)
    mapper.run_simulation(steps=1000)