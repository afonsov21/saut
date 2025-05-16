import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi, log, exp, atan2, sqrt
import random

class OccupancyGridMapping:
    def __init__(self, map_size=10, resolution=0.1):
        # Updated LIDAR parameters (based on our specifications)
        self.min_range = 0.12  # 120mm minimum range (in meters)
        self.max_range = 3.5    # 3,500mm maximum range (in meters)
        self.angular_res = 1 * (pi/180)  # 1° angular resolution
        self.scan_rate = 300 / 60  # Convert rpm to scans per second
        self.beam_width = 1 * (pi/180)  # Assuming 1° beam width
        
        # Noise parameters based on specifications
        self.close_range_threshold = 0.5  # 500mm threshold
        self.close_range_acc = 0.015  # ±15mm accuracy (<500mm)
        self.close_range_prec = 0.010  # ±10mm precision (<500mm)
        self.far_range_acc = 0.05     # ±5% accuracy (≥500mm)
        self.far_range_prec = 0.035   # ±3.5% precision (≥500mm)
        
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
        self.fig.canvas.manager.set_window_title('Occupancy Grid Mapping with Realistic LIDAR')        
    
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
        """Create a classroom environment with desks, walls, and features"""
        env = np.zeros((self.grid_size, self.grid_size))
        
        # Convert measurements to grid cells (1m = 1/0.1 = 10 cells)
        wall_thickness = int(0.2 / self.resolution)  # 20cm thick walls
        desk_size = int(0.6 / self.resolution)      # 60cm square desks
        aisle_width = int(1.2 / self.resolution)     # 1.2m aisles
        
        # Outer walls (leave opening for door)
        env[:wall_thickness, :] = 1                      # Top wall
        env[-wall_thickness:, :] = 1                     # Bottom wall
        env[:, :wall_thickness] = 1                      # Left wall
        env[:, -wall_thickness:] = 1                     # Right wall
        door_start = int(0.4 * self.grid_size)
        door_end = int(0.6 * self.grid_size)
        env[door_start:door_end, -wall_thickness:] = 0   # Door opening
        
        # Teacher's area at front (top of map)
        teacher_desk_start = int(0.3 * self.grid_size)
        teacher_desk_end = int(0.7 * self.grid_size)
        front_wall_pos = int(0.1 * self.grid_size)
        env[front_wall_pos:front_wall_pos+wall_thickness, :] = 1  # Front wall
        env[front_wall_pos+wall_thickness:front_wall_pos+wall_thickness+desk_size, 
            teacher_desk_start:teacher_desk_end] = 1  # Teacher's desk
        
        # Whiteboard (on front wall)
        whiteboard_start = int(0.2 * self.grid_size)
        whiteboard_end = int(0.8 * self.grid_size)
        env[front_wall_pos-wall_thickness:front_wall_pos, 
            whiteboard_start:whiteboard_end] = 1
        
        # Student desks (6 rows with 5 desks each)
        for row in range(5):  # 5 rows of desks
            row_pos = int((0.2 + 0.12 * row) * self.grid_size)
            for col in range(5):  # 5 desks per row
                col_pos = int((0.15 + 0.15 * col) * self.grid_size)
                env[row_pos:row_pos+desk_size, col_pos:col_pos+desk_size] = 1
        
        
        return env
    
    def inverse_sensor_model(self, robot_pos, angle, measurement, cell_pos):
        """Updated sensor model with realistic range constraints"""
        dx = cell_pos[0] - robot_pos[0]
        dy = cell_pos[1] - robot_pos[1]
        r = sqrt(dx**2 + dy**2)
        
        # Check if cell is within valid range
        if r < self.min_range or r > self.max_range:
            return 0
            
        phi = atan2(dy, dx) - robot_pos[2]
        phi = (phi + pi) % (2*pi) - pi  # Normalize angle
        
        if abs(phi - angle) > self.beam_width/2:
            return 0
            
        # Dynamic obstacle thickness based on distance
        obstacle_thickness = max(0.05, 0.02 * r)  # Increases with distance
        
        if measurement < self.max_range and abs(r - measurement) < obstacle_thickness/2:
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
            """Simulate LIDAR measurements with realistic noise characteristics"""
            measurements = []
            scan_start_time = 0  # For simulating scan rate
            
            for angle in np.arange(0, 2*pi, self.angular_res):
                dist = self.max_range
                
                # Simulate ray casting with realistic step size
                for r in np.linspace(self.min_range, self.max_range, 
                                    int((self.max_range-self.min_range)/0.01)):
                    x = robot_pos[0] + r * cos(robot_pos[2] + angle)
                    y = robot_pos[1] + r * sin(robot_pos[2] + angle)
                    
                    # Check if we hit the map boundary first
                    if not self.check_boundaries(x, y):
                        dist = self.add_realistic_noise(r)
                        break
                        
                    i, j = self.world_to_grid(x, y)
                    if 0 <= i < self.grid_size and 0 <= j < self.grid_size and env[i, j] == 1:
                        dist = self.add_realistic_noise(r)
                        break
                
                measurements.append((angle, dist))
                
                # Simulate scan rate timing
                scan_start_time += 1/(2*pi/self.angular_res * self.scan_rate)
            
            return measurements
    
    def add_realistic_noise(self, true_distance):
        """Add noise based on LIDAR specifications"""
        if true_distance < self.close_range_threshold:
            # Close range noise model (<500mm)
            accuracy_noise = random.gauss(0, self.close_range_acc)
            precision_noise = random.gauss(0, self.close_range_prec)
            return max(self.min_range, true_distance + accuracy_noise + precision_noise)
        else:
            # Far range noise model (≥500mm)
            accuracy_noise = random.gauss(0, true_distance * self.far_range_acc)
            precision_noise = random.gauss(0, true_distance * self.far_range_prec)
            return min(self.max_range, true_distance + accuracy_noise + precision_noise)
    
    def visualize(self, env, robot_pos, step):
        """Enhanced visualization with colors for classroom features"""
        self.ax1.clear()
        self.ax2.clear()
        
        # Create colored version of real environment
        colored_env = np.zeros((*env.shape, 3))
        colored_env[env == 0] = [1, 1, 1]        # Free space - white
        colored_env[env == 1] = [0.4, 0.2, 0]    # Wooden desks - brown
        colored_env[env == 2] = [0.8, 0.8, 0.8]  # Whiteboard - light gray
        
        self.ax1.imshow(colored_env, origin='lower')
        self.ax1.set_title(f"Classroom Environment (Step {step})")
        rx, ry = self.world_to_grid(robot_pos[0], robot_pos[1])
        self.ax1.plot(ry, rx, 'ro', markersize=8)
        
        # Occupancy grid (standard grayscale)
        prob_map = 1/(1 + np.exp(-self.logodds_map))
        self.ax2.imshow(prob_map, cmap='binary', vmin=0, vmax=1, origin='lower')
        self.ax2.set_title("Occupancy Grid Map")
        self.ax2.plot(ry, rx, 'ro', markersize=8)
        
        plt.tight_layout()
        plt.pause(0.05)
    
    def run_simulation(self, steps=1000):
        """Main simulation loop"""
        env = self.create_environment()
        robot_pos = np.array([5.0, 9.5, pi]) # x, y, theta
        
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