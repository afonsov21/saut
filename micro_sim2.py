import numpy as np
import pygame
import math
from pygame.locals import *

# Constants
MAP_WIDTH, MAP_HEIGHT = 800, 600  # Real map dimensions
OGM_WIDTH, OGM_HEIGHT = 400, 300  # Occupancy grid display size
GRID_RESOLUTION = 0.1  # meters per cell
Z_MAX = 5.0  # Sensor max range (meters)
L_OCC = 0.9  # Occupied log-odds
L_FREE = -0.7  # Free log-odds

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)  # Obstacles
DARK_GRAY = (50, 50, 50)  # OGM background

class OccupancyGrid:
    def __init__(self, width, height, resolution):
        self.width = int(width / resolution)
        self.height = int(height / resolution)
        self.resolution = resolution
        self.grid = np.zeros((self.height, self.width))  # Log-odds grid
        self.origin_x = self.width // 2
        self.origin_y = self.height // 2

    def world_to_grid(self, x, y):
        grid_x = int(round(x / self.resolution)) + self.origin_x
        grid_y = int(round(y / self.resolution)) + self.origin_y
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        x = (grid_x - self.origin_x) * self.resolution
        y = (grid_y - self.origin_y) * self.resolution
        return x, y

    def inverse_sensor_model(self, cell_x, cell_y, robot_x, robot_y, robot_theta, z_measurements):
        x_i, y_i = self.grid_to_world(cell_x, cell_y)
        r = math.sqrt((x_i - robot_x)**2 + (y_i - robot_y)**2)
        phi = math.atan2(y_i - robot_y, x_i - robot_x) - robot_theta

        # Find closest beam
        min_diff = float('inf')
        z_k = Z_MAX
        for z, angle in z_measurements:
            angle_diff = abs((phi - angle + math.pi) % (2 * math.pi) - math.pi)
            if angle_diff < min_diff:
                min_diff = angle_diff
                z_k = z

        # Apply inverse sensor model
        if r > min(Z_MAX, z_k + 0.1) or min_diff > math.radians(5):
            return 0.0  # No information
        if z_k < Z_MAX and abs(r - z_k) < 0.1:
            return L_OCC  # Occupied
        if r <= z_k:
            return L_FREE  # Free
        return 0.0

    def update(self, robot_x, robot_y, robot_theta, z_measurements):
        for cell_x in range(self.width):
            for cell_y in range(self.height):
                l = self.inverse_sensor_model(cell_x, cell_y, robot_x, robot_y, robot_theta, z_measurements)
                self.grid[cell_y, cell_x] += l

    def get_probability(self, cell_x, cell_y):
        return 1.0 - 1.0 / (1.0 + math.exp(self.grid[cell_y, cell_x]))

class RobotSimulator:
    def __init__(self):
        pygame.init()
        # Main window with map and OGM side by side
        self.screen = pygame.display.set_mode((MAP_WIDTH + OGM_WIDTH, max(MAP_HEIGHT, OGM_HEIGHT)))
        self.clock = pygame.time.Clock()
        self.robot_x, self.robot_y = MAP_WIDTH // 2, MAP_HEIGHT // 2
        self.robot_theta = 0
        self.ogrid = OccupancyGrid(MAP_WIDTH / 100, MAP_HEIGHT / 100, GRID_RESOLUTION)
        self.running = True
        self.obstacles = [
            {"x": 200, "y": 150, "width": 50, "height": 50},  # Example obstacle
            # Add more obstacles as needed
        ]

    def simulate_lidar(self):
        angles = np.linspace(0, 2 * math.pi, 36, endpoint=False)
        measurements = []
        for angle in angles:
            z = Z_MAX
            for obs in self.obstacles:
                ray_end_x = self.robot_x + Z_MAX * 100 * math.cos(angle)
                ray_end_y = self.robot_y + Z_MAX * 100 * math.sin(angle)
                intersect = self.line_rect_intersect(
                    (self.robot_x, self.robot_y), (ray_end_x, ray_end_y),
                    (obs["x"], obs["y"], obs["width"], obs["height"])
                )
                if intersect:
                    dist = math.sqrt((intersect[0]-self.robot_x)**2 + (intersect[1]-self.robot_y)**2)
                    if dist < z * 100:
                        z = dist / 100
            measurements.append((z, angle))
        return measurements

    def line_rect_intersect(self, start, end, rect):
        """Check if line segment intersects with rectangle"""
        x1, y1 = start
        x2, y2 = end
        rx, ry, rw, rh = rect

        def point_in_rect(px, py):
            return (rx <= px <= rx+rw) and (ry <= py <= ry+rh)
        
        if point_in_rect(x1, y1) or point_in_rect(x2, y2):
            return (x1, y1) if point_in_rect(x1, y1) else (x2, y2)

        lines = [
            [(rx, ry), (rx+rw, ry)],    # Top
            [(rx+rw, ry), (rx+rw, ry+rh)], # Right
            [(rx+rw, ry+rh), (rx, ry+rh)], # Bottom
            [(rx, ry+rh), (rx, ry)]     # Left
        ]

        for line in lines:
            intersect = self.line_line_intersect(start, end, line[0], line[1])
            if intersect:
                return intersect
        return None

    def line_line_intersect(self, A, B, C, D):
        """Line segment intersection algorithm"""
        a1 = B[1] - A[1]
        b1 = A[0] - B[0]
        c1 = a1*A[0] + b1*A[1]

        a2 = D[1] - C[1]
        b2 = C[0] - D[0]
        c2 = a2*C[0] + b2*C[1]

        determinant = a1*b2 - a2*b1

        if determinant == 0:
            return None
        else:
            x = (b2*c1 - b1*c2)/determinant
            y = (a1*c2 - a2*c1)/determinant
            if (min(A[0], B[0]) <= x <= max(A[0], B[0]) and \
               min(A[1], B[1]) <= y <= max(A[1], B[1]) and \
               min(C[0], D[0]) <= x <= max(C[0], D[0]) and \
               min(C[1], D[1]) <= y <= max(C[1], D[1])):
                return (x, y)
            return None

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False

            # Move robot
            keys = pygame.key.get_pressed()
            if keys[K_UP]:
                self.robot_y -= 2
            if keys[K_DOWN]:
                self.robot_y += 2
            if keys[K_LEFT]:
                self.robot_x -= 2
            if keys[K_RIGHT]:
                self.robot_x += 2
            if keys[K_a]:
                self.robot_theta -= 0.1
            if keys[K_d]:
                self.robot_theta += 0.1

            # Update map
            z_measurements = self.simulate_lidar()
            self.ogrid.update(self.robot_x / 100, self.robot_y / 100, self.robot_theta, z_measurements)

            # Render
            self.screen.fill(DARK_GRAY)
            
            # Draw real map (left side)
            pygame.draw.rect(self.screen, BLACK, (0, 0, MAP_WIDTH, MAP_HEIGHT))
            self.render_obstacles()
            self.render_robot()
            
            # Draw OGM (right side)
            self.render_ogm(MAP_WIDTH, 0, OGM_WIDTH, OGM_HEIGHT)
            
            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()

    def render_obstacles(self):
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, GRAY, 
                           (obs["x"], obs["y"], obs["width"], obs["height"]))

    def render_robot(self):
        pygame.draw.circle(self.screen, RED, (int(self.robot_x), int(self.robot_y)), 10)
        end_x = self.robot_x + 20 * math.cos(self.robot_theta)
        end_y = self.robot_y + 20 * math.sin(self.robot_theta)
        pygame.draw.line(self.screen, GREEN, (self.robot_x, self.robot_y), (end_x, end_y), 2)

    def render_ogm(self, x, y, width, height):
        """Render the occupancy grid map scaled to the display size"""
        ogm_surface = pygame.Surface((self.ogrid.width, self.ogrid.height))
        for cell_x in range(self.ogrid.width):
            for cell_y in range(self.ogrid.height):
                p = self.ogrid.get_probability(cell_x, cell_y)
                color = int(p * 255)
                ogm_surface.set_at((cell_x, cell_y), (color, color, color))
        
        # Scale and position the OGM
        scaled_ogm = pygame.transform.scale(ogm_surface, (width, height))
        self.screen.blit(scaled_ogm, (x, y))
        
        # Add border
        pygame.draw.rect(self.screen, WHITE, (x, y, width, height), 2)
        
        # Add title
        font = pygame.font.SysFont(None, 24)
        text = font.render("Occupancy Grid Map", True, WHITE)
        self.screen.blit(text, (x + 10, y + 10))

if __name__ == "__main__":
    simulator = RobotSimulator()
    simulator.run()