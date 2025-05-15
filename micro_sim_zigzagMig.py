import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi, atan2, sqrt, log, exp
import random

class Config:
    MAP_SIZE = 10
    RESOLUTION = 0.1
    MIN_RANGE = 0.12
    MAX_RANGE = 3.5
    ANGULAR_RES = 1 * (pi / 180)
    BEAM_WIDTH = 1 * (pi / 180)
    CLOSE_RANGE_THRESHOLD = 0.5
    CLOSE_ACC = 0.015
    CLOSE_PREC = 0.010
    FAR_ACC = 0.05
    FAR_PREC = 0.035
    L_OCC = log(0.7 / (1 - 0.7))
    L_FREE = log(0.3 / (1 - 0.3))
    L_0 = log(0.5 / (1 - 0.5))

class OccupancyGrid:
    def __init__(self, config):
        self.config = config
        self.size = int(config.MAP_SIZE / config.RESOLUTION)
        self.logodds = np.zeros((self.size, self.size))

    def update(self, i, j, value):
        if 0 <= i < self.size and 0 <= j < self.size:
            self.logodds[i, j] += value

    def probability_map(self):
        return 1 / (1 + np.exp(-self.logodds))

    def world_to_grid(self, x, y):
        return int(x / self.config.RESOLUTION), int(y / self.config.RESOLUTION)

    def grid_to_world(self, i, j):
        return i * self.config.RESOLUTION, j * self.config.RESOLUTION

class Environment:
    def __init__(self, config):
        self.config = config
        self.grid_size = int(config.MAP_SIZE / config.RESOLUTION)
        self.env = np.zeros((self.grid_size, self.grid_size))
        self.create_classroom()

    def create_classroom(self):
        wall = int(0.2 / self.config.RESOLUTION)
        desk = int(0.6 / self.config.RESOLUTION)
        self.env[:wall, :] = 1
        self.env[-wall:, :] = 1
        self.env[:, :wall] = 1
        self.env[:, -wall:] = 1
        for row in range(5):
            for col in range(5):
                i = int((0.2 + 0.12 * row) * self.grid_size)
                j = int((0.15 + 0.15 * col) * self.grid_size)
                self.env[i:i+desk, j:j+desk] = 1

    def is_obstacle(self, x, y):
        i, j = int(x / self.config.RESOLUTION), int(y / self.config.RESOLUTION)
        if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
            return self.env[i, j] == 1
        return False

class Robot:
    def __init__(self, config):
        self.config = config
        self.pos = np.array([0.5, 0.5, 0.0])
        self.step_size = 0.2
        self.direction = 1  # +1 = right, -1 = left
        self.side_step = self.step_size * 3
        self.last_axis = 'x'
        self.prev_positions = []
        self.stuck_threshold = 10

    def move(self, env, grid):
        current_cell = grid.world_to_grid(self.pos[0], self.pos[1])

        if len(self.prev_positions) >= self.stuck_threshold and all(p == current_cell for p in self.prev_positions):
            self.pos[2] += random.choice([pi/2, -pi/2, pi])
            self.direction *= -1
            self.last_axis = 'x'
            self.prev_positions.clear()
            return

        self.prev_positions.append(current_cell)
        if len(self.prev_positions) > self.stuck_threshold:
            self.prev_positions.pop(0)

        moved = False
        step_attempted = False

        if self.last_axis == 'x':
            new_x = self.pos[0] + self.step_size * self.direction
            new_y = self.pos[1]
            if (0 <= new_x < self.config.MAP_SIZE and not env.is_obstacle(new_x, new_y)):
                self.pos[0] = new_x
                self.pos[1] = new_y
                self.pos[2] = atan2(0, self.direction)
                moved = True
            else:
                self.last_axis = 'y'  # switch to vertical movement next

        if self.last_axis == 'y' and not moved:
            for attempt in [1, -1]:  # try up, then down
                new_y = self.pos[1] + self.side_step * attempt
                new_x = self.pos[0]
                if (0 <= new_y < self.config.MAP_SIZE and not env.is_obstacle(new_x, new_y)):
                    self.pos[0] = new_x
                    self.pos[1] = new_y
                    self.pos[2] = atan2(attempt, 0)
                    self.direction *= -1
                    self.last_axis = 'x'
                    step_attempted = True
                    break

            if not step_attempted:
                forced_y = self.pos[1] + self.side_step
                if 0 <= forced_y < self.config.MAP_SIZE and not env.is_obstacle(self.pos[0], forced_y):
                    self.pos[1] = forced_y
                    self.pos[2] = pi / 2
                self.direction *= -1
                self.last_axis = 'x'

class Sensor:
    def __init__(self, config):
        self.config = config

    def simulate(self, pos, env):
        measurements = []
        for angle in np.arange(0, 2 * pi, self.config.ANGULAR_RES):
            dist = self.config.MAX_RANGE
            for r in np.linspace(self.config.MIN_RANGE, self.config.MAX_RANGE, 100):
                x = pos[0] + r * cos(pos[2] + angle)
                y = pos[1] + r * sin(pos[2] + angle)
                if not (0 <= x < self.config.MAP_SIZE and 0 <= y < self.config.MAP_SIZE):
                    dist = self.noise(r)
                    break
                if env.is_obstacle(x, y):
                    dist = self.noise(r)
                    break
            measurements.append((angle, dist))
        return measurements

    def noise(self, dist):
        if dist < self.config.CLOSE_RANGE_THRESHOLD:
            return max(self.config.MIN_RANGE, dist + random.gauss(0, self.config.CLOSE_ACC + self.config.CLOSE_PREC))
        return min(self.config.MAX_RANGE, dist + random.gauss(0, dist * (self.config.FAR_ACC + self.config.FAR_PREC)))

class Simulator:
    def __init__(self):
        self.config = Config()
        self.grid = OccupancyGrid(self.config)
        self.env = Environment(self.config)
        self.robot = Robot(self.config)
        self.sensor = Sensor(self.config)
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))

    def inverse_model(self, pos, angle, meas, cell):
        dx, dy = cell[0] - pos[0], cell[1] - pos[1]
        r = sqrt(dx**2 + dy**2)
        if r < self.config.MIN_RANGE or r > self.config.MAX_RANGE:
            return 0
        phi = atan2(dy, dx) - pos[2]
        phi = (phi + pi) % (2 * pi) - pi
        if abs(phi - angle) > self.config.BEAM_WIDTH / 2:
            return 0
        if meas < self.config.MAX_RANGE and abs(r - meas) < 0.05:
            return self.config.L_OCC - self.config.L_0
        if r <= meas:
            return self.config.L_FREE - self.config.L_0
        return 0

    def update_map(self, measurements):
        for angle, dist in measurements:
            x0, y0 = self.grid.world_to_grid(self.robot.pos[0], self.robot.pos[1])
            x1 = self.robot.pos[0] + dist * cos(self.robot.pos[2] + angle)
            y1 = self.robot.pos[1] + dist * sin(self.robot.pos[2] + angle)
            x1, y1 = self.grid.world_to_grid(x1, y1)
            cells = self.bresenham(x0, y0, x1, y1)
            for i, j in cells:
                cx, cy = self.grid.grid_to_world(i + 0.5, j + 0.5)
                delta = self.inverse_model(self.robot.pos, angle, dist, (cx, cy))
                self.grid.update(i, j, delta)

    def bresenham(self, x0, y0, x1, y1):
        cells = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = 1 if x0 < x1 else -1, 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            if 0 <= x0 < self.grid.size and 0 <= y0 < self.grid.size:
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

    def visualize(self, step):
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.imshow(self.env.env, origin='lower')
        self.ax1.set_title(f"Environment (Step {step})")
        px, py = self.grid.world_to_grid(self.robot.pos[0], self.robot.pos[1])
        self.ax1.plot(py, px, 'ro')
        self.ax2.imshow(self.grid.probability_map(), cmap='gray', origin='lower', vmin=0, vmax=1)
        self.ax2.set_title("Occupancy Grid")
        self.ax2.plot(py, px, 'ro')
        plt.pause(0.05)

    def run(self, steps=300):
        for step in range(steps):
            measurements = self.sensor.simulate(self.robot.pos, self.env)
            self.update_map(measurements)
            self.robot.move(self.env, self.grid)
            self.visualize(step)
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    sim = Simulator()
    sim.run(steps=1000)
