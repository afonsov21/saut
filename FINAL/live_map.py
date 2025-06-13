# ========== live_map.py ==========
# This module provides a live matplotlib-based visualizer for the occupancy grid,
# robot pose and LIDAR rays.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

class LiveMapVisualizer:
    def __init__(self, config):
        """
        Initialize the visualizer.
        Args:
            config: configuration dictionary
        """
        self.config = config
        self.resolution = config['map']['resolution']

        # Create figure and axis
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('TurtleBot3 Occupancy Grid Mapping')
        self.ax.grid(True)

        self.map_img = None  # Will hold the probability map image

        # Robot position circle
        self.robot_artist = patches.Circle(
            (0, 0), radius=0.1,
            color=config['plot']['robot_color'],
            zorder=2
        )
        self.ax.add_patch(self.robot_artist)

        # LIDAR FOV wedge (semi-circle shaped)
        self.lidar_artist = patches.Wedge(
            center=(0, 0), r=0,
            theta1=0, theta2=0,
            color=config['plot']['lidar_color'],
            alpha=config['plot']['lidar_alpha'],
            zorder=1
        )
        self.ax.add_patch(self.lidar_artist)

        # LIDAR rays (thin orange lines)
        self.lidar_rays, = self.ax.plot([], [], color='orange', linewidth=0.5, alpha=0.6)

        # Enable interactive plotting
        plt.ion()
        plt.show()

    def update(self, pose, scan, grid_mapper,
               min_range, max_range,
               angle_inc, angle_min, angle_max,
               pose_history=None):
        """
        Update the visualization based on latest robot state.
        Args:
            pose: robot pose [x, y, theta]
            scan: LIDAR scan ranges
            grid_mapper: reference to mapping logic (for map & conversion)
            min_range, max_range: LIDAR limits
            angle_inc, angle_min, angle_max: LIDAR angles
            pose_history: list of past robot poses (optional)
        """
        # === Update Map ===
        prob_map = grid_mapper.get_probability_map()
        origin_i, origin_j = grid_mapper.map_origin
        resolution = grid_mapper.resolution

        # Determine image extent in world coordinates
        extent = [
            origin_j * resolution,
            (origin_j + prob_map.shape[1]) * resolution,
            origin_i * resolution,
            (origin_i + prob_map.shape[0]) * resolution
        ]

        if self.map_img is None:
            # Initialize map image
            self.map_img = self.ax.imshow(
                prob_map,
                cmap='binary',
                vmin=0, vmax=1,
                origin='lower',
                extent=extent
            )
        else:
            # Update existing map image
            self.map_img.set_data(prob_map)
            self.map_img.set_extent(extent)

        # === Update Robot Marker ===
        self.robot_artist.center = pose[:2]

        # === Update LIDAR FOV Arc ===
        self.lidar_artist.center = pose[:2]
        self.lidar_artist.r = max_range
        self.lidar_artist.theta1 = np.degrees(pose[2] + angle_min)
        self.lidar_artist.theta2 = np.degrees(pose[2] + angle_max)

        # === LIDAR Rays ===
        rays_x = []
        rays_y = []

        if scan is not None:
            angle = angle_min
            for dist in scan:
                # Clamp scan values
                dist = np.clip(dist, min_range, max_range)

                # Compute global ray endpoint
                angle += angle_inc
                global_angle = pose[2] + angle
                x_end = pose[0] + dist * np.cos(global_angle)
                y_end = pose[1] + dist * np.sin(global_angle)

                # Add line segment (with None to separate each ray)
                rays_x += [pose[0], x_end, None]
                rays_y += [pose[1], y_end, None]

        self.lidar_rays.set_data(rays_x, rays_y)


        # === Refresh Plot ===
        self.fig.canvas.draw_idle()
        plt.pause(self.config['plot']['liveplot_speed'])
