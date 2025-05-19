import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.colors as colors

class LiveMapVisualizer:
    def __init__(self, config):
        self.config = config
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111)
        self.resolution = config['map']['resolution']
        self.map_size = np.array(config['map']['size'])
        
        # Setup plot
        self.ax.set_xlim(0, self.map_size[0]/2)
        self.ax.set_ylim(0, self.map_size[1]/2)
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_title('TurtleBot3 Occupancy Grid Mapping')
        self.ax.grid(True)
        
        # Create artists
        self.map_img = self.ax.imshow(
            np.zeros((self.map_size[1]*self.resolution, self.map_size[0]*self.resolution)),
            cmap='binary',
            vmin=0,
            vmax=1,
            origin='lower',
            extent=[-self.map_size[0]/2, self.map_size[0]/2, 
                   -self.map_size[1]/2, self.map_size[1]/2]
        )
        
        self.robot_artist = patches.Circle(
            (0, 0), radius=0.1, color=config['plot']['robot_color'], zorder=2
        )
        self.ax.add_patch(self.robot_artist)
        
        self.lidar_artist = patches.Wedge(
            center=(0, 0), r=0, theta1=0, theta2=0,
            color=config['plot']['lidar_color'], alpha=config['plot']['lidar_alpha'], zorder=1
        )
        self.lidar_rays, = self.ax.plot([], [], color='orange', linewidth=0.5, alpha=0.6)

        self.ax.add_patch(self.lidar_artist)
        
        self.trail_artist, = self.ax.plot([], [], 'b-', linewidth=1, alpha=0.5)
        
        # Add colorbar
        cbar = self.fig.colorbar(self.map_img, ax=self.ax)
        cbar.set_label('Occupancy Probability')
        
        plt.ion()
        plt.show()
        
    
    def update(self, pose, scan, grid_mapper, pose_history=None):
        """Update visualization with new data"""
        # Update map display
        prob_map = grid_mapper.get_probability_map()
        self.map_img.set_data(prob_map.T)
        
        # Update robot position
        self.robot_artist.center = pose[:2]
        
        # Update lidar display
        self.lidar_artist.center = pose[:2]
        self.lidar_artist.r = self.config['laser']['max_range']
        self.lidar_artist.theta1 = np.degrees(pose[2] + self.config['laser']['min_angle'])
        self.lidar_artist.theta2 = np.degrees(pose[2] + self.config['laser']['max_angle'])
        # Compute and plot Lidar rays
        rays_x = []
        rays_y = []

        if scan is not None:
            angle_min = self.config['laser']['min_angle']
            angle_max = self.config['laser']['max_angle']
            angle_increment = (angle_max - angle_min) / len(scan)


            num_readings = len(scan)
            for i in range(num_readings):
                dist = scan[i]
                if 0.01 < dist < self.config['laser']['max_range']:  # Filter invalid
                    angle = angle_min + i * angle_increment
                    global_angle = pose[2] + angle
                    x_end = pose[0] + dist * np.cos(global_angle)
                    y_end = pose[1] + dist * np.sin(global_angle)
                    
                    # Add start and end for each ray as a line segment
                    rays_x += [pose[0], x_end, None]  # None breaks the line for each ray
                    rays_y += [pose[1], y_end, None]

        # Update line data
        self.lidar_rays.set_data(rays_x, rays_y)

        # Update trail if enabled
        if self.config['plot']['show_trail'] and pose_history:
            trail = np.array([p[:2] for p in pose_history])
            self.trail_artist.set_data(trail[:,0], trail[:,1])
        
        # Redraw
        self.fig.canvas.draw_idle()
        plt.pause(self.config['plot']['liveplot_speed'])