import rosbag
import numpy as np
import matplotlib.pyplot as plt

x, y = [], []
bag = rosbag.Bag('2025-05-13-16-49-43.bag')

for topic, msg, t in bag.read_messages(topics=['/odom']):
    x.append(msg.pose.pose.position.x)
    y.append(msg.pose.pose.position.y)

plt.plot(x, y, 'r-', label='Robot Path')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Odometry Trajectory')
plt.grid()
plt.show()

bag.close()