import rosbag
import matplotlib.pyplot as plt

bag = rosbag.Bag('2025-05-13-16-49-43.bag')

for topic, msg, t in bag.read_messages(topics=['/scan']):
    plt.clf()
    plt.plot(msg.ranges, 'b-', label='LaserScan Ranges')
    plt.ylim(0, max(msg.ranges))  # Adjust based on your sensor range
    plt.xlabel('Scan Index')
    plt.ylabel('Distance (m)')
    plt.title(f'LaserScan at Time: {t.to_sec():.2f}s')
    plt.pause(0.01)

bag.close()