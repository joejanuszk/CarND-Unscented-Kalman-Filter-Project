import matplotlib.pyplot as plt

def normalize_timestamps(timestamps):
    start = timestamps[0]
    normalized = []
    for timestamp in timestamps:
        normalized.append((timestamp - start) / 1.e6)
    return normalized

lines = [];
with open('../build/output.txt', 'r') as output:
    for line in output:
        lines.append(line)

radar_time = []
radar_nis = []
lidar_time = []
lidar_nis = []
for line in lines[1:]:
    parts = line.split()
    timestamp = int(parts[0])
    sensor_type = parts[6]
    nis = float(parts[7])
    if sensor_type == 'radar':
        radar_time.append(timestamp)
        radar_nis.append(nis)
    elif sensor_type == 'lidar':
        lidar_time.append(timestamp)
        lidar_nis.append(nis)


fig, ax_list = plt.subplots(1, 2)

# plot NIS for radar
plt.axes(ax_list[0])
chi_high_3df = 7.815
chi_low_3df = 0.352
radar_time_n = normalize_timestamps(radar_time)
plt.plot(radar_time_n, radar_nis)
plt.plot([radar_time_n[0], radar_time_n[-1]],
         [chi_high_3df, chi_high_3df],
         color='darkred',
         label=str(chi_high_3df))
plt.plot([radar_time_n[0], radar_time_n[-1]],
         [chi_low_3df, chi_low_3df],
         color='orangered',
         label=str(chi_low_3df))
plt.ylim([0, 15])
plt.xlabel('Timestamp (normalized)')
plt.ylabel('NIS')
plt.title('Radar (3 DOF)')
plt.legend()

# plot NIS for lidar
plt.axes(ax_list[1])
chi_high_2df = 5.991
chi_low_2df = 0.103
lidar_time_n = normalize_timestamps(lidar_time)
plt.plot(lidar_time_n, lidar_nis)
plt.plot([lidar_time_n[0], lidar_time_n[-1]],
         [chi_high_2df, chi_high_2df],
         color='darkred',
         label=str(chi_high_2df))
plt.plot([lidar_time_n[0], lidar_time_n[-1]],
         [chi_low_2df, chi_low_2df],
         color='orangered',
         label=str(chi_low_2df))
plt.ylim([0, 15])
plt.xlabel('Timestamp (normalized)')
plt.ylabel('NIS')
plt.title('Lidar (2 DOF)')
plt.legend()

plt.show()
