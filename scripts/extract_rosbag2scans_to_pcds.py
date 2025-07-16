import os
import struct
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

# === PARAMETER ===
bagpath = '/home/thor_unix_2204/RUGDLIO/input_data/dataset_quadhard/'
target_topic = '/os_cloud_node/points'
start_scan = 50
end_scan = 79
output_dir = '/home/thor_unix_2204/RUGDLIO/input_data/dataset_quadhard/quadhard_seq_50_79'

# === PREPARE OUTPUT DIR ===
os.makedirs(output_dir, exist_ok=True)

# === PCD WRITER ===
def write_pcd_dynamic(points, fields, filename):
    with open(filename, 'w') as f:
        f.write('# .PCD v0.7 - Point Cloud Data file\n')
        f.write('VERSION 0.7\n')
        f.write(f'FIELDS {" ".join(fields)}\n')
        f.write(f'SIZE {" ".join(["4"] * len(fields))}\n')
        f.write(f'TYPE {" ".join(["F"] * len(fields))}\n')
        f.write(f'COUNT {" ".join(["1"] * len(fields))}\n')
        f.write(f'WIDTH {points.shape[0]}\n')
        f.write('HEIGHT 1\n')
        f.write('VIEWPOINT 0 0 0 1 0 0 0\n')
        f.write(f'POINTS {points.shape[0]}\n')
        f.write('DATA ascii\n')
        for p in points:
            f.write(" ".join([f"{val}" for val in p]) + "\n")

# === OPEN BAG ===
with Reader(bagpath) as reader:
    print('Topics in bag:', reader.topics)
    count = -1

    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == target_topic:
            count += 1
            if count < start_scan:
                continue
            if count > end_scan:
                break

            # Deserialize PointCloud2 message
            msg = deserialize_cdr(rawdata, connection.msgtype)
            print(f"Processing scan #{count} with frame_id: {msg.header.frame_id}")

            # Get fields & offsets
            field_names = [f.name for f in msg.fields]
            field_offsets = [f.offset for f in msg.fields]

            print(f"Fields: {field_names}")
            print(f"Offsets: {field_offsets}")
            print(f"point_step: {msg.point_step}")

            # Parse all points dynamically
            num_points = int(len(msg.data) / msg.point_step)
            points = []
            for i in range(num_points):
                offset = i * msg.point_step
                values = []
                for field_offset in field_offsets:
                    value = struct.unpack_from('f', msg.data, offset + field_offset)[0]
                    values.append(value)
                points.append(values)

            points = np.array(points, dtype=np.float32)

            # Save as .pcd
            output_file = os.path.join(output_dir, f'scan_{count:04d}.pcd')
            write_pcd_dynamic(points, field_names, output_file)
            print(f"Saved {output_file}")
