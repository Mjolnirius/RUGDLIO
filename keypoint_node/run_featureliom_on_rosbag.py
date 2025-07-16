#!/usr/bin/env python3

import sys
import os
import tempfile
from pathlib import Path
import rclpy
from rclpy.serialization import serialize_message, deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import PointCloud2, PointField
from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions
from dfliom_single_pcd import dfliom_single_pcd
import open3d as o3d
import numpy as np
import struct

from rosbag2_py._storage import TopicMetadata

def count_points(m): return m.width * m.height

def get_xyz_from_msg(m):
    xyz = []
    for i in range(m.width):
        offset = i * m.point_step
        x, y, z = struct.unpack_from('fff', m.data, offset=offset)
        xyz.append([x, y, z])
    return np.array(xyz)


def pointcloud2_to_open3d(msg: PointCloud2) -> o3d.geometry.PointCloud:
    """Convert ROS PointCloud2 to Open3D PointCloud (XYZ only)."""
    step = msg.point_step
    data = msg.data
    points = []

    for i in range(msg.width):
        offset = i * step
        x, y, z = struct.unpack_from('fff', data, offset=offset)
        points.append([x, y, z])

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float32))
    return cloud

def open3d_to_pointcloud2(pcd: o3d.geometry.PointCloud, template_msg: PointCloud2) -> PointCloud2:
    """Convert Open3D PointCloud back to ROS PointCloud2 (XYZ only)."""
    import struct
    import numpy as np

    msg = PointCloud2()
    msg.header = template_msg.header
    msg.height = 1
    msg.width = len(pcd.points)
    msg.is_dense = True
    msg.is_bigendian = False
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width

    pts = np.asarray(pcd.points, dtype=np.float32).flatten()
    msg.data = struct.pack('<' + 'f' * len(pts), *pts)

    return msg


def run_featureliom_on_rosbag(
    input_bag_path: str,
    output_bag_path: str,
    topic: str,
    start_idx: int,
    end_idx: int
):
    """
    Modify selected PointCloud2 messages in a ROS2 bag using DFLIOM.
    """
    rclpy.init()

    input_path = Path(input_bag_path).resolve()
    output_path = Path(output_bag_path).resolve()
    
    #output_path = Path(output_bag_path).resolve()
    # Falls Datei/Ordner schon existiert → V1, V2, V3 ... anhängen
    base = output_path.stem
    ext = output_path.suffix
    parent = output_path.parent
    i = 1
    while output_path.exists():
        output_path = parent / f"{base}_V{i}{ext}"
        i += 1


    assert input_path.exists(), f"❌ Input bag not found: {input_path}"

    print(f"📦 Opening input bag: {input_path}")
    print(f"📝 Will write modified bag to: {output_path}")

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(input_path), storage_id='sqlite3'),
        ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    )
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    writer = SequentialWriter()
    writer.open(
        StorageOptions(uri=str(output_path), storage_id='sqlite3'),
        ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    )
    for topic_name, type_str in type_map.items():
        writer.create_topic(TopicMetadata(
            name=topic_name,
            type=type_str,
            serialization_format='cdr'
        ))


    pc_count = 0
    total_pc = 0

    print(f"🔍 Counting PointCloud2 messages in topic '{topic}'...")
    reader_temp = SequentialReader()
    reader_temp.open(
        StorageOptions(uri=str(input_path), storage_id='sqlite3'),
        ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    )
    while reader_temp.has_next():
        tpc, dat, _ = reader_temp.read_next()
        if tpc == topic:
            total_pc += 1


    print(f"📊 Total PointCloud2 messages: {total_pc}")

    # Reset original reader
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(input_path), storage_id='sqlite3'),
        ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    )

    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()

        if topic_name == topic:
            try:
                msg_type = get_message(type_map[topic_name])
                msg = deserialize_message(data, msg_type)

                if start_idx == -1 or (start_idx <= pc_count < end_idx):
                    print(f"🔄 Processing PCD #{pc_count} of {total_pc} through DFLIOM...")
                    pcd_temp = pointcloud2_to_open3d(msg)

                    print(f"[DEBUG] Vor Write: Open3D-PointCloud hat {len(pcd_temp.points)} Punkte")

                    with tempfile.NamedTemporaryFile(suffix=".pcd", delete=False) as tmpfile:
                        o3d.io.write_point_cloud(tmpfile.name, pcd_temp)        # problematisch...
                        # Debug: Was enthält die Datei?
                        loaded = o3d.io.read_point_cloud(tmpfile.name)
                        print(f"[DEBUG] Nach Write & Reload: PCD-Datei enthält {len(loaded.points)} Punkte")

                        reduced_pcd = dfliom_single_pcd(tmpfile.name)           # funktioniert, aber nicht deterministisch
                        print(f"[DEBUG] dfliom_single_pcd() → reduziert auf {len(reduced_pcd.points)} Punkte")

                        reduced_msg = open3d_to_pointcloud2(reduced_pcd, msg)   # msg als Template
                        os.remove(tmpfile.name)

                    # DEBUGGING
                    orig_points = count_points(msg)
                    red_points = count_points(reduced_msg)
                    print(f"   🧾 Points: original={orig_points} vs reduced={red_points}")
                    print(f"   🧾 Frame: {msg.header.frame_id} vs {reduced_msg.header.frame_id}")
                    print(f"   🧾 Fields: {[(f.name, f.offset) for f in msg.fields]} vs {[(f.name, f.offset) for f in reduced_msg.fields]}")
                    print(f"   🧾 Point step: {msg.point_step} vs {reduced_msg.point_step}")
                    print(f"   🧾 Is dense: {msg.is_dense} vs {reduced_msg.is_dense}")

                    writer.write(topic_name, serialize_message(reduced_msg), timestamp)
                else:
                    writer.write(topic_name, data, timestamp)

                pc_count += 1

            except Exception as e:
                print(f"⚠️  Failed to process PointCloud2 message at idx {pc_count}: {e}")
                writer.write(topic_name, data, timestamp)

        else:
            # All other topics → write raw
            writer.write(topic_name, data, timestamp)

    print(f"✅ Done. Saved modified bag to: {output_path}")
    rclpy.shutdown()


# ============================
# 🔧 HIER EINSTELLUNGEN ÄNDERN
# ============================

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("❌ Usage: python run_featureliom_on_rosbag.py <input_bag> <output_bag> <topic> <start_idx> <end_idx>")
        sys.exit(1)

    INPUT_BAG = sys.argv[1]
    OUTPUT_BAG = sys.argv[2]
    TOPIC = sys.argv[3]
    START_IDX = int(sys.argv[4])
    END_IDX = int(sys.argv[5])

    # -1 bedeutet: alles verarbeiten
    if START_IDX == -1:
        START_IDX = 0
    if END_IDX == -1:
        END_IDX = float('inf')

    run_featureliom_on_rosbag(
        input_bag_path=INPUT_BAG,
        output_bag_path=OUTPUT_BAG,
        topic=TOPIC,
        start_idx=START_IDX,
        end_idx=END_IDX
    )