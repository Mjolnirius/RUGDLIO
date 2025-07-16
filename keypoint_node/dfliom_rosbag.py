import rclpy
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata

from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import PointCloud2

from sensor_msgs_py import point_cloud2
import torch
import numpy as np
import struct
import time
import datetime

from model.bimodal_compressor import BimodalCompressor
from model.handcrafted_feature_extractor import compute_smoothness
from model.utils import gridSampling
from ruamel import yaml
import os

class KeypointExtractor:
    def __init__(self, use_model=True, cfg="src/FeatureLIOM/config/bimodal_NCL_Pretrained_Match.yaml", mode="bimodal", lidar_range=128.0):
        self.use_model = use_model
        self.mode = mode
        self.range = lidar_range

        y = yaml.YAML(typ='safe', pure=True)
        self.bimodal_config = y.load(open(cfg, 'r'))

        ckpt_path = "./src/FeatureLIOM/" + self.bimodal_config['ckpt_dir'] + '/' + self.bimodal_config['experiment_name'] + "_best.pth"

        self.bimodal_model = BimodalCompressor(self.bimodal_config)

        if os.path.isfile(ckpt_path) and use_model:
            print(f"Loaded model from {ckpt_path}")
            self.bimodal_model.load_state_dict(torch.load(ckpt_path))

        self.bimodal_model = self.bimodal_model.cuda()
        self.bimodal_model.eval()

        for _ in range(5):
            randinput = torch.rand((1000, 3)).float().cuda()
            _ = self.bimodal_model(randinput, None)

    def msg_to_torch_pcd(self, msg):
        pc = point_cloud2.read_points(msg, skip_nans=True, field_names=["x", "y", "z", "intensity", "t", "ring"], reshape_organized_cloud=True)
        points = np.hstack([pc['x'].reshape(-1, 1), pc['y'].reshape(-1, 1), pc['z'].reshape(-1, 1)])
        rings = pc['ring'].reshape(-1, 1)
        return torch.from_numpy(points).float().cuda(), torch.from_numpy(rings).float().cuda()

    def filter_pointcloud_by_indices(self, msg: PointCloud2, keep_indices: np.ndarray, preserve_structure=False) -> PointCloud2:
        point_step = msg.point_step
        height = msg.height
        width = msg.width

        raw = np.frombuffer(msg.data, dtype=np.uint8).copy()

        if preserve_structure:
            total_points = height * width
            assert len(raw) == total_points * point_step, f"Raw data length mismatch: {len(raw)} != {total_points} * {point_step}"

            mask = np.ones(total_points, dtype=bool)
            mask[keep_indices] = False  # wir nullen *nicht gewollte* Punkte

            for idx in np.where(mask)[0]:
                start = idx * point_step
                raw[start:start + point_step] = np.zeros(point_step, dtype=np.uint8)  # oder z. B. float('nan') schreiben, wenn sinnvoll

            new_data = raw.tobytes()

            msg_out = PointCloud2()
            msg_out.header = msg.header
            msg_out.height = height
            msg_out.width = width
            msg_out.fields = msg.fields
            msg_out.is_bigendian = msg.is_bigendian
            msg_out.point_step = point_step
            msg_out.row_step = width * point_step
            msg_out.is_dense = msg.is_dense
            msg_out.data = new_data

        else:
            # alte Logik – ohne Struktur
            dtype_list = []
            for field in msg.fields:
                if field.datatype == 1: dtype = np.int8
                elif field.datatype == 2: dtype = np.uint8
                elif field.datatype == 3: dtype = np.int16
                elif field.datatype == 4: dtype = np.uint16
                elif field.datatype == 5: dtype = np.int32
                elif field.datatype == 6: dtype = np.uint32
                elif field.datatype == 7: dtype = np.float32
                elif field.datatype == 8: dtype = np.float64
                else: raise ValueError(f"Unsupported datatype: {field.datatype}")
                dtype_list.append((field.name, dtype))

            reshaped = raw.reshape(-1, point_step)
            structured = np.ndarray(shape=(reshaped.shape[0],), dtype=np.dtype(dtype_list), buffer=reshaped.copy().tobytes())
            selected_points = structured[keep_indices]

            msg_out = PointCloud2()
            msg_out.header = msg.header
            msg_out.height = 1
            msg_out.width = len(selected_points)
            msg_out.fields = msg.fields
            msg_out.is_bigendian = msg.is_bigendian
            msg_out.point_step = msg.point_step
            msg_out.row_step = msg_out.width * msg_out.point_step
            msg_out.is_dense = msg.is_dense
            msg_out.data = selected_points.tobytes()

        expected = msg_out.width * msg_out.height * msg_out.point_step
        actual = len(msg_out.data)
        print(f"[DEBUG] width={msg_out.width}, height={msg_out.height}, point_step={msg_out.point_step}, expected={expected}, actual={actual}")
        assert expected == actual, f"Mismatch in msg.data length: expected {expected}, got {actual}"

        return msg_out



    def process_pointcloud_msg(self, msg):
        first_message = not hasattr(self, '_printed_field_info')
        if first_message:
            print("FIELDS IN INPUT MESSAGE:")
            for f in msg.fields:
                print(f"  - {f.name} (offset: {f.offset}, datatype: {f.datatype}, count: {f.count})")

        point_cloud, rings = self.msg_to_torch_pcd(msg)

        with torch.no_grad():
            all_indices = torch.arange(len(point_cloud)).cuda()

            if self.use_model:
                if self.mode == 'handcrafted':
                    smoothness = compute_smoothness(point_cloud, rings, k=10)
                    rough_indices = gridSampling(point_cloud, resolution=0.25)
                    rough_num = len(rough_indices)

                    edge_indices = torch.topk(smoothness, k=int(0.05*rough_num), largest=True).indices.flatten()
                    all_indices = torch.arange(0, len(smoothness)).cuda()
                    planar_indices = ~torch.isin(all_indices, edge_indices).cuda()
                    planar_indices = all_indices[planar_indices].flatten()
                    indices = torch.randperm(len(planar_indices))[:int(0.15*rough_num)]
                    planar_indices = planar_indices[indices]
                    direct_kp_indices = torch.cat([edge_indices, planar_indices])
                else:
                    match_score, gicp_score, _ = self.bimodal_model(point_cloud, None)

                    if self.mode == 'gicp_only':
                        direct_kp_indices = torch.topk(gicp_score, k=int(0.2*len(gicp_score)), largest=False).indices.flatten()
                    elif self.mode == 'match_only':
                        direct_kp_indices = torch.topk(match_score, k=int(0.2*len(match_score)), largest=True).indices.flatten()
                    else:
                        gicp_indices = torch.topk(gicp_score, k=int(0.1*len(gicp_score)), largest=False).indices.flatten()
                        match_score[gicp_indices] = torch.min(match_score)
                        match_indices = torch.topk(match_score, k=int(0.1*len(match_score)), largest=True).indices.flatten()
                        direct_kp_indices = torch.cat([match_indices, gicp_indices])

                    point_norms = torch.norm(point_cloud, dim=1)
                    coverage_indices = (point_norms >= 0.90 * self.range).nonzero().flatten()
                    if len(coverage_indices) >= 0.01 * point_cloud.shape[0]:
                        direct_kp_indices = torch.cat([direct_kp_indices, coverage_indices])

            else:
                direct_kp_indices = torch.randint(low=0, high=len(point_cloud), size=(int(0.2*len(point_cloud)),)).cuda()

            mask = torch.ones(len(point_cloud), dtype=torch.bool, device=point_cloud.device)
            mask[direct_kp_indices] = False
            to_be_compressed = point_cloud[mask]
            to_be_compressed_indices = all_indices[mask]
            sparse_indices = gridSampling(to_be_compressed, resolution=3.5)

            all_indices = torch.hstack([
                direct_kp_indices,
                to_be_compressed_indices[sparse_indices]
            ])

            indices_msg = Int32MultiArray()
            indices_msg.data = all_indices.to(torch.int32).tolist()
            # Int32MultiArray hat kein Header-Feld, daher keine Zeitstempel möglich

            #compressed_msg = self.filter_pointcloud_by_indices(msg, all_indices.cpu().numpy())
            compressed_msg = self.filter_pointcloud_by_indices(msg, all_indices.cpu().numpy(), preserve_structure=True)


            if first_message:
                print("FIELDS IN PROCESSED MESSAGE:")
                for f in compressed_msg.fields:
                    print(f"  - {f.name} (offset: {f.offset}, datatype: {f.datatype}, count: {f.count})")
                self._printed_field_info = True

        return indices_msg, compressed_msg


def process_bag(input_path, output_path, pointcloud_topic):
    rclpy.init()
    extractor = KeypointExtractor()

    reader = SequentialReader()
    reader.open(StorageOptions(uri=input_path, storage_id='sqlite3'),
                ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    all_msgs = []
    cloud_msg_indices = []
    while reader.has_next():
        entry = reader.read_next()
        all_msgs.append(entry)
        if entry[0] == pointcloud_topic:
            cloud_msg_indices.append(len(all_msgs) - 1)

    total_cloud_msgs = len(cloud_msg_indices)

    writer = SequentialWriter()
    writer.open(StorageOptions(uri=output_path, storage_id='sqlite3'),
                ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr'))

    for topic in topic_types:
        writer.create_topic(TopicMetadata(name=topic.name, type=topic.type, serialization_format='cdr'))

    writer.create_topic(TopicMetadata(name="/keypoints", type="std_msgs/msg/Int32MultiArray", serialization_format='cdr'))
    writer.create_topic(TopicMetadata(name="/compressed_cloud", type="sensor_msgs/msg/PointCloud2", serialization_format='cdr'))

    cloud_count = 0
    start_time = time.time()

    for i, (topic, data, t) in enumerate(all_msgs):
        if topic == pointcloud_topic:
            cloud_count += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / cloud_count
            remaining = total_cloud_msgs - cloud_count
            eta = datetime.timedelta(seconds=int(avg_time * remaining))

            print(f"Processing pointcloud message {cloud_count} of {total_cloud_msgs} - ETA: {eta} remaining")

            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)

            indices_msg, compressed_msg = extractor.process_pointcloud_msg(msg)

            original_count = len(point_cloud2.read_points(msg, field_names=['x'], skip_nans=True))
            reduced_count = len(indices_msg.data)

            print(f"Message at timestamp {t} contained {original_count} points, reduced cloud contains {reduced_count} points.")

            writer.write("/keypoints", serialize_message(indices_msg), t)
            writer.write("/compressed_cloud", serialize_message(compressed_msg), t)

        writer.write(topic, data, t)

    rclpy.shutdown()


if __name__ == '__main__':
    import sys
    input_bag = sys.argv[1]
    output_bag = sys.argv[2]
    topic = sys.argv[3]
    process_bag(input_bag, output_bag, topic)
