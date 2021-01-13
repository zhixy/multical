import numpy as np
from sensor_msgs import point_cloud2

from DatasetReaderWrapper import BagDatasetReaderWrapper


class LiDARDataParser:
    def __init__(self, relative_timestamp):
        self.relative_timestamp = relative_timestamp
        self.field_indices = None

    def parseData(self, data):
        if not self.field_indices:
            self.field_indices = self.findFieldIndices(data.fields)
        points = np.array(point_cloud2.read_points_list(data, skip_nans=True))
        points = points[:, self.field_indices]

        if self.relative_timestamp:
            points[:, 3] += data.header.stamp.to_sec()

        return points

    def findFieldIndices(self, fields):
        field_names = []
        for field in fields:
            field_names.append(field.name)
        # extract fields as the order x, y, z, timestamp, intensity,
        # as there are several potential names for timestamp,
        # we put them all here for matching
        potential_field_names = ['x', 'y', 'z', 'time', 'stamp', 'timestamp',
                                 'time_stamp', 'intensity']
        field_indices = []
        for name in potential_field_names:
            if name in field_names:
                field_indices.append(field_names.index(name))
        if len(field_indices) != 5:
            raise RuntimeError(
                "can not sparse LiDAR data with expected 5 fields")

        return field_indices


class BagLiDARDatasetReader(BagDatasetReaderWrapper):
    def __init__(self, bag_file, topic,
                 relative_timestamp=False, bag_from_to=None,
                 perform_synchronization=False):
        lidar_data_parser = LiDARDataParser(relative_timestamp)
        BagDatasetReaderWrapper.__init__(self, lidar_data_parser, bag_file,
                                         topic, bag_from_to,
                                         perform_synchronization)
