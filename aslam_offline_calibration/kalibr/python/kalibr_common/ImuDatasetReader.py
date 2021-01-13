import numpy as np
from DatasetReaderWrapper import BagDatasetReaderWrapper


class ImuDataParser:
    def parseData(self, data):
        omega = np.array([data.angular_velocity.x, data.angular_velocity.y,
                          data.angular_velocity.z])
        alpha = np.array(
            [data.linear_acceleration.x, data.linear_acceleration.y,
             data.linear_acceleration.z])
        return omega, alpha


class BagImuDatasetReader(BagDatasetReaderWrapper):
    def __init__(self, bag_file, topic, bag_from_to=None,
                 perform_synchronization=False):
        imu_data_parser = ImuDataParser()
        BagDatasetReaderWrapper.__init__(self, imu_data_parser, bag_file, topic,
                                         bag_from_to, perform_synchronization)
