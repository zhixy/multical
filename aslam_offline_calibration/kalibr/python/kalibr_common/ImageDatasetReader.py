import cv_bridge
import cv2
import numpy as np

from DatasetReaderWrapper import BagDatasetReaderWrapper


class ImageDataParser:
    def __init__(self):
        self.uncompress = None
        self.CVB = cv_bridge.CvBridge()

    def parseData(self, data):
        if data._type == 'mv_cameras/ImageSnappyMsg':
            if self.uncompress is None:
                from snappy import uncompress
                self.uncompress = uncompress
            img_data = np.reshape(self.uncompress(np.fromstring(
                data.data, dtype='uint8')), (data.height, data.width),
                order="C")
        elif data._type == 'sensor_msgs/CompressedImage':
            np_arr = np.fromstring(data.data, np.uint8)
            img_data = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        elif data.encoding == "16UC1" or data.encoding == "mono16":
            image_16u = np.array(self.CVB.imgmsg_to_cv2(data))
            img_data = (image_16u / 256).astype("uint8")
        elif data.encoding == "8UC1" or data.encoding == "mono8":
            img_data = np.array(self.CVB.imgmsg_to_cv2(data))
        elif data.encoding == "8UC3" or data.encoding == "bgr8":
            img_data = np.array(self.CVB.imgmsg_to_cv2(data))
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        elif data.encoding == "rgb8":
            img_data = np.array(self.CVB.imgmsg_to_cv2(data))
            img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
        elif data.encoding == "8UC4" or data.encoding == "bgra8":
            img_data = np.array(self.CVB.imgmsg_to_cv2(data))
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2GRAY)
        elif data.encoding == "bayer_rggb8" or data.encoding == "bayer_gbrg8":
            img_data = np.array(self.CVB.imgmsg_to_cv2(data))
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BAYER_BG2GRAY)
        else:
            raise RuntimeError(
                "Unsupported Image format '{}' (Supported are: 16UC1 / mono16,"
                " 8UC1 / mono8, 8UC3 / rgb8 / bgr8, 8UC4 / bgra8, bayer_rggb8"
                " and ImageSnappyMsg)".format(data.encoding));

        return img_data


class BagImageDatasetReader(BagDatasetReaderWrapper):
    def __init__(self, bag_file, topic, bag_from_to=None,
                 perform_synchronization=False):
        image_data_parser = ImageDataParser()
        BagDatasetReaderWrapper.__init__(self, image_data_parser, bag_file,
                                         topic, bag_from_to,
                                         perform_synchronization)
