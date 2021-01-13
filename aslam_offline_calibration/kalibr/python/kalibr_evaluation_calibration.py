#!/usr/bin/env python
import argparse
import kalibr_common as kc
from mpl_toolkits.mplot3d import art3d, Axes3D, proj3d
import numpy as np
import pylab as pl
import sm
import glob


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='read calibration results from yaml and compare with ground truth')
    parser.add_argument('--reference-sensor', dest='reference_sensor',
                          help='Specify the sensor as the reference coordinate system: camera0 or imu0', required=True)

    parser.add_argument(
        '--cam-ground-truth',
        dest='cam_ground_truth',
        help=
        'the name of yaml file which stores the ground truth of camera extrinsics',
        required=False)
    parser.add_argument(
        '--cam-file-name-prefix',
        dest='cam_file_name_prefix',
        help=
        'the name prefix of yaml file which stores the calibration results of camera extrinsics',
        required=False)
    parser.add_argument(
        '--lidar-ground-truth',
        dest='lidar_ground_truth',
        help=
        'the name of yaml file which stores the ground truth of lidar extrinsics',
        required=False)
    parser.add_argument(
        '--lidar-file-name-prefix',
        dest='lidar_file_name_prefix',
        help=
        'the name prefix of yaml file which stores the calibration results of lidar extrinsics',
        required=False)
    parser.add_argument(
        '--imu-ground-truth',
        dest='imu_ground_truth',
        help=
        'the name of yaml file which stores the ground truth of imu extrinsics',
        required=False)
    parser.add_argument(
        '--imu-file-name-prefix',
        dest='imu_file_name_prefix',
        help=
        'the name prefix of yaml file which stores the calibration results of imu extrinsics',
        required=False)

    parsed_args = parser.parse_args()
    return parsed_args

def calcErrorGTAndEstimation(ext_gt, ext):
    err_T = ext_gt.inverse() * ext
    err_vec = sm.fromTEuler(err_T.T())
    return err_vec

def main():
    parsed_args = parse_arguments()
    if parsed_args.cam_ground_truth and parsed_args.cam_file_name_prefix:
        cam_chain_ext_gt = kc.CameraChainParameters(parsed_args.cam_ground_truth)
        ext_gt_list = []
        num_cam = cam_chain_ext_gt.numCameras()
        for camNr in range(1, num_cam):
            ext_gt_list.append(cam_chain_ext_gt.getExtrinsicsReferenceToCam(camNr))
        err_vec_list_list = [[] for _ in range(num_cam - 1)]
        for file_name in glob.glob(parsed_args.cam_file_name_prefix):
            cam_chain_ext = kc.CameraChainParameters(file_name, parsed_args.reference_sensor)
            for camNr in range(1, num_cam):
                ext = cam_chain_ext.getExtrinsicsReferenceToCam(camNr)
                err_vec = calcErrorGTAndEstimation(ext_gt_list[camNr-1], ext)
                err_vec_list_list[camNr-1].append(err_vec)
        for idx, err_vec_list in enumerate(err_vec_list_list):
            err_mat = np.array(err_vec_list)
            err_mean = np.mean(err_mat, axis=0)
            err_variance = np.var(err_mat, axis=0)
            print ("cam {} extrinsic calibration error".format(idx+1))
            print ("mean of error: ", err_mean)
            print ("variance of error: ", err_variance)

    if parsed_args.lidar_ground_truth and parsed_args.lidar_file_name_prefix:
        lidar_list_ext_gt = kc.LiDARListParameters(parsed_args.lidar_ground_truth, parsed_args.reference_sensor)
        ext_gt_list = []
        num_lidar = lidar_list_ext_gt.numLiDARs()
        for idx in range(0, num_lidar):
            lidar_parameter = lidar_list_ext_gt.getLiDARParameters(idx)
            ext_gt_list.append(lidar_parameter.getExtrinsicsReferenceToHere())
        err_vec_list_list = [[] for _ in range(num_lidar)]
        for file_name in glob.glob(parsed_args.lidar_file_name_prefix):
            lidar_list_ext = kc.LiDARListParameters(file_name, parsed_args.reference_sensor)
            for idx in range(num_lidar):
                lidar_parameter = lidar_list_ext.getLiDARParameters(idx)
                ext = lidar_parameter.getExtrinsicsReferenceToHere()
                err_vec = calcErrorGTAndEstimation(ext_gt_list[idx], ext)
                err_vec_list_list[idx].append(err_vec)
        for idx, err_vec_list in enumerate(err_vec_list_list):
            err_mat = np.array(err_vec_list)
            err_mean = np.mean(err_mat, axis=0)
            err_variance = np.var(err_mat, axis=0)
            print ("LiDAR {} extrinsic calibration error".format(idx))
            print ("mean of error: ", err_mean)
            print ("variance of error: ", err_variance)

    if parsed_args.imu_ground_truth and parsed_args.imu_file_name_prefix:
        imu_list_ext_gt = kc.ImuSetParameters(parsed_args.imu_ground_truth, parsed_args.reference_sensor)
        ext_gt_list = []
        num_imu = imu_list_ext_gt.numImus()
        for idx in range(0, num_imu):
            imu_parameter = imu_list_ext_gt.getImuParameters(idx)
            ext_gt_list.append(imu_parameter.getExtrinsicsReferenceToHere())
        err_vec_list_list = [[] for _ in range(num_imu)]
        for file_name in glob.glob(parsed_args.imu_file_name_prefix):
            imu_list_ext = kc.ImuSetParameters(file_name, parsed_args.reference_sensor)
            for idx in range(num_imu):
                imu_parameter = imu_list_ext.getImuParameters(idx)
                ext = imu_parameter.getExtrinsicsReferenceToHere()
                err_vec = calcErrorGTAndEstimation(ext_gt_list[idx], ext)
                err_vec_list_list[idx].append(err_vec)
        for idx, err_vec_list in enumerate(err_vec_list_list):
            err_mat = np.array(err_vec_list)
            err_mean = np.mean(err_mat, axis=0)
            err_variance = np.var(err_mat, axis=0)
            print ("IMU {} extrinsic calibration error".format(idx))
            print ("mean of error: ", err_mean)
            print ("variance of error: ", err_variance)

if __name__ == "__main__":
    main()
