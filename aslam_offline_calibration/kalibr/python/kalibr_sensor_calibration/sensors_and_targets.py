import sm
import aslam_cv as acv
import aslam_cameras_april as acv_april
import aslam_splines as asp
import aslam_backend as aopt
import bsplines
import kalibr_common as kc
import kalibr_errorterms as ket
import calibrator as ic
from LiDARToSensorCalibration import *
import util as util
from FindTargetFromPointCloud import find_target_pose
import cv2
import sys
import math
import numpy as np
import pylab as pl
import scipy.optimize
from copy import deepcopy
import open3d as o3d
import colorsys
import random
import Queue
# from matplotlib import rc
# # make numpy print prettier
# np.set_printoptions(suppress=True)
# rc('text', usetex=True)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def initLiDARBagDataset(bag_file, topic, relative_timestamp=False, from_to=None):
    print "Initializing LiDAR rosbag dataset reader:"
    print "\tDataset:          {0}".format(bag_file)
    print "\tTopic:            {0}".format(topic)
    #     reader = kc.BagScanDatasetReader(bagfile, topic, bag_from_to=from_to)
    reader = kc.BagLiDARDatasetReader(bag_file, topic,
                                      relative_timestamp=relative_timestamp, bag_from_to=from_to)
    print "\tNumber of messages: {0}".format(len(reader.index))
    return reader


def initCameraBagDataset(bag_file, topic, from_to=None, perform_synchronization=False):
    print "Initializing camera rosbag dataset reader:"
    print "\tDataset:          {0}".format(bag_file)
    print "\tTopic:            {0}".format(topic)
    reader = kc.BagImageDatasetReader(bag_file, topic, bag_from_to=from_to, \
                                      perform_synchronization=perform_synchronization)
    print "\tNumber of images: {0}".format(len(reader.index))
    return reader


def initImuBagDataset(bag_file, topic, from_to=None, perform_synchronization=False):
    print "Initializing imu rosbag dataset reader:"
    print "\tDataset:          {0}".format(bag_file)
    print "\tTopic:            {0}".format(topic)
    reader = kc.BagImuDatasetReader(bag_file, topic, bag_from_to=from_to, \
                                    perform_synchronization=perform_synchronization)
    print "\tNumber of messages: {0}".format(len(reader.index))
    return reader


def initCalibrationTarget(targetConfig, showExtraction=False):
    # load the calibration target configuration
    targetParams = targetConfig.getTargetParams()
    targetType = targetConfig.getTargetType()

    if targetType == 'checkerboard':
        options = acv.CheckerboardOptions()
        options.filterQuads = True
        options.normalizeImage = True
        options.useAdaptiveThreshold = True
        options.performFastCheck = False
        options.windowWidth = 5
        options.showExtractionVideo = showExtraction
        grid = acv.GridCalibrationTargetCheckerboard(targetParams['targetRows'],
                                                     targetParams['targetCols'],
                                                     targetParams['rowSpacingMeters'],
                                                     targetParams['colSpacingMeters'],
                                                     options)
    elif targetType == 'circlegrid':
        options = acv.CirclegridOptions()
        options.showExtractionVideo = showExtraction
        options.useAsymmetricCirclegrid = targetParams['asymmetricGrid']
        grid = acv.GridCalibrationTargetCirclegrid(targetParams['targetRows'],
                                                   targetParams['targetCols'],
                                                   targetParams['spacingMeters'],
                                                   options)
    elif targetType == 'aprilgrid':
        options = acv_april.AprilgridOptions()
        options.showExtractionVideo = showExtraction
        options.minTagsForValidObs = int(np.max([targetParams['tagRows'], targetParams['tagCols']]) + 1)

        grid = acv_april.GridCalibrationTargetAprilgrid(targetParams['tagRows'],
                                                        targetParams['tagCols'],
                                                        targetParams['tagSize'],
                                                        targetParams['tagSpacing'],
                                                        options)

        return grid, targetParams['numberTargets']

    else:
        raise RuntimeError("Unknown calibration target.")
    return grid, 1


class CalibrationTarget(object):
    def __init__(self, grid, initExtrinsic=sm.Transformation()):
        self.grid = grid
        targetPoints = grid.points()
        min = np.min(targetPoints, axis=0).reshape((3, 1))
        max = np.max(targetPoints, axis=0).reshape((3, 1))
        self.range = (min, max)
        self.setInitialGuess(initExtrinsic)

    def setInitialGuess(self, initExtrinsic):
        self.initExtrinsic = initExtrinsic

    def getResultTrafoWorldToTarget(self):
        return self.T_p_w_Dv.toTransformationMatrix()

    def addDesignVariables(self, problem, fixed=False):
        self.C_t_w_Dv = aopt.RotationQuaternionDv(self.initExtrinsic.q())
        self.C_t_w_Dv.setActive(not fixed)
        problem.addDesignVariable(self.C_t_w_Dv, ic.HELPER_GROUP_ID)
        self.t_t_w_Dv = aopt.EuclideanPointDv(self.initExtrinsic.t())
        self.t_t_w_Dv.setActive(not fixed)
        problem.addDesignVariable(self.t_t_w_Dv, ic.HELPER_GROUP_ID)
        self.T_p_w_Dv = aopt.TransformationBasicDv(self.C_t_w_Dv.toExpression(), self.t_t_w_Dv.toExpression())


class LiDAR:
    def __init__(self, config, parsed, targets, distanceSigma=2e-2):
        self.dataset = initLiDARBagDataset(parsed.bagfile[0], config.getRosTopic(),
                                           relative_timestamp=config.getRelativePointTimestamp(),
                                           from_to=parsed.bag_from_to)
        self.planes = targets
        self.targetObs = [self.TargetObservation() for _ in range(len(targets))]
        self.showPointCloud = parsed.showpointcloud
        self.config = config
        self.hasInitializedExtrinsics = config.hasExtrinsics()
        if self.hasInitializedExtrinsics:
            self.init_T_l_b = config.getExtrinsicsReferenceToHere()
        else:
            self.init_T_l_b = sm.Transformation()

        self.distanceUncertainty = distanceSigma
        self.invR = 1. / np.array([self.distanceUncertainty ** 2])

        self.timeOffsetPadding = parsed.timeoffset_padding

        self.loadLiDARDataAndFindTarget(config.getReservedPointsPerFrame())

    class TargetObservation(object):
        def __init__(self):
            self.inliers = None
            self.errorTerms = []


    def loadLiDARDataAndFindTarget(self, reservedPointsPerFrame):
        print "Reading LiDAR data ({0})".format(self.dataset.topic)

        iProgress = sm.Progress2(self.dataset.numMessages())
        iProgress.sample()
        self.targetPoses = []
        reserved_num_points = self.dataset.numMessages()*reservedPointsPerFrame*2
        self.lidarData = np.zeros((reserved_num_points, 4), dtype=float)
        idx = 0
        num_points = 0
        for timestamp, cloud in self.dataset:
            interval = max(1, cloud.shape[0] // reservedPointsPerFrame)
            downsampled_cloud = cloud[::interval, 0:4]
            num = downsampled_cloud.shape[0]
            self.lidarData[num_points:num_points+num] = downsampled_cloud
            num_points += num
            if not self.hasInitializedExtrinsics and idx % 5 == 0:
                targetPose = find_target_pose(cloud, self.showPointCloud)
                if targetPose is not None:
                    self.targetPoses.append(targetPose)

            idx += 1
            iProgress.sample()

        np.resize(self.lidarData, (num_points, 4))
        numFrames = self.dataset.numMessages()
        numPoints = self.lidarData.shape[0]
        numFramesWithTapes = len(self.targetPoses)
        timeSpan = self.lidarData[-1, 3] - self.lidarData[0, 3]

        if numPoints > 100:
            print "\r  Read %d LiDAR readings from %d frames over %.1f seconds, and " \
                  "detect target by tapes from %d frames                    " \
                  % (numPoints, numFrames, timeSpan, numFramesWithTapes)
        else:
            sm.logFatal("Could not find any LiDAR messages. Please check the dataset.")
            sys.exit(-1)

    def transformMeasurementsToWorldFrame(self, poseSplineDv):
        t_min = poseSplineDv.spline().t_min()
        t_max = poseSplineDv.spline().t_max()
        tk = self.lidarData[:, 3] + self.lidarOffsetDv.toScalar()
        indices = np.bitwise_and(tk > t_min, tk < t_max)
        lidarData = self.lidarData[indices, :]
        tk = tk[indices]
        T_b_l = np.linalg.inv(self.T_l_b_Dv.T())
        C_b_l = T_b_l[0:3, 0:3]
        t_b_l = T_b_l[0:3, 3:]
        points = lidarData[:, 0:3].T
        points = C_b_l.dot(points) + t_b_l

        pointsInWorldFrame = []
        for i, time in enumerate(tk):
            T_w_b = poseSplineDv.transformation(time).toTransformationMatrix()
            p_l = np.append(points[:, i], 1.0)
            p_w = np.dot(T_w_b, p_l)
            pointsInWorldFrame.append(p_w[0:3])

        return lidarData, np.asarray(pointsInWorldFrame).T

    def _onPlane(self, plane, points, threshold=0.1):
        min_range = plane.range[0] - np.array([[0], [0], [threshold]])
        max_range = plane.range[1] + np.array([[0], [0], [threshold]])
        C_p_w = plane.C_t_w_Dv.toRotationMatrix()
        t_p_w = plane.t_t_w_Dv.toEuclidean()
        p = np.dot(C_p_w, points) + t_p_w.reshape((3, 1))
        return np.where(np.alltrue(np.logical_and(p > min_range, p < max_range), axis=0))[0]

    def findPointsOnTarget(self, poseSplineDv, threshold=0.1):

        self.lidarData, self.pointCloud = self.transformMeasurementsToWorldFrame(poseSplineDv)
        geometries = []
        interval = 1.0 / len(self.planes)
        for idx, plane in enumerate(self.planes):
            self.targetObs[idx].inliers = self._onPlane(plane, self.pointCloud, threshold)

            if self.showPointCloud:
                min_range = plane.range[0] - np.array([[0], [0], [threshold]])
                max_range = plane.range[1] + np.array([[0], [0], [threshold]])
                center = (min_range + max_range) / 2.0

                T_w_p = np.linalg.inv(plane.T_p_w_Dv.toTransformationMatrix())
                orientation = T_w_p[0:3, 0:3]
                position = T_w_p[0:3, 3:]
                center = np.dot(orientation, center) + position
                extent = max_range - min_range
                boundingBox = o3d.geometry.OrientedBoundingBox(center, orientation, extent)
                boundingBox.color = colorsys.hsv_to_rgb(idx * interval, 1, 1)
                geometries.append(boundingBox)

        if self.showPointCloud:
            coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
            geometries.append(coor)
            util.showPointCloud([self.pointCloud.T], geometries=geometries)

    def addDesignVariables(self, problem, noTimeCalibration=True):
        self.lidarOffsetDv = aopt.Scalar(0.0e-3)
        self.lidarOffsetDv.setActive(not noTimeCalibration)
        problem.addDesignVariable(self.lidarOffsetDv, ic.HELPER_GROUP_ID)

        self.T_l_b_Dv = aopt.TransformationDv(self.init_T_l_b)
        for i in range(0, self.T_l_b_Dv.numDesignVariables()):
            problem.addDesignVariable(self.T_l_b_Dv.getDesignVariable(i), ic.CALIBRATION_GROUP_ID)

    def removeLiDARErrorTerms(self, problem):
        for obs in self.targetObs:
            for error in obs.errorTerms:
                problem.removeErrorTerm(error)

    def addLiDARErrorTerms(self, problem, poseSplineDv):
        for idx, obs in enumerate(self.targetObs):
            obs.errorTerms = []
            plane_normal = aopt.EuclideanExpression(np.array([0.0, 0.0, 1.0]))

            for i in obs.inliers:
                tk = self.lidarOffsetDv.toExpression() + self.lidarData[i, 3]
                T_w_b = poseSplineDv.transformationAtTime(tk, self.timeOffsetPadding, self.timeOffsetPadding)

                point = self.lidarData[i, 0:3]
                distance = np.linalg.norm(point)
                dir_l = point / distance
                T_b_l = self.T_l_b_Dv.toExpression().inverse()
                T_w_l = T_w_b * T_b_l
                T_p_l = self.planes[idx].T_p_w_Dv.toExpression() * T_w_l
                C_p_l = T_p_l.toRotationExpression()
                t_p = T_p_l.toEuclideanExpression()

                d = plane_normal.dot(t_p)
                theta = plane_normal.dot(C_p_l * dir_l)

                predictedMeasurement = d / theta * -1.0

                if predictedMeasurement.toScalar() < 0.:
                    predictedMeasurement = predictedMeasurement * -1.0
                    print "Swapped sign! This should not happen normally!"

                error = ket.ScalarError(distance, self.invR,
                                       predictedMeasurement)

                obs.errorTerms.append(error)
                problem.addErrorTerm(error)

    def filterLiDARErrorTerms(self, problem, threshold_scale_factor):
        for obs in self.targetObs:
            residuals = np.hstack([error_terms.error() for error_terms in obs.errorTerms])
            residual_threshold = threshold_scale_factor * np.std(residuals)
            inliers = np.where(np.fabs(residuals) <= residual_threshold)[0]
            obs.inliers = obs.inliers[inliers]
            for inlier in inliers:
                problem.removeErrorTerm(obs.errorTerms[inlier])

    def getTransformationReferenceToLiDAR(self):
        return sm.Transformation(self.T_l_b_Dv.T())

    def getResultTimeShift(self):
        return self.lidarOffsetDv.toScalar()

    def getLiDARConfig(self):
        self.updateLiDARConfig()
        return self.config

    def updateLiDARConfig(self):
        self.config.setExtrinsicsReferenceToHere(self.getTransformationReferenceToLiDAR())
        self.config.setTimeshiftToReference(self.getResultTimeShift())

# mono camera
class Camera():
    def __init__(self, camConfig, target, dataset,  isReference=False, reprojectionSigma=1.0, showCorners=True, \
                 showReproj=True, showOneStep=False):

        # store the configuration
        self.dataset = dataset
        self.camConfig = camConfig

        # Corner uncertainty
        self.cornerUncertainty = reprojectionSigma

        # set the extrinsic prior to default
        self.init_T_c_b = sm.Transformation()

        # initialize timeshift prior to zero
        self.timeshiftCamToReferencePrior = 0.0

        # initialize the camera data
        self.camera = kc.AslamCamera.fromParameters(camConfig)

        self.target = target
        # extract corners
        self.setupCalibrationTarget(target, showExtraction=showCorners, showReproj=showReproj,
                                    imageStepping=showOneStep)
        multithreading = not (showCorners or showReproj or showOneStep)
        self.targetObservations = kc.extractCornersFromDataset(self.dataset, self.detector,
                                                               multithreading=multithreading)
        if self.targetObservations and type(self.targetObservations[0]) is not list:
            self.targetObservations = [[obs] for obs in self.targetObservations]
        self.isReference = isReference

    def setupCalibrationTarget(self, target, showExtraction=False, showReproj=False, imageStepping=False):
        options = acv.GridDetectorOptions()
        options.imageStepping = imageStepping
        options.plotCornerReprojection = showReproj
        options.filterCornerOutliers = True
        # options.filterCornerSigmaThreshold = 2.0
        # options.filterCornerMinReprojError = 0.2
        if len(target) > 1:
            self.detector = acv_april.MultipleTargetAprilGridDetector(self.camera.geometry, target[0].grid, len(target), options)
        else:
            self.detector = acv.GridDetector(self.camera.geometry, target[0].grid, options)

    def findStaticFrame(self):
        num = len(self.targetObservations)
        cornersPerTarget = self.target[0].grid.size()
        prevObs = {}
        currObs = {}
        staticFrameObversation = []
        for i in range(-1, num-1):
            nextObs = {}
            for obsPerTarget in self.targetObservations[i+1]:
                cornerIdxBase = obsPerTarget.targetId() * cornersPerTarget
                corners = obsPerTarget.getCornersImageFrame()
                cornersIdx = obsPerTarget.getCornersIdx()
                for corner, idx in zip(corners, cornersIdx):
                    nextObs[cornerIdxBase + idx] = corner
            if i > 0:
                opticalFlow = 0.0
                count = 0
                for idx in currObs:
                    if(prevObs.has_key(idx)):
                        opticalFlow += np.linalg.norm(currObs[idx] - prevObs[idx])
                        count += 1
                    if(nextObs.has_key(idx)):
                        opticalFlow += np.linalg.norm(currObs[idx] - nextObs[idx])
                        count += 1
                if count == 0:
                    continue
                meanOpticalFlow = opticalFlow / count
                if meanOpticalFlow < 2.0:
                    staticFrameObversation.append((self.targetObservations[i][0].time().toSec(), self.targetObservations[i]))

            prevObs = currObs
            currObs = nextObs

        return staticFrameObversation

    # estimates the timeshift between the camearas and the imu using a crosscorrelation approach
    #
    # approach: angular rates are constant on a fixed body independent of location
    #          using only the norm of the gyro outputs and assuming that the biases are small
    #          we can estimate the timeshift between the cameras and the imu by calculating
    #          the angular rates of the cameras by fitting a spline and evaluating the derivatives
    #          then computing the cross correlating between the "predicted" angular rates (camera)
    #          and imu, the maximum corresponds to the timeshift...
    #          in a next step we can use the time shift to estimate the rotation between camera and imu
    def findTimeshiftCameraImuPrior(self, imu, verbose=False):
        print "Estimating time shift camera to imu:"

        # fit a spline to the camera observations
        poseSpline = self.initPoseSplineFromCamera(timeOffsetPadding=0.0)

        # predict time shift prior
        t = []
        omega_measured_norm = []
        omega_predicted_norm = []

        for im in imu.imuData:
            tk = im.stamp.toSec()
            if tk > poseSpline.t_min() and tk < poseSpline.t_max():
                # get imu measurements and spline from camera
                omega_measured = im.omega
                omega_predicted = aopt.EuclideanExpression(
                    np.matrix(poseSpline.angularVelocityBodyFrame(tk)).transpose())

                # calc norm
                t = np.hstack((t, tk))
                omega_measured_norm = np.hstack((omega_measured_norm, np.linalg.norm(omega_measured)))
                omega_predicted_norm = np.hstack((omega_predicted_norm, np.linalg.norm(omega_predicted.toEuclidean())))

        if len(omega_predicted_norm) == 0 or len(omega_measured_norm) == 0:
            sm.logFatal("The time ranges of the camera and IMU do not overlap. " \
                        "Please make sure that your sensors are synchronized correctly.")
            sys.exit(-1)

        # get the time shift
        corr = np.correlate(omega_predicted_norm, omega_measured_norm, "full")
        discrete_shift = corr.argmax() - (np.size(omega_measured_norm) - 1)

        # get cont. time shift
        times = [im.stamp.toSec() for im in imu.imuData]
        dT = np.mean(np.diff(times))
        shift = -discrete_shift * dT

        # Create plots
        if verbose:
            pl.plot(t, omega_measured_norm, label="measured_raw")
            pl.plot(t, omega_predicted_norm, label="predicted")
            pl.plot(t - shift, omega_measured_norm, label="measured_corrected")
            pl.legend()
            pl.title("Time shift prior camera-imu estimation")
            pl.figure()
            pl.plot(corr)
            pl.title("Cross-correlation ||omega_predicted||, ||omega_measured||")
            pl.show()
            sm.logDebug("discrete time shift: {0}".format(discrete_shift))
            sm.logDebug("cont. time shift: {0}".format(shift))
            sm.logDebug("dT: {0}".format(dT))

        # store the timeshift (t_imu = t_cam + timeshiftCamToImuPrior)
        self.timeshiftCamToReferencePrior = shift

        print "  Time shift camera to imu (t_imu = t_cam + shift):"
        print self.timeshiftCamToReferencePrior

    # initialize a pose spline using camera poses (pose spline = T_wb)
    def initPoseSplineFromCamera(self, splineOrder=6, poseKnotsPerSecond=100, timeOffsetPadding=0.02):
        pose = bsplines.BSplinePose(splineOrder, sm.RotationVector())

        time_interval_threshold = 3.0 * splineOrder / poseKnotsPerSecond
        print "time interval threshold {0}".format(time_interval_threshold)
        # Get the checkerboard times.
        times = []
        curve = []
        previous_time = None
        for obs in self.targetObservations:
            firstObs = obs[0]
            targetId = firstObs.targetId()
            current_time = firstObs.time().toSec() + self.timeshiftCamToReferencePrior
            if previous_time is None:
                previous_time = current_time
            elif (current_time - previous_time) > time_interval_threshold:
                print "The data gathering will break because of too large time interval ({0}s)".format(current_time - previous_time)
                print " Time span of gathered data is {0}s".format(times[-1] - times[0])
                break
            else:
                previous_time = current_time

            times.append(current_time)
            T_w_t = self.target[targetId].initExtrinsic.inverse().T()
            T_w_c = np.dot(T_w_t, firstObs.T_t_c().T())
            curve.append(pose.transformationToCurveValue(T_w_c))

        times = np.array(times)
        curve = np.matrix(curve).T

        if np.isnan(curve).any():
            raise RuntimeError("Nans in curve values")
            sys.exit(0)

        # Add 2 seconds on either end to allow the spline to slide during optimization
        times = np.hstack((times[0] - (timeOffsetPadding * 2.0), times, times[-1] + (timeOffsetPadding * 2.0)))
        curve = np.hstack((curve[:, 0], curve, curve[:, -1]))

        # Make sure the rotation vector doesn't flip
        for i in range(1, curve.shape[1]):
            previousRotationVector = curve[3:6, i - 1]
            r = curve[3:6, i]
            angle = np.linalg.norm(r)
            axis = r / angle
            best_r = r
            best_dist = np.linalg.norm(best_r - previousRotationVector)

            for s in range(-3, 4):
                aa = axis * (angle + math.pi * 2.0 * s)
                dist = np.linalg.norm(aa - previousRotationVector)
                if dist < best_dist:
                    best_r = aa
                    best_dist = dist
            curve[3:6, i] = best_r;

        seconds = times[-1] - times[0]
        knots = int(round(seconds * poseKnotsPerSecond))

        print
        print "Initializing a pose spline with %d knots (%f knots per second over %f seconds)" % (
        knots, poseKnotsPerSecond, seconds)
        pose.initPoseSplineSparse(times, curve, knots, 1e-4)
        return pose

    def addDesignVariables(self, problem, noTimeCalibration=True,
                           baselinedv_group_id=ic.HELPER_GROUP_ID):
        # Add the calibration design variables.
        active = not self.isReference
        self.T_c_b_Dv = aopt.TransformationDv(self.init_T_c_b, rotationActive=active, translationActive=active)
        for i in range(0, self.T_c_b_Dv.numDesignVariables()):
            problem.addDesignVariable(self.T_c_b_Dv.getDesignVariable(i), baselinedv_group_id)

        # Add the time delay design variable.
        self.cameraTimeToReferenceTimeDv = aopt.Scalar(0.0)
        self.cameraTimeToReferenceTimeDv.setActive(not noTimeCalibration)
        problem.addDesignVariable(self.cameraTimeToReferenceTimeDv, ic.CALIBRATION_GROUP_ID)

    def addCameraErrorTerms(self, problem, poseSplineDv, blakeZissermanDf=0.0, timeOffsetPadding=0.0):
        print
        print "Adding camera error terms ({0})".format(self.dataset.topic)

        # progress bar
        iProgress = sm.Progress2(len(self.targetObservations))
        iProgress.sample()

        allReprojectionErrors = list()
        error_t = self.camera.reprojectionErrorType

        T_c_b = self.T_c_b_Dv.toExpression()
        for obs in self.targetObservations:
            for obsPerTarget in obs:
                # Build a transformation expression for the time.
                frameTime = self.cameraTimeToReferenceTimeDv.toExpression() + obsPerTarget.time().toSec() + self.timeshiftCamToReferencePrior
                frameTimeScalar = frameTime.toScalar()

                # as we are applying an initial time shift outside the optimization so
                # we need to make sure that we dont add data outside the spline definition
                if frameTimeScalar <= poseSplineDv.spline().t_min() or frameTimeScalar >= poseSplineDv.spline().t_max():
                    continue

                T_w_b = poseSplineDv.transformationAtTime(frameTime, timeOffsetPadding, timeOffsetPadding)
                T_b_w = T_w_b.inverse()

                # calibration target coords to camera N coords
                # T_b_w: from world to imu coords
                # T_cN_b: from imu to camera N coords
                T_w_p = self.target[obsPerTarget.targetId()].T_p_w_Dv.toExpression().inverse()
                T_c_p = T_c_b * T_b_w * T_w_p

                # get the image and target points corresponding to the frame
                imageCornerPoints = np.array(obsPerTarget.getCornersImageFrame()).T
                targetCornerPoints = np.array(obsPerTarget.getCornersTargetFrame()).T

                # setup an aslam frame (handles the distortion)
                frame = self.camera.frameType()
                frame.setGeometry(self.camera.geometry)

                # corner uncertainty
                R = np.eye(2) * self.cornerUncertainty * self.cornerUncertainty
                invR = np.linalg.inv(R)

                for pidx in range(0, imageCornerPoints.shape[1]):
                    # add all image points
                    k = self.camera.keypointType()
                    k.setMeasurement(imageCornerPoints[:, pidx])
                    k.setInverseMeasurementCovariance(invR)
                    frame.addKeypoint(k)

                reprojectionErrors = list()
                for pidx in range(0, imageCornerPoints.shape[1]):
                    # add all target points
                    targetPoint = np.insert(targetCornerPoints.transpose()[pidx], 3, 1)
                    p = T_c_p * aopt.HomogeneousExpression(targetPoint)

                    # build and append the error term
                    rerr = error_t(frame, pidx, p)

                    # add blake-zisserman m-estimator
                    if blakeZissermanDf > 0.0:
                        mest = aopt.BlakeZissermanMEstimator(blakeZissermanDf)
                        rerr.setMEstimatorPolicy(mest)

                    problem.addErrorTerm(rerr)
                    reprojectionErrors.append(rerr)

                allReprojectionErrors.append(reprojectionErrors)

            # update progress bar
            iProgress.sample()

        print "\r  Added {0} camera error terms                      ".format(len(self.targetObservations))
        self.allReprojectionErrors = allReprojectionErrors


# pair of cameras with overlapping field of view (perfectly synced cams required!!)
#
#     Sensor "chain"                    R_C1C0 source: *fixed as input from stereo calib
#                                                      *optimized using stereo error terms
#         R_C1C0(R,t)   C1   R_C2C1(R,t)    C2         Cn
# C0  o------------------o------------------o    ...    o 
#     |
#     | R_C0I (R,t)
#     |
#     o (IMU)
#
# imu is need to initialize an orientation prior between imu and camera chain
class CameraChain():
    def __init__(self, chainConfig, target, parsed, isReference=False):

        # create all camera in the chain
        self.camList = []
        for camNr in range(0, chainConfig.numCameras()):
            camConfig = chainConfig.getCameraParameters(camNr)
            dataset = initCameraBagDataset(parsed.bagfile[0], camConfig.getRosTopic(), \
                                           parsed.bag_from_to, parsed.perform_synchronization)

            # create the camera
            self.camList.append(Camera(camConfig,
                                       target,
                                       dataset,
                                       isReference=camNr is 0 and isReference,
                                       # Ultimately, this should come from the camera yaml.
                                       reprojectionSigma=parsed.reprojection_sigma,
                                       showCorners=parsed.showextraction,
                                       showReproj=parsed.showextraction,
                                       showOneStep=parsed.extractionstepping))

        self.chainConfig = chainConfig
        self.target = target
        # find and store time between first and last image over all cameras
        # self.findCameraTimespan()

        self.has_initialized = self.readBaselinesFromFile()

    def readBaselinesFromFile(self):
        for camNr in range(1, len(self.camList)):
            try:
                T_camN_camNMinus1 = self.chainConfig.getExtrinsicsLastCamToHere(camNr)
            except:
                print "No camera extrinsics are provide in config File"
                return False

            self.camList[camNr].init_T_c_b = T_camN_camNMinus1 * self.camList[camNr-1].init_T_c_b
            print "Baseline between cam{0} and cam{1} set to:".format(camNr - 1, camNr)
            print "T= ", T_camN_camNMinus1.T()
            print "Baseline: ", np.linalg.norm(T_camN_camNMinus1.t()), " [m]"

        return True

    # initialize a pose spline for the chain
    def initializePoseSplineFromCameraChain(self, splineOrder=6, poseKnotsPerSecond=100, timeOffsetPadding=0.02):

        pose = bsplines.BSplinePose(splineOrder, sm.RotationVector())

        # Get the checkerboard times.
        times = []
        curve = []
        for camNr, cam in enumerate(self.camList):
            # from imu coords to camerea N coords (as DVs)
            T_cN_b = cam.init_T_c_b.T()
            for obs in cam.targetObservations:
                for obsPerTarget in obs:
                    targetId = obsPerTarget.targetId()
                    times.append(obsPerTarget.time().toSec() + cam.timeshiftCamToReferencePrior)
                    T_w_t = self.target[targetId].initExtrinsic.inverse().T()
                    T_w_b = np.dot(T_w_t, np.dot(obsPerTarget.T_t_c().T(), T_cN_b))
                    curve.append(pose.transformationToCurveValue(T_w_b))

        sorted_indices = np.argsort(times)
        times = np.array(times)
        curve = np.matrix(curve).T
        times = times[sorted_indices]
        curve = curve[:, sorted_indices]

        if np.isnan(curve).any():
            raise RuntimeError("Nans in curve values")
            sys.exit(0)

        # Add 2 seconds on either end to allow the spline to slide during optimization
        times = np.hstack((times[0] - (timeOffsetPadding * 2.0), times, times[-1] + (timeOffsetPadding * 2.0)))
        curve = np.hstack((curve[:, 0], curve, curve[:, -1]))

        # Make sure the rotation vector doesn't flip
        for i in range(1, curve.shape[1]):
            previousRotationVector = curve[3:6, i - 1]
            r = curve[3:6, i]
            angle = np.linalg.norm(r)
            axis = r / angle
            best_r = r
            best_dist = np.linalg.norm(best_r - previousRotationVector)

            for s in range(-3, 4):
                aa = axis * (angle + math.pi * 2.0 * s)
                dist = np.linalg.norm(aa - previousRotationVector)
                if dist < best_dist:
                    best_r = aa
                    best_dist = dist
            curve[3:6, i] = best_r;

        seconds = times[-1] - times[0]
        knots = int(round(seconds * poseKnotsPerSecond))

        print
        print "Initializing a pose spline with %d knots (%f knots per second over %f seconds)" % (
            knots, poseKnotsPerSecond, seconds)
        pose.initPoseSplineSparse(times, curve, knots, 1e-4)
        return pose

    # find the timestamp for the first and last image considering all cameras in the chain
    def findCameraTimespan(self):
        tStart = acv.Time(0.0)
        tEnd = acv.Time(0.0)

        for cam in self.camList:
            if len(cam.targetObservations) > 0:
                tStartCam = cam.targetObservations[0][0].time()
                tEndCam = cam.targetObservations[-1][0].time()

                if tStart.toSec() > tStartCam.toSec():
                    tStart = tStartCam

                if tEndCam.toSec() > tEnd.toSec():
                    tEnd = tEndCam

        self.timeStart = tStart
        self.timeEnd = tEnd

    # pose graph optimization to get initial guess of calibration target
    def findTargetPoseInWorld(self, targets):
        if len(targets) == 1:
            return
        targetTransformations = {}
        adjacentList = [[] for _ in range(len(targets))]
        for cam in self.camList:
            for observation in cam.targetObservations:
                observedTargetsNumber = len(observation)
                if observedTargetsNumber < 2:
                    continue
                for i in range(0, observedTargetsNumber-1):
                    for j in range(i+1, observedTargetsNumber):
                        targetIdx1 = observation[i].targetId()
                        targetIdx2 = observation[j].targetId()
                        T_ti_tj_mea = observation[i].T_t_c() * observation[j].T_t_c().inverse()
                        key = (targetIdx1, targetIdx2)
                        if key not in targetTransformations:
                            targetTransformations[key] = []
                        targetTransformations[key].append(T_ti_tj_mea)
                        if targetIdx2 not in adjacentList[targetIdx1]:
                            adjacentList[targetIdx1].append(targetIdx2)

        initialGuess = [None] * len(targets)
        initialGuess[0] = sm.Transformation()
        q = Queue.Queue()
        q.put(0)
        while not q.empty():
            idx = q.get()
            for neighbour in adjacentList[idx]:
                if initialGuess[neighbour] is None:
                    q.put(neighbour)
                    key = (idx, neighbour)
                    initialGuess[neighbour] = targetTransformations[key][0].inverse() * initialGuess[idx]

        # build the problem
        problem = aopt.OptimizationProblem()
        T_t_w_Dv = []
        for i in range(len(targets)):
            if initialGuess[i] is None:
                raise RuntimeError("Target {0} is not observed simultaneously with other target!".format(i))
            isActive = i is not 0
            T_t_w_Dv.append(aopt.TransformationDv(initialGuess[i], rotationActive=isActive, translationActive=isActive))
            for j in range(0, T_t_w_Dv[i].numDesignVariables()):
                problem.addDesignVariable(T_t_w_Dv[i].getDesignVariable(j))

        for key, transformations in targetTransformations.items():
            T_ti_tj_pre = T_t_w_Dv[key[0]].toExpression() * \
                          T_t_w_Dv[key[1]].toExpression().inverse()
            for transformation in transformations:
                error = aopt.ErrorTermTransformation(T_ti_tj_pre, transformation, 1.0, 0.1)
                problem.addErrorTerm(error)

        # define the optimization
        options = aopt.Optimizer2Options()
        options.verbose = True
        options.linearSolver = aopt.BlockCholeskyLinearSystemSolver()
        options.nThreads = 2
        options.convergenceDeltaX = 1e-4
        options.convergenceJDescentRatioThreshold = 1e-5
        options.maxIterations = 50

        # run the optimization
        optimizer = aopt.Optimizer2(options)
        optimizer.setProblem(problem)

        # get the prior
        try:
            optimizer.optimize()
        except:
            sm.logFatal("Failed to obtain target pose prior!")
            sys.exit(-1)
        for i in range(len(targets)):
            targets[i].setInitialGuess(sm.Transformation(T_t_w_Dv[i].T()))
            print "Transformation prior calibration world to target {0} found as: (T_t{0}_t0)".format(i)
            print T_t_w_Dv[i].T()

    #
    def findExtrinsicPriorSensorsToCamera(self, imu=None, lidar_list=[]):

        print
        print "Estimating initial extrinsic parameters between primary camera and all other sensors"

        # build the problem
        problem = aopt.OptimizationProblem()

        pose_spline = self.camList[0].initPoseSplineFromCamera(6, 50, 0.0)

        if imu:
            # Add the rotation as design variable
            q_i_c_Dv = aopt.RotationQuaternionDv(np.array([0., 0., 0., 1.]))
            q_i_c_Dv.setActive(True)
            problem.addDesignVariable(q_i_c_Dv)

            # Add the gyro bias as design variable
            gyroBiasDv = aopt.EuclideanPointDv(np.zeros(3))
            gyroBiasDv.setActive(True)
            problem.addDesignVariable(gyroBiasDv)
            # DV expressions
            R_i_c = q_i_c_Dv.toExpression()
            bias = gyroBiasDv.toExpression()

            for im in imu.imuData:
                tk = im.stamp.toSec()
                if pose_spline.t_min() < tk < pose_spline.t_max():

                    # get the vision predicted omega and measured omega (IMU)
                    omega_predicted = R_i_c * aopt.EuclideanExpression(
                        np.matrix(pose_spline.angularVelocityBodyFrame(tk)).transpose())
                    omega_measured = im.omega

                    # error term
                    gerr = ket.GyroscopeError(omega_measured, im.omegaInvR, omega_predicted, bias)
                    problem.addErrorTerm(gerr)


        if not self.has_initialized:
            # Add Other cameras
            T_ci_c_Dv = []
            for i in range(len(self.camList) - 1):
                T_ci_c_Dv.append(aopt.TransformationDv(sm.Transformation()))
                for j in range(0, T_ci_c_Dv[i].numDesignVariables()):
                    problem.addDesignVariable(T_ci_c_Dv[i].getDesignVariable(j))

                cam = self.camList[i+1]
                for obs in cam.targetObservations:
                    obsTime = obs[0].time().toSec()
                    if pose_spline.t_min() < obsTime < pose_spline.t_max():
                        T_w_c = sm.Transformation(pose_spline.transformation(obsTime))
                        for obsPerTarget in obs:
                            T_ci_t = obsPerTarget.T_t_c().inverse()
                            targetId = obsPerTarget.targetId()
                            T_ci_w = T_ci_t * self.target[targetId].initExtrinsic
                            mea_T_ci_c = T_ci_w * T_w_c
                            error = aopt.ErrorTermTransformation(T_ci_c_Dv[i].toExpression(), mea_T_ci_c, 1.0, 0.1)
                            problem.addErrorTerm(error)

        # Add LiDARs
        T_l_c_Dv_list = []
        for lidar in lidar_list:
            if lidar.hasInitializedExtrinsics:
                continue
            T_l_c_Dv = aopt.TransformationDv(sm.Transformation())
            T_l_c_Dv_list.append(T_l_c_Dv)
            for j in range(0, T_l_c_Dv.numDesignVariables()):
                problem.addDesignVariable(T_l_c_Dv.getDesignVariable(j))

            for position, orientation, time in lidar.targetPoses:
                if pose_spline.t_min() < time < pose_spline.t_max():
                    T_w_c = sm.Transformation(pose_spline.transformation(time))
                    mea_T_l_w = sm.Transformation(sm.r2quat(orientation), position)
                    mea_T_l_c = mea_T_l_w * T_w_c
                    error = aopt.ErrorTermTransformation(T_l_c_Dv.toExpression(), mea_T_l_c, 1.0, 0.1)
                    problem.addErrorTerm(error)

        if problem.numErrorTerms() == 0:
            print "No initial extrinsic parameter is waited to estimate"
            return

        # define the optimization
        options = aopt.Optimizer2Options()
        options.verbose = False
        options.linearSolver = aopt.BlockCholeskyLinearSystemSolver()
        options.nThreads = 2
        options.convergenceDeltaX = 1e-4
        options.convergenceJDescentRatioThreshold = 1e-5
        options.maxIterations = 50

        # run the optimization
        optimizer = aopt.Optimizer2(options)
        optimizer.setProblem(problem)

        # get the prior
        try:
            optimizer.optimize()
        except:
            sm.logFatal("Failed to obtain extrinsic parameters of sensors!")
            sys.exit(-1)

        if imu:
            if imu.isReference:
                R_c_b = q_i_c_Dv.toRotationMatrix().transpose()
                self.camList[0].init_T_c_b = sm.Transformation(sm.rt2Transform(R_c_b, self.camList[0].init_T_c_b.t()))
            else:
                R_i_b = q_i_c_Dv.toRotationMatrix()
                imu.init_q_i_b = sm.r2quat(R_i_b)

            print "  Orientation prior camera-imu found as: (T_i_c)"
            print q_i_c_Dv.toRotationMatrix()

        if not self.has_initialized:
            # Add cameras' extrinsics
            for i in range(len(self.camList) - 1):
                self.camList[i+1].init_T_c_b = sm.Transformation(T_ci_c_Dv[i].T()) * \
                                                    self.camList[0].init_T_c_b

                print "Transformation from body to cam{0} set to:".format(i+1)
                print "T= ", self.camList[i+1].init_T_c_b.T()

        idx_T_l_c_Dv = 0
        for idx, lidar in enumerate(lidar_list):
            if not lidar.hasInitializedExtrinsics:
                lidar.init_T_l_b = sm.Transformation(T_l_c_Dv_list[idx_T_l_c_Dv].T()) * self.camList[0].init_T_c_b
                idx_T_l_c_Dv += 1
                lidar.hasInitializedExtrinsics = True
            print "Transformation from reference sensor to LiDAR{0} set to:".format(idx)
            print "T= ", lidar.init_T_l_b.T()

        if imu:
            # estimate gravity in the world coordinate frame as the mean specific force
            R_c_i = q_i_c_Dv.toRotationMatrix().transpose()
            a_w = []
            for im in imu.imuData:
                tk = im.stamp.toSec()
                if pose_spline.t_min() < tk < pose_spline.t_max():
                    a_w.append(np.dot(pose_spline.orientation(tk), np.dot(R_c_i, - im.alpha)))
            mean_a_w = np.mean(np.asarray(a_w).T, axis=1)
            gravity_w = mean_a_w / np.linalg.norm(mean_a_w) * 9.80655
            print "Gravity was intialized to", gravity_w, "[m/s^2]"

            # set the gyro bias prior (if we have more than 1 cameras use recursive average)
            b_gyro = gyroBiasDv.toExpression().toEuclidean()
            imu.GyroBiasPriorCount += 1
            imu.GyroBiasPrior = (imu.GyroBiasPriorCount - 1.0) / imu.GyroBiasPriorCount * imu.GyroBiasPrior + 1.0 / imu.GyroBiasPriorCount * b_gyro

            # print result
            print "  Gyro bias prior found as: (b_gyro)"
            print b_gyro

            return gravity_w

    # return the baseline transformation from camA to camB
    def getResultBaseline(self, fromCamANr, toCamBNr):

        T_cB_cA = sm.Transformation(self.camList[toCamBNr].T_c_b_Dv.T()).inverse() * \
                  sm.Transformation(self.camList[fromCamANr].T_c_b_Dv.T())
        # calculate the metric baseline
        baseline = np.linalg.norm(T_cB_cA.t())

        return T_cB_cA, baseline

    def getTransformationReferenceToCam(self, camNr):
        return sm.Transformation(self.camList[camNr].T_c_b_Dv.T())


    def getResultTimeShift(self, camNr):
        return self.camList[camNr].cameraTimeToReferenceTimeDv.toScalar() + self.camList[camNr].timeshiftCamToReferencePrior

    def addDesignVariables(self, problem, noTimeCalibration=True):
        # add the design variables (T(R,t) & time)  for all indevidual cameras
        for camNr, cam in enumerate(self.camList):
            # the first "baseline" dv is between the imu and cam0
            if camNr == 0:
                baselinedv_group_id = ic.CALIBRATION_GROUP_ID
            else:
                baselinedv_group_id = ic.HELPER_GROUP_ID
            cam.addDesignVariables(problem, noTimeCalibration, baselinedv_group_id=baselinedv_group_id)

    # add the reprojection error terms for all cameras in the chain
    def addCameraChainErrorTerms(self, problem, poseSplineDv, blakeZissermanDf=-1, timeOffsetPadding=0.0):
        # add the error terms for all cameras
        for cam in self.camList:
            cam.addCameraErrorTerms(problem, poseSplineDv, blakeZissermanDf, timeOffsetPadding)


# IMU
class Imu(object):
    def __init__(self, imuConfig, parsed, isReference=True, estimateTimedelay=True):

        # determine whether IMU coincides with body frame (for multi-IMU setups)
        self.isReference = isReference
        self.estimateTimedelay = estimateTimedelay

        # store input
        self.imuConfig = imuConfig

        # load dataset
        self.dataset = initImuBagDataset(parsed.bagfile[0], imuConfig.getRosTopic(), \
                                         parsed.bag_from_to, parsed.perform_synchronization)

        # statistics
        self.accelUncertaintyDiscrete, self.accelRandomWalk, self.accelUncertainty = self.imuConfig.getAccelerometerStatistics()
        self.gyroUncertaintyDiscrete, self.gyroRandomWalk, self.gyroUncertainty = self.imuConfig.getGyroStatistics()

        # init GyroBiasPrior (+ count for recursive averaging if we have more than 1 measurement = >1 cameras)
        self.GyroBiasPrior = np.array([0, 0, 0])
        self.GyroBiasPriorCount = 0

        # load the imu dataset
        self.loadImuData()

        # initial estimates for multi IMU calibration
        self.init_q_i_b = np.array([0., 0., 0., 1.])
        self.timeOffset = 0.0

        self.staticBias = parsed.static_bias

    # omega -- angular_velocity
    # alpha -- linear_acceleration
    class ImuMeasurement(object):
        def __init__(self, stamp, omega, alpha, Rgyro, Raccel):
            self.omega = omega
            self.alpha = alpha
            self.omegaR = Rgyro
            self.omegaInvR = np.linalg.inv(Rgyro)
            self.alphaR = Raccel
            self.alphaInvR = np.linalg.inv(Raccel)
            self.stamp = stamp

    def loadImuData(self):
        print "Reading IMU data ({0})".format(self.dataset.topic)

        # prepare progess bar
        iProgress = sm.Progress2(self.dataset.numMessages())
        iProgress.sample()

        Rgyro = np.eye(3) * self.gyroUncertaintyDiscrete * self.gyroUncertaintyDiscrete
        Raccel = np.eye(3) * self.accelUncertaintyDiscrete * self.accelUncertaintyDiscrete

        # Now read the imu measurements.
        # omega -- angular_velocity
        # alpha -- linear_acceleration
        imu = []
        for timestamp, (omega, alpha) in self.dataset:
            timestamp = acv.Time(timestamp.toSec())
            imu.append(self.ImuMeasurement(timestamp, omega, alpha, Rgyro, Raccel))
            iProgress.sample()

        self.imuData = imu

        if len(self.imuData) > 1:
            print "\r  Read %d imu readings over %.1f seconds                   " \
                  % (len(imu), imu[-1].stamp.toSec() - imu[0].stamp.toSec())
        else:
            sm.logFatal("Could not find any IMU messages. Please check the dataset.")
            sys.exit(-1)

    def addDesignVariables(self, problem):
        if self.staticBias:
            self.gyroBiasDv = aopt.EuclideanPointDv(self.GyroBiasPrior)
            self.gyroBiasDv.setActive(True)
            problem.addDesignVariable(self.gyroBiasDv, ic.HELPER_GROUP_ID)

            self.accelBiasDv = aopt.EuclideanPointDv(np.zeros(3))
            self.accelBiasDv.setActive(True)
            problem.addDesignVariable(self.accelBiasDv, ic.HELPER_GROUP_ID)
        else:
            # create design variables
            self.gyroBiasDv = asp.EuclideanBSplineDesignVariable(self.gyroBias)
            self.accelBiasDv = asp.EuclideanBSplineDesignVariable(self.accelBias)

            ic.addSplineDesignVariables(problem, self.gyroBiasDv, setActive=True, \
                                        group_id=ic.HELPER_GROUP_ID)
            ic.addSplineDesignVariables(problem, self.accelBiasDv, setActive=True, \
                                        group_id=ic.HELPER_GROUP_ID)

        self.q_i_b_Dv = aopt.RotationQuaternionDv(self.init_q_i_b)
        problem.addDesignVariable(self.q_i_b_Dv, ic.HELPER_GROUP_ID)
        self.q_i_b_Dv.setActive(False)
        self.r_b_i_Dv = aopt.EuclideanPointDv(np.array([0., 0., 0.]))
        problem.addDesignVariable(self.r_b_i_Dv, ic.HELPER_GROUP_ID)
        self.r_b_i_Dv.setActive(False)

        if not self.isReference:
            self.q_i_b_Dv.setActive(True)
            self.r_b_i_Dv.setActive(True)

    def addAccelerometerErrorTerms(self, problem, poseSplineDv, g_w, mSigma=0.0, \
                                   accelNoiseScale=1.0):
        print
        print "Adding accelerometer error terms ({0})".format(self.dataset.topic)

        # progress bar
        iProgress = sm.Progress2(len(self.imuData))
        iProgress.sample()

        # AccelerometerError(measurement,  invR,  C_b_w,  acceleration_w,  bias,  g_w)
        weight = 1.0 / accelNoiseScale
        accelErrors = []
        num_skipped = 0

        if mSigma > 0.0:
            mest = aopt.HuberMEstimator(mSigma)
        else:
            mest = aopt.NoMEstimator()

        for im in self.imuData:
            tk = im.stamp.toSec() + self.timeOffset
            if tk > poseSplineDv.spline().t_min() and tk < poseSplineDv.spline().t_max():
                C_b_w = poseSplineDv.orientation(tk).inverse()
                a_w = poseSplineDv.linearAcceleration(tk)
                if self.staticBias:
                    b_i = self.accelBiasDv.toExpression()
                else:
                    b_i = self.accelBiasDv.toEuclideanExpression(tk, 0)
                w_b = poseSplineDv.angularVelocityBodyFrame(tk)
                w_dot_b = poseSplineDv.angularAccelerationBodyFrame(tk)
                C_i_b = self.q_i_b_Dv.toExpression()
                r_b = self.r_b_i_Dv.toExpression()
                a = C_i_b * (C_b_w * (a_w - g_w) + \
                             w_dot_b.cross(r_b) + w_b.cross(w_b.cross(r_b)))
                aerr = ket.EuclideanError(im.alpha, im.alphaInvR * weight, a + b_i)
                aerr.setMEstimatorPolicy(mest)
                accelErrors.append(aerr)
                problem.addErrorTerm(aerr)
            else:
                num_skipped = num_skipped + 1

            # update progress bar
            iProgress.sample()

        print "\r  Added {0} of {1} accelerometer error terms (skipped {2} out-of-bounds measurements)".format(
            len(self.imuData) - num_skipped, len(self.imuData), num_skipped)
        self.accelErrors = accelErrors

    def addGyroscopeErrorTerms(self, problem, poseSplineDv, mSigma=0.0, gyroNoiseScale=1.0, \
                               g_w=None):
        print
        print "Adding gyroscope error terms ({0})".format(self.dataset.topic)

        # progress bar
        iProgress = sm.Progress2(len(self.imuData))
        iProgress.sample()

        num_skipped = 0
        gyroErrors = []
        weight = 1.0 / gyroNoiseScale
        if mSigma > 0.0:
            mest = aopt.HuberMEstimator(mSigma)
        else:
            mest = aopt.NoMEstimator()

        for im in self.imuData:
            tk = im.stamp.toSec() + self.timeOffset
            if tk > poseSplineDv.spline().t_min() and tk < poseSplineDv.spline().t_max():
                # GyroscopeError(measurement, invR, angularVelocity, bias)
                w_b = poseSplineDv.angularVelocityBodyFrame(tk)
                if self.staticBias:
                    b_i = self.gyroBiasDv.toExpression()
                else:
                    b_i = self.gyroBiasDv.toEuclideanExpression(tk, 0)
                C_i_b = self.q_i_b_Dv.toExpression()
                w = C_i_b * w_b
                gerr = ket.EuclideanError(im.omega, im.omegaInvR * weight, w + b_i)
                gerr.setMEstimatorPolicy(mest)
                gyroErrors.append(gerr)
                problem.addErrorTerm(gerr)
            else:
                num_skipped = num_skipped + 1

            # update progress bar
            iProgress.sample()

        print "\r  Added {0} of {1} gyroscope error terms (skipped {2} out-of-bounds measurements)".format(
            len(self.imuData) - num_skipped, len(self.imuData), num_skipped)
        self.gyroErrors = gyroErrors

    def initBiasSplines(self, poseSpline, splineOrder, biasKnotsPerSecond):
        start = poseSpline.t_min();
        end = poseSpline.t_max();
        seconds = end - start;
        knots = int(round(seconds * biasKnotsPerSecond))

        print
        print "Initializing the bias splines with %d knots" % (knots)

        if not self.staticBias:
            # initialize the bias splines
            self.gyroBias = bsplines.BSpline(splineOrder)
            self.gyroBias.initConstantSpline(start, end, knots, self.GyroBiasPrior)

            self.accelBias = bsplines.BSpline(splineOrder)
            self.accelBias.initConstantSpline(start, end, knots, np.zeros(3))

    def addBiasMotionTerms(self, problem):
        Wgyro = np.eye(3) / (self.gyroRandomWalk * self.gyroRandomWalk)
        Waccel = np.eye(3) / (self.accelRandomWalk * self.accelRandomWalk)
        gyroBiasMotionErr = asp.BSplineEuclideanMotionError(self.gyroBiasDv, Wgyro, 1)
        problem.addErrorTerm(gyroBiasMotionErr)
        accelBiasMotionErr = asp.BSplineEuclideanMotionError(self.accelBiasDv, Waccel, 1)
        problem.addErrorTerm(accelBiasMotionErr)

    def getTransformationFromReferenceToImu(self):
        if self.isReference:
            return sm.Transformation()
        return sm.Transformation(sm.r2quat(self.q_i_b_Dv.toRotationMatrix()), \
                                 -np.dot(self.q_i_b_Dv.toRotationMatrix(), \
                                        self.r_b_i_Dv.toEuclidean()))

    def findOrientationPrior(self, referenceImu):
        print
        print "Estimating imu-imu rotation initial guess."

        # build the problem
        problem = aopt.OptimizationProblem()

        # Add the rotation as design variable
        q_i_b_Dv = aopt.RotationQuaternionDv(np.array([0.0, 0.0, 0.0, 1.0]))
        q_i_b_Dv.setActive(True)
        problem.addDesignVariable(q_i_b_Dv)

        # Add spline representing rotational velocity of in body frame
        startTime = self.imuData[0].stamp.toSec()
        endTime = self.imuData[-1].stamp.toSec()
        knotsPerSecond = 50
        knots = int(round((endTime - startTime) * knotsPerSecond))

        angularVelocity = bsplines.BSpline(3)
        angularVelocity.initConstantSpline(startTime, endTime, knots, np.array([0., 0., 0.]))
        angularVelocityDv = asp.EuclideanBSplineDesignVariable(angularVelocity)

        for i in range(0, angularVelocityDv.numDesignVariables()):
            dv = angularVelocityDv.designVariable(i)
            dv.setActive(True)
            problem.addDesignVariable(dv)

        # Add constant reference gyro bias as design variable
        referenceGyroBiasDv = aopt.EuclideanPointDv(np.zeros(3))
        referenceGyroBiasDv.setActive(True)
        problem.addDesignVariable(referenceGyroBiasDv)

        for im in referenceImu.imuData:
            tk = im.stamp.toSec()
            if tk > angularVelocity.t_min() and tk < angularVelocity.t_max():
                # DV expressions
                bias = referenceGyroBiasDv.toExpression()

                omega_predicted = angularVelocityDv.toEuclideanExpression(tk, 0)
                omega_measured = im.omega

                # error term
                gerr = ket.GyroscopeError(im.omega, im.omegaInvR, omega_predicted, bias)
                problem.addErrorTerm(gerr)

        # define the optimization
        options = aopt.Optimizer2Options()
        options.verbose = False
        options.linearSolver = aopt.BlockCholeskyLinearSystemSolver()
        options.nThreads = 2
        options.convergenceDeltaX = 1e-4
        options.convergenceJDescentRatioThreshold = 1e-5
        options.maxIterations = 50

        # run the optimization
        optimizer = aopt.Optimizer2(options)
        optimizer.setProblem(problem)

        try:
            optimizer.optimize()
        except:
            sm.logFatal("Failed to obtain initial guess for the relative orientation!")
            sys.exit(-1)

        referenceAbsoluteOmega = lambda dt=np.array([0.]): \
            np.asarray([np.linalg.norm(angularVelocityDv.toEuclidean(im.stamp.toSec() + dt[0], 0)) \
                        for im in self.imuData \
                        if (im.stamp.toSec() + dt[0] > angularVelocity.t_min() \
                            and im.stamp.toSec() + dt[0] < angularVelocity.t_max())])
        absoluteOmega = lambda dt=np.array([0.]): \
            np.asarray([np.linalg.norm(im.omega) for im in self.imuData \
                        if (im.stamp.toSec() + dt[0] > angularVelocity.t_min() \
                            and im.stamp.toSec() + dt[0] < angularVelocity.t_max())])

        if len(referenceAbsoluteOmega()) == 0 or len(absoluteOmega()) == 0:
            sm.logFatal("The time ranges of the IMUs published as topics {0} and {1} do not overlap. " \
                        "Please make sure that the sensors are synchronized correctly." \
                        .format(referenceImu.imuConfig.getRosTopic(), self.imuConfig.getRosTopic()))
            sys.exit(-1)

        # get the time shift
        corr = np.correlate(referenceAbsoluteOmega(), absoluteOmega(), "full")
        discrete_shift = corr.argmax() - (np.size(absoluteOmega()) - 1)
        # get cont. time shift
        times = [im.stamp.toSec() for im in self.imuData]
        dT = np.mean(np.diff(times))
        shift = discrete_shift * dT

        if self.estimateTimedelay and not self.isReference:
            # refine temporal offset only when used.
            objectiveFunction = lambda dt: np.linalg.norm(referenceAbsoluteOmega(dt) - absoluteOmega(dt)) ** 2
            refined_shift = scipy.optimize.fmin(objectiveFunction, np.array([shift]), maxiter=100)[0]
            self.timeOffset = float(refined_shift)

        print "Temporal correction with respect to reference IMU "
        print self.timeOffset, "[s]", ("" if self.estimateTimedelay else \
                                           " (this offset is not accounted for in the calibration)")

        # Add constant gyro bias as design variable
        gyroBiasDv = aopt.EuclideanPointDv(np.zeros(3))
        gyroBiasDv.setActive(True)
        problem.addDesignVariable(gyroBiasDv)

        for im in self.imuData:
            tk = im.stamp.toSec() + self.timeOffset
            if tk > angularVelocity.t_min() and tk < angularVelocity.t_max():
                # DV expressions
                C_i_b = q_i_b_Dv.toExpression()
                bias = gyroBiasDv.toExpression()

                omega_predicted = C_i_b * angularVelocityDv.toEuclideanExpression(tk, 0)
                omega_measured = im.omega

                # error term
                gerr = ket.GyroscopeError(im.omega, im.omegaInvR, omega_predicted, bias)
                problem.addErrorTerm(gerr)

        # get the prior
        try:
            optimizer.optimize()
        except:
            sm.logFatal("Failed to obtain initial guess for the relative orientation!")
            sys.exit(-1)

        print "Estimated imu to reference imu Rotation: "
        print q_i_b_Dv.toRotationMatrix()

        self.init_q_i_b = sm.r2quat(q_i_b_Dv.toRotationMatrix())

    def getImuConfig(self):
        self.updateImuConfig()
        return self.imuConfig

    def updateImuConfig(self):
        self.imuConfig.setExtrinsicsReferenceToHere(self.getTransformationFromReferenceToImu())
        self.imuConfig.setTimeshiftToReference(self.timeOffset)


class ScaledMisalignedImu(Imu):
    class ImuParameters(kc.ImuParameters):
        def __init__(self, imuConfig):
            kc.ImuParameters.__init__(self, imuConfig)
            self.data = imuConfig.data
            self.data["model"] = "scale-misalignment"

        def printDetails(self, dest=sys.stdout):
            kc.ImuParameters.printDetails(self, dest)
            print >> dest, "  Gyroscope: "
            print >> dest, "    M:"
            print >> dest, self.formatIndented("      ", np.array(self.data["gyroscopes"]["M"]))
            print >> dest, "    A [(rad/s)/(m/s^2)]:"
            print >> dest, self.formatIndented("      ", np.array(self.data["gyroscopes"]["A"]))
            print >> dest, "    C_gyro_i:"
            print >> dest, self.formatIndented("      ", np.array(self.data["gyroscopes"]["C_gyro_i"]))
            print >> dest, "  Accelerometer: "
            print >> dest, "    M:"
            print >> dest, self.formatIndented("      ", np.array(self.data["accelerometers"]["M"]))

        def setIntrisicsMatrices(self, M_accel, C_gyro_i, M_gyro, Ma_gyro):
            self.data["accelerometers"] = dict()
            self.data["accelerometers"]["M"] = M_accel.tolist()
            self.data["gyroscopes"] = dict()
            self.data["gyroscopes"]["M"] = M_gyro.tolist()
            self.data["gyroscopes"]["A"] = Ma_gyro.tolist()
            self.data["gyroscopes"]["C_gyro_i"] = C_gyro_i.tolist()

    def updateImuConfig(self):
        Imu.updateImuConfig(self)
        self.imuConfig.setIntrisicsMatrices(self.M_accel_Dv.toMatrix3x3(), \
                                            self.q_gyro_i_Dv.toRotationMatrix(), \
                                            self.M_gyro_Dv.toMatrix3x3(), \
                                            self.M_accel_gyro_Dv.toMatrix3x3())

    def addDesignVariables(self, problem):
        Imu.addDesignVariables(self, problem)

        self.q_gyro_i_Dv = aopt.RotationQuaternionDv(np.array([0., 0., 0., 1.]))
        problem.addDesignVariable(self.q_gyro_i_Dv, ic.HELPER_GROUP_ID)
        self.q_gyro_i_Dv.setActive(True)

        self.M_accel_Dv = aopt.MatrixBasicDv(np.eye(3), np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], \
                                                                 dtype=int))
        problem.addDesignVariable(self.M_accel_Dv, ic.HELPER_GROUP_ID)
        self.M_accel_Dv.setActive(True)

        self.M_gyro_Dv = aopt.MatrixBasicDv(np.eye(3), np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], \
                                                                dtype=int))
        problem.addDesignVariable(self.M_gyro_Dv, ic.HELPER_GROUP_ID)
        self.M_gyro_Dv.setActive(True)

        self.M_accel_gyro_Dv = aopt.MatrixBasicDv(np.zeros((3, 3)), np.ones((3, 3), dtype=int))
        problem.addDesignVariable(self.M_accel_gyro_Dv, ic.HELPER_GROUP_ID)
        self.M_accel_gyro_Dv.setActive(True)

    def addAccelerometerErrorTerms(self, problem, poseSplineDv, g_w, mSigma=0.0, \
                                   accelNoiseScale=1.0):
        print
        print "Adding accelerometer error terms ({0})".format(self.dataset.topic)

        # progress bar
        iProgress = sm.Progress2(len(self.imuData))
        iProgress.sample()

        # AccelerometerError(measurement,  invR,  C_b_w,  acceleration_w,  bias,  g_w)
        weight = 1.0 / accelNoiseScale
        accelErrors = []
        num_skipped = 0

        if mSigma > 0.0:
            mest = aopt.HuberMEstimator(mSigma)
        else:
            mest = aopt.NoMEstimator()

        for im in self.imuData:
            tk = im.stamp.toSec() + self.timeOffset
            if tk > poseSplineDv.spline().t_min() and tk < poseSplineDv.spline().t_max():
                C_b_w = poseSplineDv.orientation(tk).inverse()
                a_w = poseSplineDv.linearAcceleration(tk)
                if self.staticBias:
                    b_i = self.accelBiasDv.toExpression()
                else:
                    b_i = self.accelBiasDv.toEuclideanExpression(tk, 0)
                M = self.M_accel_Dv.toExpression()
                w_b = poseSplineDv.angularVelocityBodyFrame(tk)
                w_dot_b = poseSplineDv.angularAccelerationBodyFrame(tk)
                C_i_b = self.q_i_b_Dv.toExpression()
                r_b = self.r_b_i_Dv.toExpression()
                a = M * (C_i_b * (C_b_w * (a_w - g_w) + \
                                  w_dot_b.cross(r_b) + w_b.cross(w_b.cross(r_b))))

                aerr = ket.EuclideanError(im.alpha, im.alphaInvR * weight, a + b_i)
                aerr.setMEstimatorPolicy(mest)
                accelErrors.append(aerr)
                problem.addErrorTerm(aerr)
            else:
                num_skipped = num_skipped + 1

            # update progress bar
            iProgress.sample()

        print "\r  Added {0} of {1} accelerometer error terms (skipped {2} out-of-bounds measurements)".format(
            len(self.imuData) - num_skipped, len(self.imuData), num_skipped)
        self.accelErrors = accelErrors

    def addGyroscopeErrorTerms(self, problem, poseSplineDv, mSigma=0.0, gyroNoiseScale=1.0, g_w=None):
        print
        print "Adding gyroscope error terms ({0})".format(self.dataset.topic)

        # progress bar
        iProgress = sm.Progress2(len(self.imuData))
        iProgress.sample()

        num_skipped = 0
        gyroErrors = []
        weight = 1.0 / gyroNoiseScale
        if mSigma > 0.0:
            mest = aopt.HuberMEstimator(mSigma)
        else:
            mest = aopt.NoMEstimator()

        for im in self.imuData:
            tk = im.stamp.toSec() + self.timeOffset
            if tk > poseSplineDv.spline().t_min() and tk < poseSplineDv.spline().t_max():
                # GyroscopeError(measurement, invR, angularVelocity, bias)
                w_b = poseSplineDv.angularVelocityBodyFrame(tk)
                w_dot_b = poseSplineDv.angularAccelerationBodyFrame(tk)
                if self.staticBias:
                    b_i = self.gyroBiasDv.toExpression()
                else:
                    b_i = self.gyroBiasDv.toEuclideanExpression(tk, 0)
                C_b_w = poseSplineDv.orientation(tk).inverse()
                a_w = poseSplineDv.linearAcceleration(tk)
                r_b = self.r_b_i_Dv.toExpression()
                a_b = C_b_w * (a_w - g_w) + w_dot_b.cross(r_b) + w_b.cross(w_b.cross(r_b))

                C_i_b = self.q_i_b_Dv.toExpression()
                C_gyro_i = self.q_gyro_i_Dv.toExpression()
                C_gyro_b = C_gyro_i * C_i_b
                M = self.M_gyro_Dv.toExpression()
                Ma = self.M_accel_gyro_Dv.toExpression()

                w = M * (C_gyro_b * w_b) + Ma * (C_gyro_b * a_b)

                gerr = ket.EuclideanError(im.omega, im.omegaInvR * weight, w + b_i)
                gerr.setMEstimatorPolicy(mest)
                gyroErrors.append(gerr)
                problem.addErrorTerm(gerr)
            else:
                num_skipped = num_skipped + 1

            # update progress bar
            iProgress.sample()

        print "\r  Added {0} of {1} gyroscope error terms (skipped {2} out-of-bounds measurements)".format(
            len(self.imuData) - num_skipped, len(self.imuData), num_skipped)
        self.gyroErrors = gyroErrors


class ScaledMisalignedSizeEffectImu(ScaledMisalignedImu):
    class ImuParameters(ScaledMisalignedImu.ImuParameters):
        def __init__(self, imuConfig):
            ScaledMisalignedImu.ImuParameters.__init__(self, imuConfig)
            self.data = imuConfig.data
            self.data["model"] = "scale-misalignment-size-effect"

        def printDetails(self, dest=sys.stdout):
            ScaledMisalignedImu.ImuParameters.printDetails(self, dest)
            print >> dest, "    rx_i [m]:"
            print >> dest, self.formatIndented("      ", \
                                               np.array(self.data["accelerometers"]["rx_i"]))
            print >> dest, "    ry_i [m]:"
            print >> dest, self.formatIndented("      ", \
                                               np.array(self.data["accelerometers"]["ry_i"]))
            print >> dest, "    rz_i [m]:"
            print >> dest, self.formatIndented("      ", \
                                               np.array(self.data["accelerometers"]["rz_i"]))

        def setAccelerometerLeverArms(self, rx_i, ry_i, rz_i):
            self.data["accelerometers"]["rx_i"] = rx_i.tolist()
            self.data["accelerometers"]["ry_i"] = ry_i.tolist()
            self.data["accelerometers"]["rz_i"] = rz_i.tolist()

    def updateImuConfig(self):
        ScaledMisalignedImu.updateImuConfig(self)
        self.imuConfig.setAccelerometerLeverArms(self.rx_i_Dv.toEuclidean(), \
                                                 self.ry_i_Dv.toEuclidean(), \
                                                 self.rz_i_Dv.toEuclidean())

    def addDesignVariables(self, problem):
        ScaledMisalignedImu.addDesignVariables(self, problem)

        self.rx_i_Dv = aopt.EuclideanPointDv(np.array([0., 0., 0.]))
        problem.addDesignVariable(self.rx_i_Dv, ic.HELPER_GROUP_ID)
        self.rx_i_Dv.setActive(False)

        self.ry_i_Dv = aopt.EuclideanPointDv(np.array([0., 0., 0.]))
        problem.addDesignVariable(self.ry_i_Dv, ic.HELPER_GROUP_ID)
        self.ry_i_Dv.setActive(True)

        self.rz_i_Dv = aopt.EuclideanPointDv(np.array([0., 0., 0.]))
        problem.addDesignVariable(self.rz_i_Dv, ic.HELPER_GROUP_ID)
        self.rz_i_Dv.setActive(True)

        self.Ix_Dv = aopt.MatrixBasicDv(np.diag([1., 0., 0.]), np.zeros((3, 3), dtype=int))
        problem.addDesignVariable(self.Ix_Dv, ic.HELPER_GROUP_ID)
        self.Ix_Dv.setActive(False)
        self.Iy_Dv = aopt.MatrixBasicDv(np.diag([0., 1., 0.]), np.zeros((3, 3), dtype=int))
        problem.addDesignVariable(self.Iy_Dv, ic.HELPER_GROUP_ID)
        self.Iy_Dv.setActive(False)
        self.Iz_Dv = aopt.MatrixBasicDv(np.diag([0., 0., 1.]), np.zeros((3, 3), dtype=int))
        problem.addDesignVariable(self.Iz_Dv, ic.HELPER_GROUP_ID)
        self.Iz_Dv.setActive(False)

    def addAccelerometerErrorTerms(self, problem, poseSplineDv, g_w, mSigma=0.0, \
                                   accelNoiseScale=1.0):
        print
        print "Adding accelerometer error terms ({0})".format(self.dataset.topic)

        # progress bar
        iProgress = sm.Progress2(len(self.imuData))
        iProgress.sample()

        # AccelerometerError(measurement,  invR,  C_b_w,  acceleration_w,  bias,  g_w)
        weight = 1.0 / accelNoiseScale
        accelErrors = []
        num_skipped = 0

        if mSigma > 0.0:
            mest = aopt.HuberMEstimator(mSigma)
        else:
            mest = aopt.NoMEstimator()

        for im in self.imuData:
            tk = im.stamp.toSec() + self.timeOffset
            if tk > poseSplineDv.spline().t_min() and tk < poseSplineDv.spline().t_max():
                C_b_w = poseSplineDv.orientation(tk).inverse()
                a_w = poseSplineDv.linearAcceleration(tk)
                b_i = self.accelBiasDv.toEuclideanExpression(tk, 0)
                M = self.M_accel_Dv.toExpression()
                w_b = poseSplineDv.angularVelocityBodyFrame(tk)
                w_dot_b = poseSplineDv.angularAccelerationBodyFrame(tk)
                C_i_b = self.q_i_b_Dv.toExpression()
                rx_b = self.r_b_i_Dv.toExpression() + C_i_b.inverse() * self.rx_i_Dv.toExpression()
                ry_b = self.r_b_i_Dv.toExpression() + C_i_b.inverse() * self.ry_i_Dv.toExpression()
                rz_b = self.r_b_i_Dv.toExpression() + C_i_b.inverse() * self.rz_i_Dv.toExpression()
                Ix = self.Ix_Dv.toExpression()
                Iy = self.Iy_Dv.toExpression()
                Iz = self.Iz_Dv.toExpression()

                a = M * (C_i_b * (C_b_w * (a_w - g_w)) + \
                         Ix * (C_i_b * (w_dot_b.cross(rx_b) + w_b.cross(w_b.cross(rx_b)))) + \
                         Iy * (C_i_b * (w_dot_b.cross(ry_b) + w_b.cross(w_b.cross(ry_b)))) + \
                         Iz * (C_i_b * (w_dot_b.cross(rz_b) + w_b.cross(w_b.cross(rz_b)))))

                aerr = ket.EuclideanError(im.alpha, im.alphaInvR * weight, a + b_i)
                aerr.setMEstimatorPolicy(mest)
                accelErrors.append(aerr)
                problem.addErrorTerm(aerr)
            else:
                num_skipped = num_skipped + 1

            # update progress bar
            iProgress.sample()

        print "\r  Added {0} of {1} accelerometer error terms (skipped {2} out-of-bounds measurements)".format(
            len(self.imuData) - num_skipped, len(self.imuData), num_skipped)
        self.accelErrors = accelErrors
