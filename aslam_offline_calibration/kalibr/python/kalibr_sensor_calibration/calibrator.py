import aslam_backend as aopt
import aslam_splines as asp
import util as util
import incremental_calibration as inc
import kalibr_common as kc
import sm

import gc
import numpy as np
import multiprocessing
import sys

# make numpy print prettier
np.set_printoptions(suppress=True)

CALIBRATION_GROUP_ID = 0
HELPER_GROUP_ID = 1


def addSplineDesignVariables(problem, dvc, setActive=True, group_id=HELPER_GROUP_ID):
    for i in range(0, dvc.numDesignVariables()):
        dv = dvc.designVariable(i)
        dv.setActive(setActive)
        problem.addDesignVariable(dv, group_id)


def optimize(problem, options=None, maxIterations=30):
    if options is None:
        options = aopt.Optimizer2Options()
        options.verbose = True
        options.doLevenbergMarquardt = True
        options.levenbergMarquardtLambdaInit = 10.0
        options.nThreads = max(1, multiprocessing.cpu_count() - 1)
        options.convergenceDeltaX = 1e-5
        options.convergenceJDescentRatioThreshold = 1e-6
        options.maxIterations = maxIterations
        options.trustRegionPolicy = aopt.LevenbergMarquardtTrustRegionPolicy(options.levenbergMarquardtLambdaInit)
        options.linearSolver = aopt.BlockCholeskyLinearSystemSolver()

    # run the optimization
    optimizer = aopt.Optimizer2(options)
    optimizer.setProblem(problem)

    optimizationFailed = False
    try:
        retval = optimizer.optimize()
        if retval.linearSolverFailure:
            optimizationFailed = True
    except:
        optimizationFailed = True

    if optimizationFailed:
        sm.logError("Optimization failed!")
        raise RuntimeError("Optimization failed!")

    # free some memory
    del optimizer
    gc.collect()


class Calibrator(object):
    def __init__(self, reference_sensor):
        self.ImuList = []
        self.LiDARList = []
        self.reference_sensor = reference_sensor

    def registerLiDAR(self, sensor):
        self.LiDARList.append(sensor)

    def constructLiDARErrorTerms(self, threshold=0.1):
        for lidar in self.LiDARList:
            lidar.findPointsOnTarget(self.poseDv, threshold=threshold)
            lidar.removeLiDARErrorTerms(self.problem)
            lidar.addLiDARErrorTerms(self.problem, self.poseDv)

    def optimize(self, options=None, maxIterations=30, recoverCov=False):
        optimize(self.problem, options, maxIterations=maxIterations)

        if self.LiDARList:
            num = 3
            for i in xrange(1, num):
                self.constructLiDARErrorTerms(0.3 / i)
                optimize(self.problem, maxIterations=maxIterations//i)

            for lidar in self.LiDARList:
                lidar.filterLiDARErrorTerms(self.problem, 1.0)
            optimize(self.problem, maxIterations=maxIterations)

        if recoverCov:
            self.recoverCovariance()

    def initDesignVariables(self, problem, poseSpline, noTimeCalibration, \
                            estimateGravityLength=False, initialGravityEstimate=np.array([0.0, 9.81, 0.0])):
        # Initialize the system pose spline
        self.poseDv = asp.BSplinePoseDesignVariable(poseSpline)
        addSplineDesignVariables(problem, self.poseDv)

        if self.ImuList:
            # Add the calibration target orientation design variable. (expressed as gravity vector in target frame)
            if estimateGravityLength:
                self.gravityDv = aopt.EuclideanPointDv(initialGravityEstimate)
            else:
                self.gravityDv = aopt.EuclideanDirection(initialGravityEstimate)
            self.gravityExpression = self.gravityDv.toExpression()
            self.gravityDv.setActive(True)
            problem.addDesignVariable(self.gravityDv, HELPER_GROUP_ID)

        # Add targets, fix the first target
        self.targets[0].addDesignVariables(problem, fixed=True)
        for target in self.targets[1:]:
            target.addDesignVariables(problem)

        # Add all DVs for all IMUs
        for imu in self.ImuList:
            imu.addDesignVariables(problem)

        # Add all DVs for the camera chain
        self.CameraChain.addDesignVariables(problem, noTimeCalibration)

        for lidar in self.LiDARList:
            lidar.addDesignVariables(problem, noTimeCalibration)

    def addPoseMotionTerms(self, problem, tv, rv):
        wt = 1.0 / tv
        wr = 1.0 / rv
        W = np.diag([wt, wt, wt, wr, wr, wr])
        errorOrder = 2
        asp.addMotionErrorTerms(problem, self.poseDv, W, errorOrder)

    def registerCalibrationTarget(self, targets):
        self.targets = targets
    # add camera to sensor list (create list if necessary)
    def registerCamChain(self, sensor):
        self.CameraChain = sensor

    def registerImu(self, sensor):
        self.ImuList.append(sensor)

    def buildProblem(self,
                     splineOrder=6,
                     poseKnotsPerSecond=70,
                     biasKnotsPerSecond=70,
                     doPoseMotionError=False,
                     mrTranslationVariance=1e6,
                     mrRotationVariance=1e5,
                     doBiasMotionError=True,
                     blakeZisserCam=-1,
                     huberAccel=-1,
                     huberGyro=-1,
                     noTimeCalibration=False,
                     maxIterations=20,
                     gyroNoiseScale=1.0,
                     accelNoiseScale=1.0,
                     timeOffsetPadding=0.02,
                     verbose=False):

        print "\tSpline order: %d" % splineOrder
        print "\tPose knots per second: %d" % poseKnotsPerSecond
        print "\tDo pose motion regularization: %s" % doPoseMotionError
        print "\t\txddot translation variance: %f" % mrTranslationVariance
        print "\t\txddot rotation variance: %f" % mrRotationVariance
        print "\tBias knots per second: %d" % biasKnotsPerSecond
        print "\tDo bias motion regularization: %s" % doBiasMotionError
        print "\tBlake-Zisserman on reprojection errors %s" % blakeZisserCam
        print "\tAcceleration Huber width (sigma): %f" % huberAccel
        print "\tGyroscope Huber width (sigma): %f" % huberGyro
        print "\tDo time calibration: %s" % (not noTimeCalibration)
        print "\tMax iterations: %d" % maxIterations
        print "\tTime offset padding: %f" % timeOffsetPadding

        ############################################
        ## initialize camera chain
        ############################################
        # estimate the timeshift for all cameras to the main imu
        self.noTimeCalibration = noTimeCalibration
        if not noTimeCalibration:
            for cam in self.CameraChain.camList:
                cam.findTimeshiftCameraImuPrior(self.ImuList[0], verbose)

        # obtain orientation prior between main imu and camera chain (if no external input provided)
        # and initial estimate for the direction of gravity

        self.CameraChain.findTargetPoseInWorld(self.targets)

        # self.CameraChain.findExtrinsicPriorCameraChain()

        # estimatedGravity = self.CameraChain.findOrientationPriorCameraChainToImu(self.ImuList[0])
        imu = self.ImuList[0] if self.ImuList else None
        estimatedGravity = self.CameraChain.findExtrinsicPriorSensorsToCamera(imu, self.LiDARList)

        ############################################
        ## init optimization problem
        ############################################
        # initialize a pose spline using the camera poses in the camera chain
        poseSpline = self.CameraChain.initializePoseSplineFromCameraChain(splineOrder, poseKnotsPerSecond,
                                                                          timeOffsetPadding)

        # Initialize bias splines for all IMUs
        for imu in self.ImuList:
            imu.initBiasSplines(poseSpline, splineOrder, biasKnotsPerSecond)

        # Now I can build the problem
        problem = inc.CalibrationOptimizationProblem()

        # Initialize all design variables.
        self.initDesignVariables(problem, poseSpline, noTimeCalibration,
                                 initialGravityEstimate=estimatedGravity)

        ############################################
        ## add error terms
        ############################################
        # Add calibration target reprojection error terms for all camera in chain
        self.CameraChain.addCameraChainErrorTerms(problem, self.poseDv, blakeZissermanDf=blakeZisserCam,
                                                  timeOffsetPadding=timeOffsetPadding)

        # Initialize IMU error terms.
        for imu in self.ImuList:
            imu.addAccelerometerErrorTerms(problem, self.poseDv, self.gravityExpression, mSigma=huberAccel,
                                           accelNoiseScale=accelNoiseScale)
            imu.addGyroscopeErrorTerms(problem, self.poseDv, mSigma=huberGyro, gyroNoiseScale=gyroNoiseScale,
                                       g_w=self.gravityExpression)

            # Add the bias motion terms.
            if doBiasMotionError and not imu.staticBias:
                imu.addBiasMotionTerms(problem)

        # Add the pose motion terms.
        if doPoseMotionError:
            self.addPoseMotionTerms(problem, mrTranslationVariance, mrRotationVariance)

        # Add a gravity prior
        self.problem = problem

    def recoverCovariance(self):
        # Covariance ordering (=dv ordering)
        # ORDERING:   N=num cams
        #            1. transformation imu-cam0 --> 6
        #            2. camera time2imu --> 1*numCams (only if enabled)

        print "Recovering covariance..."
        estimator = inc.IncrementalEstimator(CALIBRATION_GROUP_ID)
        rval = estimator.addBatch(self.problem, True)
        est_stds = np.sqrt(estimator.getSigma2Theta().diagonal())

        # split and store the variance
        self.std_trafo_ic = np.array(est_stds[0:6])
        self.std_times = np.array(est_stds[6:])

    def saveImuSetParametersYaml(self, resultFile):
        imuSetConfig = kc.ImuSetParameters(resultFile, self.reference_sensor, True)
        for imu in self.ImuList:
            imuConfig = imu.getImuConfig()
            imuSetConfig.addImuParameters(imu_parameters=imuConfig)
        imuSetConfig.writeYaml(resultFile)

    def saveCamChainParametersYaml(self, resultFile):
        chain = self.CameraChain.chainConfig
        nCams = len(self.CameraChain.camList)

        # Calibration results
        for camNr in range(0, nCams):
            # cam-cam baselines
            if camNr > 0:
                T_cB_cA, baseline = self.CameraChain.getResultBaseline(camNr - 1, camNr)
                chain.setExtrinsicsLastCamToHere(camNr, T_cB_cA)

            # imu-cam trafos
            t_c_ref = self.CameraChain.getTransformationReferenceToCam(camNr)
            chain.setExtrinsicsReferenceToCam(camNr, t_c_ref)

            if not self.noTimeCalibration:
                # imu to cam timeshift
                timeshift = float(self.CameraChain.getResultTimeShift(camNr))
                chain.setTimeshiftCamReference(camNr, timeshift)

        try:
            chain.writeYaml(resultFile)
        except:
            print "ERROR: Could not write parameters to file: {0}\n".format(resultFile)

    def saveLiDARsParametersYaml(self, resultFile):
        lidarListConfig = kc.LiDARListParameters(resultFile, self.reference_sensor, True)

        for lidar in self.LiDARList:
            lidarConfig = lidar.getLiDARConfig()
            lidarListConfig.addLiDARParameters(lidarConfig)
        lidarListConfig.writeYaml(resultFile)
