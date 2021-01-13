from __future__ import print_function
from __future__ import division

from collections import namedtuple
import sys
import numpy as np
import aslam_backend as aopt
import sm
import kalibr_errorterms as ket
import util as util


EstepResult = namedtuple('EstepResult', ['m0', 'm1', 'nx'])
MstepResult = namedtuple('MstepResult', ['transformation', 'q'])

class Permutohedral(object):
    def __init__(self, p, with_blur=True):
        self._impl = sm.Permutohedral()
        self._impl.init(p.astype(np.float32).T, with_blur)

    def get_lattice_size(self):
        return self._impl.get_lattice_size()

    def filter(self, v, start=0):
        return self._impl.filter(v.astype(np.float32).T, start).T.astype(np.float64)


class LiDARToSensorCalibrator:
    def __init__(self, source, sensor_tfs, target_normals=None):
        self._source = source
        self._sensor_tfs = sensor_tfs
        self._target_normals = target_normals
        self._tf_result = np.eye(4, dtype=np.float32)

    def maximization_step(self, t_source, target, estep_res, w=0.0,
                           objective_type='pt2pt'):
        m, ndim = t_source.shape
        n = target.shape[0]
        assert ndim == 3, "ndim must be 3."
        m0, m1, nx = estep_res
        c = w / (1.0 - w) * n / m
        m0[m0==0] = np.finfo(np.float32).eps
        m1m0 = np.divide(m1.T, m0).T
        m0m0 = m0 / (m0 + c)
        drxdx = m0m0
        errs = []
        self.problem.clearAllErrorTerms()
        if objective_type == 'pt2pt':
            for i in xrange(m):
                T_l_l = self.T_b_l_Dv.toExpression().inverse() * \
                        aopt.TransformationExpression(self._sensor_tfs[i]) * \
                        self.T_b_l_Dv.toExpression()
                predicted = T_l_l.toRotationExpression() * t_source[i] + T_l_l.toEuclideanExpression()
                err = ket.EuclideanError(m1m0[i], drxdx[i]*np.eye(3, dtype=np.float64), predicted)
                errs.append(err)
                self.problem.addErrorTerm(err)
        else:
            raise ValueError('Unknown objective_type: %s.' % objective_type)

        # define the optimization
        options = aopt.Optimizer2Options()
        options.verbose = False
        options.linearSolver = aopt.BlockCholeskyLinearSystemSolver()
        options.nThreads = 2
        options.convergenceDeltaX = 1e-4
        options.convergenceJDescentRatioThreshold = 1e-6
        options.maxIterations = 50

        # run the optimization
        optimizer = aopt.Optimizer2(options)
        optimizer.setProblem(self.problem)

        # get the prior
        try:
            optimizer.optimize()
        except:
            sm.logFatal("Failed to obtain orientation prior!")
            sys.exit(-1)

        q = np.array([np.linalg.norm(e.error()) for e in errs]).sum()
        return MstepResult(self.T_b_l_Dv.toExpression().toTransformationMatrix(), q)

    def set_target_normals(self, target_normals):
        self._target_normals = target_normals

    def expectation_step(self, t_source, target, y,
                         objective_type='pt2pt', alpha=0.015):
        """Expectation step
        """
        assert t_source.ndim == 2 and target.ndim == 2, "source and target must have 2 dimensions."
        m, ndim = t_source.shape
        n = target.shape[0]
        fx = t_source
        fy = target
        zero_m1 = np.zeros((m, 1))
        zeros_md = np.zeros((m, y.shape[1]))
        fin = np.r_[fx, fy]
        ph = Permutohedral(fin)
        if ph.get_lattice_size() < n * alpha:
            ph = Permutohedral(fin, False)
        vin0 = np.r_[zero_m1, np.ones((n, 1))]
        vin1 = np.r_[zeros_md, y]
        m0 = ph.filter(vin0, m).flatten()[:m]
        m1 = ph.filter(vin1, m)[:m]

        if objective_type == 'pt2pt':
            nx = None
        elif objective_type == 'pt2pl':
            vin = np.r_[zeros_md, self._target_normals]
            nx = ph.filter(vin, m)[:m]
        else:
            raise ValueError('Unknown objective_type: %s.' % objective_type)
        return EstepResult(m0, m1, nx)

    def registration(self, target, w=0.0,
                     objective_type='pt2pt',
                     maxiter=50, tol=0.001,
                     feature_fn=lambda x: x):
        q = None
        ftarget = feature_fn(target)

        # build the problem
        self.problem = aopt.OptimizationProblem()
        T_init = np.array([[-0.99813747, -0.05730897, -0.02091093, 0.03399231],
                           [0.05726434, -0.99835533, 0.00272772, 0.31433327],
                           [-0.02103286, 0.00152519, 0.99977762, 0.22997991],
                           [0., 0., 0., 1.]])
        self.T_b_l_Dv = aopt.TransformationDv(sm.Transformation(), rotationActive=True, translationActive=True)
        for i in range(0, self.T_b_l_Dv.numDesignVariables()):
            self.problem.addDesignVariable(self.T_b_l_Dv.getDesignVariable(i))

        for _ in range(maxiter):
            T_b_l = self.T_b_l_Dv.toExpression().toTransformationMatrix()
            print("Inital laser to body transformation: T_b_l ")
            print(T_b_l)
            T_l_b = np.linalg.inv(T_b_l)

            numPoints = self._source.shape[0]
            source = np.hstack([self._source, np.ones((numPoints, 1), dtype=self._source.dtype)])
            t_source = np.array([np.dot(T_l_b, np.dot(self._sensor_tfs[i], np.dot(T_b_l, source[i]))) for i in xrange(numPoints)])
            t_source = t_source[:, :3]
            util.showPointCloud([t_source, target])
            fsource = feature_fn(t_source)
            estep_res = self.expectation_step(fsource, ftarget, target, objective_type)

            res = self.maximization_step(self._source, target, estep_res, w=w,
                                         objective_type=objective_type)

            if not q is None and abs(res.q - q) < tol:
                break
            q = res.q
        return res.transformation


def calibrateLiDARToSensor(source, target, sensor_tfs, target_normals=None,
                           objective_type='pt2pt', maxiter=50,
                           tol=0.001, feature_fn=lambda x: x):
    """FilterReg registration

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        sensor_tfs (list of numpy.ndarray): The Transformation of sensor pose between time of source and target
        target_normals (numpy.ndarray, optional): Normal vectors of target point cloud.
        objective_type (str, optional): The type of objective function selected by 'pt2pt' or 'pt2pl'.
        maxitr (int, optional): Maximum number of iterations to EM algorithm.
        tol (float, optional): Tolerance for termination.
        feature_fn (function, optional): Feature function. If you use FPFH feature, set `feature_fn=probreg.feature.FPFH()`.
    """

    calibrator = LiDARToSensorCalibrator(source, sensor_tfs, target_normals)
    return calibrator.registration(target, objective_type=objective_type, maxiter=maxiter,
                            tol=tol, feature_fn=feature_fn)
