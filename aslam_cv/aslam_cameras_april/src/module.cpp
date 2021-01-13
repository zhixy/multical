// It is extremely important to use this header
// if you are using the numpy_eigen interface
#include <aslam/cameras/GridCalibrationTargetAprilgrid.hpp>
#include <aslam/cameras/MultipleTargetAprilGridDetector.hpp>
#include <numpy_eigen/boost_python_headers.hpp>
#include <sm/python/stl_converters.hpp>
#include <sm/python/boost_serialization_pickle.hpp>
#include <opencv2/core/eigen.hpp>
#include <aslam/python/PythonImageList.hpp>

typedef aslam::python::PythonImageList::image_t image_t;
bool initCameraGeometryFromObservation(aslam::cameras::MultipleTargetAprilGridDetector *gd,
                                       const image_t &image) {
    cv::Mat image_cv;
    eigen2cv(image, image_cv);
    return gd->initCameraGeometryFromObservation(image_cv);
}

bool initCameraGeometryFromObservations(aslam::cameras::MultipleTargetAprilGridDetector *gd,
                                        aslam::python::PythonImageList &image_list) {
    return gd->initCameraGeometryFromObservations(image_list.getImages());
}

boost::python::tuple findTarget3(aslam::cameras::MultipleTargetAprilGridDetector *gd,
                                 const aslam::Time &stamp,
                                 const image_t &image) {
    cv::Mat to;
    eigen2cv(image, to);
    std::vector<aslam::cameras::GridCalibrationTargetObservation> obs;
    bool success = gd->findTarget(to, stamp, obs);
    boost::python::list obsList;
    sm::python::stlToList(obs.begin(), obs.end(), obsList);
    return boost::python::make_tuple(success, obsList);
}

boost::python::tuple findTarget4(
        aslam::cameras::MultipleTargetAprilGridDetector *gd,
        const Eigen::Matrix <boost::uint8_t, Eigen::Dynamic, Eigen::Dynamic> &image) {
    return findTarget3(gd, aslam::Time(0, 0), image);
}

boost::python::tuple findTargetNoTransformation3(aslam::cameras::MultipleTargetAprilGridDetector *gd,
                                                 const aslam::Time &stamp,
                                                 const image_t &image) {
    cv::Mat to;
    eigen2cv(image, to);
    std::vector<aslam::cameras::GridCalibrationTargetObservation> obs;
    bool success = gd->findTargetNoTransformation(to, stamp, obs);
    boost::python::list obsList;
    sm::python::stlToList(obs.begin(), obs.end(), obsList);
    return boost::python::make_tuple(success, obsList);
}

boost::python::tuple findTargetNoTransformation4(aslam::cameras::MultipleTargetAprilGridDetector *gd,
                                                 const image_t &image) {
    return findTargetNoTransformation3(gd, aslam::Time(0, 0), image);
}

BOOST_PYTHON_MODULE(libaslam_cameras_april_python)
{
    using namespace boost::python;
    using namespace aslam::cameras;

    class_<GridCalibrationTargetAprilgrid::AprilgridOptions>("AprilgridOptions", init<>())
    .def_readwrite("doSubpixRefinement", &GridCalibrationTargetAprilgrid::AprilgridOptions::doSubpixRefinement)
    .def_readwrite("showExtractionVideo", &GridCalibrationTargetAprilgrid::AprilgridOptions::showExtractionVideo)
    .def_readwrite("minTagsForValidObs", &GridCalibrationTargetAprilgrid::AprilgridOptions::minTagsForValidObs)
    .def_readwrite("minBorderDistance", &GridCalibrationTargetAprilgrid::AprilgridOptions::minBorderDistance)
    .def_readwrite("maxSubpixDisplacement2", &GridCalibrationTargetAprilgrid::AprilgridOptions::maxSubpixDisplacement2)
    .def_readwrite("blackTagBorder", &GridCalibrationTargetAprilgrid::AprilgridOptions::blackTagBorder)
    .def_pickle(sm::python::pickle_suite<GridCalibrationTargetAprilgrid::AprilgridOptions>());

    class_<GridCalibrationTargetAprilgrid, bases<GridCalibrationTargetBase>,
    boost::shared_ptr<GridCalibrationTargetAprilgrid>, boost::noncopyable>(
    "GridCalibrationTargetAprilgrid",
    init<size_t, size_t, double, double, GridCalibrationTargetAprilgrid::AprilgridOptions>(
    "GridCalibrationTargetAprilgrid(size_t tagRows, size_t tagCols, double tagSize, double tagSpacing, AprilgridOptions options)"))
    .def(init<size_t, size_t, double, double>(
    "GridCalibrationTargetAprilgrid(size_t tagRows, size_t tagCols, double tagSize, double tagSpacing)"))
    .def(init<>("Do not use the default constructor. It is only necessary for the pickle interface"))
    .def_pickle(sm::python::pickle_suite<GridCalibrationTargetAprilgrid>());

    class_< MultipleTargetAprilGridDetector, boost::shared_ptr < MultipleTargetAprilGridDetector >, boost::noncopyable > (
            "MultipleTargetAprilGridDetector",
            init < boost::shared_ptr < CameraGeometryBase >, GridCalibrationTargetAprilgrid::Ptr, const int,
                    GridDetector::GridDetectorOptions >
                    ("MultipleTargetAprilGridDetector::MultipleTargetAprilGridDetector( boost::shared_ptr<CameraGeometryBase> geometry, GridCalibrationTargetAprilgrid::Ptr target, const int numTargets, GridDetector::GridDetectorOptions options)"))
            .def("initCameraGeometry", &MultipleTargetAprilGridDetector::initCameraGeometry)
            .def("initCameraGeometryFromObservation", &initCameraGeometryFromObservation)
            .def("initCameraGeometryFromObservations", &initCameraGeometryFromObservations)
            .def("geometry", &MultipleTargetAprilGridDetector::geometry)
            .def("target", &MultipleTargetAprilGridDetector::target)
            .def("findTarget", &findTarget3)
            .def("findTarget", &findTarget4)
            .def("findTargetNoTransformation", &findTargetNoTransformation3)
            .def("findTargetNoTransformation", &findTargetNoTransformation4)
            .def(init < boost::shared_ptr < CameraGeometryBase > , GridCalibrationTargetAprilgrid::Ptr, const int >
                         ("MultipleTargetAprilGridDetector::MultipleTargetAprilGridDetector( boost::shared_ptr<CameraGeometryBase> geometry, GridCalibrationTargetAprilgrid::Ptr target, const int numTargets)"))
            .def(init<>("Do not use the default constructor. It is only necessary for the pickle interface"))
            .def_pickle(sm::python::pickle_suite<MultipleTargetAprilGridDetector>());

}
