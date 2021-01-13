#ifndef ASLAM_MUTLIPLE_TARGET_APRIL_GRID_DETECTOR_HPP
#define ASLAM_MUTLIPLE_TARGET_APRIL_GRID_DETECTOR_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/export.hpp>
#include <sm/boost/serialization.hpp>
#include <sm/assert_macros.hpp>
#include <aslam/cameras/CameraGeometryBase.hpp>
#include <aslam/cameras/GridCalibrationTargetObservation.hpp>
#include <aslam/cameras/GridCalibrationTargetAprilgrid.hpp>
#include <aslam/cameras/GridDetector.hpp>

// April tags detector and various tag families
#include "apriltags/TagDetector.h"
//#include "apriltags/Tag16h5.h"
//#include "apriltags/Tag25h7.h"
//#include "apriltags/Tag25h9.h"
//#include "apriltags/Tag36h9.h"
#include "apriltags/Tag36h11.h"

namespace aslam {
    namespace cameras {

        class MultipleTargetAprilGridDetector {
        public:
            SM_DEFINE_EXCEPTION(Exception, std::runtime_error
            );

            /// \brief initialize based on grid geometry
            MultipleTargetAprilGridDetector(boost::shared_ptr <CameraGeometryBase> geometry,
                                            GridCalibrationTargetAprilgrid::Ptr target,
                                            const int numTargets,
                                            const GridDetector::GridDetectorOptions &options = GridDetector::GridDetectorOptions());

            virtual ~MultipleTargetAprilGridDetector();

            /// \brief initialize the detector
            void initializeDetector();

            /// \brief initialize the geometry for a known camera
            void initCameraGeometry(boost::shared_ptr <CameraGeometryBase> geometry);

            /// \brief initialize the geometry from one grid observation
            /// \return true if successful
            bool initCameraGeometryFromObservation(const cv::Mat &image);

            /// \brief initialize the geometry from a list grid observation
            /// \return true if successful
            bool initCameraGeometryFromObservations(boost::shared_ptr <std::vector<cv::Mat>> images_ptr);

            /// \brief get the underlying geometry
            boost::shared_ptr <CameraGeometryBase> geometry() const {
                return _geometry;
            };

            /// \brief get the underlying target
            GridCalibrationTargetBase::Ptr target() const {
                return _target;
            };

            /// \brief Find the target in the image. Return true on success.
            ///
            ///        If the intrinsics are not initialized (geometry pointer null),
            ///        they will be initialized.
            ///        This method will also estimate and fill in the transformation of the
            ///        camera with respect to the grid.
            bool findTarget(const cv::Mat &image, const aslam::Time &stamp,
                            std::vector <GridCalibrationTargetObservation> &outObservation) const;

            bool findTarget(const cv::Mat &image, std::vector <GridCalibrationTargetObservation> &outObservation) const;

            /// \brief Find the target but don't estimate the transformation.
            bool findTargetNoTransformation(const cv::Mat &image, const aslam::Time &stamp,
                                            std::vector <GridCalibrationTargetObservation> &outObservation) const;

            /// \brief Find the target but don't estimate the transformation.
            bool findTargetNoTransformation(const cv::Mat &image,
                                            std::vector <GridCalibrationTargetObservation> &outObservation) const;

            ///////////////////////////////////////////////////
            // Serialization support
            ///////////////////////////////////////////////////
            enum {
                CLASS_SERIALIZATION_VERSION = 1
            };

            BOOST_SERIALIZATION_SPLIT_MEMBER()

            /// \brief serialization contstructor (don't use this)
            MultipleTargetAprilGridDetector();

        protected:
            friend class boost::serialization::access;

            /// \brief Serialization support
            template<class Archive>
            void save(Archive &ar, const unsigned int /*version*/) const {
                ar << BOOST_SERIALIZATION_NVP(_geometry);
                ar << BOOST_SERIALIZATION_NVP(_target);
                ar << BOOST_SERIALIZATION_NVP(_options);
                ar << BOOST_SERIALIZATION_NVP(_numTargets);
            }

            template<class Archive>
            void load(Archive &ar, const unsigned int /*version*/) {
                ar >> BOOST_SERIALIZATION_NVP(_geometry);
                ar >> BOOST_SERIALIZATION_NVP(_target);
                ar >> BOOST_SERIALIZATION_NVP(_options);
                ar >> BOOST_SERIALIZATION_NVP(_numTargets);
                initializeDetector();
            }

        private:
            /// the camera geometry
            boost::shared_ptr <CameraGeometryBase> _geometry;

            /// \brief the calibration target
            GridCalibrationTargetAprilgrid::Ptr _target;

            /// \brief detector options
            GridDetector::GridDetectorOptions _options;

            int _numTargets;
        };

    }  // namespace cameras
}  // namespace aslam

SM_BOOST_CLASS_VERSION(aslam::cameras::MultipleTargetAprilGridDetector);

#endif /* ASLAM_MUTLIPLE_TARGET_APRIL_GRID_DETECTOR_HPP */
