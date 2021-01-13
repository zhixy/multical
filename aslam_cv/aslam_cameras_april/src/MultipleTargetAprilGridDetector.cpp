#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sm/logging.hpp>
#include <aslam/cameras/MultipleTargetAprilGridDetector.hpp>

namespace aslam {
    namespace cameras {

//serialization constructor (don't use!)
        MultipleTargetAprilGridDetector::MultipleTargetAprilGridDetector() {}

        MultipleTargetAprilGridDetector::MultipleTargetAprilGridDetector(
                boost::shared_ptr <CameraGeometryBase> geometry,
                GridCalibrationTargetAprilgrid::Ptr target,
                const int numTargets,
                const GridDetector::GridDetectorOptions &options)
                : _geometry(geometry),
                  _target(target),
                  _numTargets(numTargets),
                  _options(options) {
            SM_ASSERT_TRUE(Exception, _geometry.get() != NULL,
                           "Unable to initialize with null camera geometry");
            SM_ASSERT_TRUE(Exception, _target.get() != NULL,
                           "Unable to initialize with null calibration target");

            initializeDetector();
        }

        void MultipleTargetAprilGridDetector::initializeDetector() {
            if (_options.plotCornerReprojection) {
                cv::namedWindow("Corner reprojection", cv::WINDOW_NORMAL);
            }
            if (_target->getOptions().showExtractionVideo) {

                cv::namedWindow("Aprilgrid: Tag detection", cv::WINDOW_NORMAL);
                cv::namedWindow("Aprilgrid: Tag corners", cv::WINDOW_NORMAL);
            }
        }

        MultipleTargetAprilGridDetector::~MultipleTargetAprilGridDetector() {

        }

        void MultipleTargetAprilGridDetector::initCameraGeometry(boost::shared_ptr <CameraGeometryBase> geometry) {
            SM_ASSERT_TRUE(Exception, geometry.get() != NULL, "Unable to initialize with null camera geometry");
            _geometry = geometry;
        }

        bool MultipleTargetAprilGridDetector::initCameraGeometryFromObservation(const cv::Mat &image) {
            boost::shared_ptr <std::vector<cv::Mat>> images_ptr = boost::make_shared < std::vector < cv::Mat >> ();
            images_ptr->push_back(image);

            return initCameraGeometryFromObservations(images_ptr);
        }

        bool MultipleTargetAprilGridDetector::initCameraGeometryFromObservations(
                boost::shared_ptr <std::vector<cv::Mat>> images_ptr) {

            std::vector <cv::Mat> &images = *images_ptr;

            SM_DEFINE_EXCEPTION(Exception, std::runtime_error);
            SM_ASSERT_TRUE(Exception, images.size() != 0, "Need min. one image");

            std::vector <GridCalibrationTargetObservation> observations;

            for (unsigned int i = 0; i < images.size(); i++) {
                std::vector<GridCalibrationTargetObservation> obs;

                //detect calibration target
                bool success = findTargetNoTransformation(images[i], obs);

                if (obs.empty() || obs[0].targetId()!=0) continue;

                //delete image copy (save memory)
                obs[0].clearImage();

                //append
                if (success)
                    observations.push_back(obs[0]);
            }

            //initialize the intrinsics
            if (observations.size() > 0)
                return _geometry->initializeIntrinsics(observations);

            return false;
        }

        bool MultipleTargetAprilGridDetector::findTarget(const cv::Mat &image,
                                                         vector <GridCalibrationTargetObservation> &outObservation) const {
            return findTarget(image, aslam::Time(0, 0), outObservation);
        }

        bool MultipleTargetAprilGridDetector::findTargetNoTransformation(const cv::Mat &image, const aslam::Time &stamp,
                                                                         vector <GridCalibrationTargetObservation> &outObservation) const {
            bool success = true;

            // Set the image, target, and timestamp regardless of success.
            GridCalibrationTargetObservation tmp(_target, image);
            tmp.setTime(stamp);
            outObservation.resize(_numTargets, tmp);
            for(int i = 0; i < _numTargets; i++){
                outObservation[i].setTargetId(i);
            }

            // detect the tags
            std::vector <AprilTags::TagDetection> detections = _target->getTagDetector()->extractTags(image);

            /* handle the case in which a tag is identified but not all tag
             * corners are in the image (all data bits in image but border
             * outside). tagCorners should still be okay as apriltag-lib
             * extrapolates them, only the subpix refinement will fail
             */

            //min. distance [px] of tag corners from image border (tag is not used if violated)
            std::vector<AprilTags::TagDetection>::iterator iter = detections.begin();
            for (iter = detections.begin(); iter != detections.end();) {
                // check all four corners for violation
                bool remove = false;

                for (int j = 0; j < 4; j++) {
                    remove |= iter->p[j].first < _target->getOptions().minBorderDistance;
                    remove |= iter->p[j].first > (float) (image.cols) - _target->getOptions().minBorderDistance;  //width
                    remove |= iter->p[j].second < _target->getOptions().minBorderDistance;
                    remove |= iter->p[j].second > (float) (image.rows) - _target->getOptions().minBorderDistance;  //height
                }

                //also remove tags that are flagged as bad
                if (iter->good != 1)
                    remove |= true;

                //also remove if the tag ID is out-of-range for this grid (faulty detection)
                if (iter->id >= _numTargets * _target->size() / 4)
                    remove |= true;

                // delete flagged tags
                if (remove) {
                    SM_DEBUG_STREAM("Tag with ID " << iter->id
                                                   << " is only partially in image (corners outside) and will be removed from the TargetObservation.\n");

                    // delete the tag and advance in list
                    iter = detections.erase(iter);
                } else {
                    //advance in list
                    ++iter;
                }
            }

            //did we find enough tags?
            if (detections.size() < _target->getOptions().minTagsForValidObs) {
                success = false;

                //immediate exit if we dont need to show video for debugging...
                //if video is shown, exit after drawing video...
                if (!_target->getOptions().showExtractionVideo)
                    return success;
            }

            //sort detections by tagId
            std::sort(detections.begin(), detections.end(),
                      AprilTags::TagDetection::sortByIdCompare);

            // check for duplicate tagIds (--> if found: wild Apriltags in image not belonging to calibration target)
            // (only if we have more than 1 tag...)
            if (detections.size() > 1) {
                for (unsigned i = 0; i < detections.size() - 1; i++)
                    if (detections[i].id == detections[i + 1].id) {

                        cv::Mat imageCopy = image.clone();
                        cv::cvtColor(imageCopy, imageCopy, CV_GRAY2RGB);

                        int duplicated_id = detections[i].id;
                        //mark all duplicate tags in image
                        for (auto iter = detections.begin()+i; iter != detections.end();) {
                            if (iter->id == duplicated_id) {
                                iter->draw(imageCopy);
                                detections.erase(iter);
                            }
                            else{
                                iter++;
                            }
                        }

                        cv::putText(imageCopy, "Duplicate Apriltags detected. Hide them.",
                                    cv::Point(50, 50), CV_FONT_HERSHEY_SIMPLEX, 0.8,
                                    CV_RGB(255, 0, 0), 2, 8, false);
                        std::string file_name = "Duplicate_Apriltags_"+std::to_string(stamp.toSec())+".jpg";
                        cv::imwrite(file_name, imageCopy);  // OpenCV call

                        SM_FATAL_STREAM(
                                "\n[ERROR]: Found apriltag not belonging to calibration board. "
                                "Has erased these duplicated tags, please check the image file " + file_name +
                                " for the tag and hide it.\n");

                    }
            }

            // convert corners to cv::Mat (4 consecutive corners form one tag)
            /// point ordering here
            ///          11-----10  15-----14
            ///          | TAG 2 |  | TAG 3 |
            ///          8-------9  12-----13
            ///          3-------2  7-------6
            ///    y     | TAG 0 |  | TAG 1 |
            ///   ^      0-------1  4-------5
            ///   |-->x
            cv::Mat tagCorners(4 * detections.size(), 2, CV_32F);

            for (unsigned i = 0; i < detections.size(); i++) {
                for (unsigned j = 0; j < 4; j++) {
                    tagCorners.at<float>(4 * i + j, 0) = detections[i].p[j].first;
                    tagCorners.at<float>(4 * i + j, 1) = detections[i].p[j].second;
                }
            }

            //store a copy of the corner list before subpix refinement
            cv::Mat tagCornersRaw = tagCorners.clone();

            //optional subpixel refinement on all tag corners (four corners each tag)
            if (_target->getOptions().doSubpixRefinement && success)
                cv::cornerSubPix(
                        image, tagCorners, cv::Size(2, 2), cv::Size(-1, -1),
                        cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

            if (_target->getOptions().showExtractionVideo) {
                //image with refined (blue) and raw corners (red)
                cv::Mat imageCopy1 = image.clone();
                cv::cvtColor(imageCopy1, imageCopy1, CV_GRAY2RGB);
                for (unsigned i = 0; i < detections.size(); i++)
                    for (unsigned j = 0; j < 4; j++) {
                        //raw apriltag corners
                        //cv::circle(imageCopy1, cv::Point2f(detections[i].p[j].first, detections[i].p[j].second), 2, CV_RGB(255,0,0), 1);

                        //subpixel refined corners
                        cv::circle(
                                imageCopy1,
                                cv::Point2f(tagCorners.at<float>(4 * i + j, 0),
                                            tagCorners.at<float>(4 * i + j, 1)),
                                3, CV_RGB(0, 0, 255), 1);

                        if (!success)
                            cv::putText(imageCopy1, "Detection failed! (frame not used)",
                                        cv::Point(50, 50), CV_FONT_HERSHEY_SIMPLEX, 0.8,
                                        CV_RGB(255, 0, 0), 3, 8, false);
                    }

                cv::imshow("Aprilgrid: Tag corners", imageCopy1);  // OpenCV call
                cv::waitKey(1);

                /* copy image for modification */
                cv::Mat imageCopy2 = image.clone();
                cv::cvtColor(imageCopy2, imageCopy2, CV_GRAY2RGB);
                /* highlight detected tags in image */
                for (unsigned i = 0; i < detections.size(); i++) {
                    detections[i].draw(imageCopy2);

                    if (!success)
                        cv::putText(imageCopy2, "Detection failed! (frame not used)",
                                    cv::Point(50, 50), CV_FONT_HERSHEY_SIMPLEX, 0.8,
                                    CV_RGB(255, 0, 0), 3, 8, false);
                }

                cv::imshow("Aprilgrid: Tag detection", imageCopy2);  // OpenCV call
                cv::waitKey(1);

                //if success is false exit here (delayed exit if _options.showExtractionVideo=true for debugging)
                if (!success)
                    return success;
            }

            //insert the observed points into the correct location of the grid point array
            /// point ordering
            ///          12-----13  14-----15
            ///          | TAG 2 |  | TAG 3 |
            ///          8-------9  10-----11
            ///          4-------5  6-------7
            ///    y     | TAG 0 |  | TAG 1 |
            ///   ^      0-------1  2-------3
            ///   |-->x

            int tagsEachTarget = _target->size() / 4;
            unsigned int cols = _target->cols();
//            outCornerObserved.resize(size(), false);
//            outImagePoints.resize(size(), 2);

            for (unsigned int i = 0; i < detections.size(); i++) {
                // get the tag id
                unsigned int tagId = detections[i].id;
                int targetId = tagId / tagsEachTarget;
                tagId %= tagsEachTarget;
                // calculate the grid idx for all four tag corners given the tagId and cols
                unsigned int baseId = (int) (tagId / (cols / 2)) * cols * 2
                                      + (tagId % (cols / 2)) * 2;
                unsigned int pIdx[] = {baseId, baseId + 1, baseId + cols + 1, baseId + cols};

                // add four points per tag
                for (int j = 0; j < 4; j++) {
                    //refined corners
                    double corner_x = tagCorners.row(4 * i + j).at<float>(0);
                    double corner_y = tagCorners.row(4 * i + j).at<float>(1);

                    //raw corners
                    double cornerRaw_x = tagCornersRaw.row(4 * i + j).at<float>(0);
                    double cornerRaw_y = tagCornersRaw.row(4 * i + j).at<float>(1);

                    //only add point if the displacement in the subpixel refinement is below a given threshold
                    double subpix_displacement_squarred = (corner_x - cornerRaw_x)
                                                          * (corner_x - cornerRaw_x)
                                                          + (corner_y - cornerRaw_y) * (corner_y - cornerRaw_y);

                    if (subpix_displacement_squarred <= _target->getOptions().maxSubpixDisplacement2) {
                        outObservation[targetId].updateImagePoint(pIdx[j], Eigen::Vector2d(corner_x, corner_y));
                    } else {
                        SM_DEBUG_STREAM("Subpix refinement failed for point: " << pIdx[j] << " with displacement: "
                                                                               << sqrt(subpix_displacement_squarred)
                                                                               << "(point removed) \n");
                    }
                }
            }

            for(int i =0; i < outObservation.size();)
            {
                if(outObservation[i].numberSuccessfulObservation() < _target->getOptions().minTagsForValidObs)
                {
                    outObservation.erase(outObservation.begin() + i);
                }
                else
                {
                    i++;
                }
            }
            return outObservation.size();
        }

        bool MultipleTargetAprilGridDetector::findTarget(const cv::Mat &image, const aslam::Time &stamp,
                                                         vector <GridCalibrationTargetObservation> &outObservation) const {
            sm::kinematics::Transformation trafo;

            // find calibration target corners
            bool success = findTargetNoTransformation(image, stamp, outObservation);

            if(success)
            {
                for(auto& outObs: outObservation)
                {
                    // also estimate the transformation:
                    if(_geometry->estimateTransformation(outObs, trafo))
                    {
                        success = true;
                        outObs.set_T_t_c(trafo);
                        //remove corners with a reprojection error above a threshold
                        //(remove detection outliers)
                        if (_options.filterCornerOutliers) {
                            //calculate reprojection errors
                            std::vector <cv::Point2f> corners_reproj;
                            std::vector <cv::Point2f> corners_detected;
                            outObs.getCornerReprojection(_geometry, corners_reproj);
                            unsigned int numCorners = outObs.getCornersImageFrame(corners_detected);

                            //calculate error norm
                            Eigen::MatrixXd reprojection_errors_norm = Eigen::MatrixXd::Zero(numCorners, 1);

                            for (unsigned int i = 0; i < numCorners; i++) {
                                cv::Point2f reprojection_err = corners_detected[i] - corners_reproj[i];

                                reprojection_errors_norm(i, 0) = sqrt(reprojection_err.x * reprojection_err.x +
                                                                      reprojection_err.y * reprojection_err.y);
                            }

                            //calculate statistics
                            double mean = reprojection_errors_norm.mean();
                            double std = 0.0;
                            for (unsigned int i = 0; i < numCorners; i++) {
                                double temp = reprojection_errors_norm(i, 0) - mean;
                                std += temp * temp;
                            }
                            std /= (double) numCorners;
                            std = sqrt(std);

                            //disable outlier corners
                            std::vector<unsigned int> cornerIdx;
                            outObs.getCornersIdx(cornerIdx);

                            unsigned int removeCount = 0;
                            for (unsigned int i = 0; i < corners_detected.size(); i++) {
                                if (reprojection_errors_norm(i, 0) > mean + _options.filterCornerSigmaThreshold * std &&
                                    reprojection_errors_norm(i, 0) > _options.filterCornerMinReprojError) {
                                    outObs.removeImagePoint(cornerIdx[i]);
                                    removeCount++;
                                    SM_DEBUG_STREAM(
                                            "removed target point with reprojection error of " << reprojection_errors_norm(i, 0)
                                                                                               << " (mean: " << mean << ", std: "
                                                                                               << std << ")\n";);
                                }
                            }

                            if (removeCount > 0)
                                SM_DEBUG_STREAM("removed " << removeCount << " of " << numCorners
                                                           << " calibration target corner outliers\n";);
                        }
                    }
                    else
                        SM_DEBUG_STREAM("estimateTransformation() failed");
                }
            }

            // show plot of reprojected corners
            if (_options.plotCornerReprojection) {
                cv::Mat imageCopy1 = image.clone();
                cv::cvtColor(imageCopy1, imageCopy1, CV_GRAY2RGB);

                if (success) {
                    for(const auto& outObs: outObservation) {
                        //calculate reprojection
                        std::vector <cv::Point2f> reprojs;
                        outObs.getCornerReprojection(_geometry, reprojs);

                        for (unsigned int i = 0; i < reprojs.size(); i++)
                            cv::circle(imageCopy1, reprojs[i], 3, CV_RGB(255, 0, 0), 1);
                    }

                } else {
                    cv::putText(imageCopy1, "Detection failed! (frame not used)",
                                cv::Point(50, 50), CV_FONT_HERSHEY_SIMPLEX, 0.8,
                                CV_RGB(255, 0, 0), 3, 8, false);
                }

                cv::imshow("Corner reprojection", imageCopy1);  // OpenCV call
                if (_options.imageStepping) {
                    cv::waitKey(0);
                } else {
                    cv::waitKey(1);
                }
            }

            return success;
        }

/// \brief Find the target but don't estimate the transformation.
        bool MultipleTargetAprilGridDetector::findTargetNoTransformation(const cv::Mat &image,
                                                                         vector <GridCalibrationTargetObservation> &outObservation) const {
            return findTargetNoTransformation(image, aslam::Time(0, 0), outObservation);
        }

    }  // namespace cameras
}  // namespace aslam

//export explicit instantions for all included archives
#include <sm/boost/serialization.hpp>
#include <boost/serialization/export.hpp>

BOOST_CLASS_EXPORT_IMPLEMENT(aslam::cameras::MultipleTargetAprilGridDetector);
