#ifndef KALIBR_PYTHONIMAGELIST_HPP
#define KALIBR_PYTHONIMAGELIST_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <utility>
#include <opencv2/core/eigen.hpp>
#include <sm/python/boost_serialization_pickle.hpp>


namespace aslam {
    namespace python {

        class PythonImageList {
        public:
            typedef std::vector <cv::Mat> ImageListType;
            typedef boost::shared_ptr <ImageListType> ImageListTypePtr;
            typedef Eigen::Matrix <boost::uint8_t, Eigen::Dynamic, Eigen::Dynamic> image_t;

            PythonImageList() : _images(new ImageListType) {};

            ~PythonImageList() {};

            void addImage(const image_t &image) {
                cv::Mat image_cv;
                eigen2cv(image, image_cv);
                _images->push_back(image_cv);
            }

            ImageListTypePtr getImages() {
                return _images;
            }

        private:
            //image lsit
            ImageListTypePtr _images;
        };
    }
}
#endif //KALIBR_PYTHONIMAGELIST_HPP
