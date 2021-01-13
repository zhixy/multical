#ifndef KALIBR_IMU_CAM_SCALAR_ERROR_HPP
#define KALIBR_IMU_CAM_SCALAR_ERROR_HPP

#include<aslam/backend/ErrorTerm.hpp>
#include<aslam/backend/EuclideanExpression.hpp>
#include<aslam/backend/RotationExpression.hpp>
#include<aslam/backend/ScalarExpression.hpp>
#include<aslam/backend/TransformationExpression.hpp>

namespace kalibr_errorterms {

    class ScalarError : public aslam::backend::ErrorTermFs<1>
    {
    public: 
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
        ScalarError(const double & measurement, const Eigen::Matrix<double,1,1> & invR, 
            const aslam::backend::ScalarExpression & predictedMeasurement);
        ~ScalarError();


    protected:
        /// \brief evaluate the error term and return the weighted squared error e^T invR e
        virtual double evaluateErrorImplementation();

        /// \brief evaluate the jacobian
        virtual void evaluateJacobiansImplementation(aslam::backend::JacobianContainer & _jacobians) const;
    private:
        Eigen::Matrix<double,1,1> _evaluatedErrorTerm;
        double _measurement;

        aslam::backend::ScalarExpression _predictedMeasurement;
    };
}


#endif
