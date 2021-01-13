#include <kalibr_errorterms/ScalarError.hpp>

namespace kalibr_errorterms
{

ScalarError::ScalarError(const double & measurement, const Eigen::Matrix<double,1,1> & invR, const aslam::backend::ScalarExpression & predictedMeasurement) :
        				_measurement(measurement), _predictedMeasurement(predictedMeasurement)
{
	setInvR( invR );

	aslam::backend::DesignVariable::set_t dvs;
	_predictedMeasurement.getDesignVariables(dvs);
	setDesignVariablesIterator(dvs.begin(), dvs.end());
	//Perform an initial error evaluation so that reasonable a priori errors can be retrieved.
	evaluateError();
}

ScalarError::~ScalarError() {}

double ScalarError::evaluateErrorImplementation()
{
	_evaluatedErrorTerm(0,0) = _predictedMeasurement.toScalar() - _measurement;
	setError(_evaluatedErrorTerm);

	return evaluateChiSquaredError();
}

void ScalarError::evaluateJacobiansImplementation(aslam::backend::JacobianContainer & _jacobians) const
{
	_predictedMeasurement.evaluateJacobians(_jacobians);
}

}