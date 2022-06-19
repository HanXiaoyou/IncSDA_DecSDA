function [meanVector, noOfSample, eigenVectors, eigenValues] = SemiGetStD(featureVectorInCol,featureLabel, eigenThreshold)
% SemiGetStD指与ISSDA中的SemiGetSt有区别
% -----------------------------------------------------------------------
% Performing principal component analysis of total scatter matrix
%
% written by T-K. Kim and S-F. Wong, 2007
% -----------------------------------------------------------------------

% Input:
% featureVectorInCol: MxN matrix - each column is a feature vector of the raw dataset, M - dimension size, N - the number of sample
% featureLabel: 1xN vector - each cell stores the class label of a sample
% eigenThreshold: 1x1 value - the min value of eigenvalues to be selected
% Output:
% meanVector: Mx1 vector - the mean vector of the raw data
% noOfSample: 1x1 value - the number of sample in the raw data
% eigenVectors: MxR matrix - each column is a principal component, R is the reduced dimension
% eigenValues: RxR matrix - diagonal matrix storing the eigenvalues of the total scatter matrix
% -----------------------------------------------------------------------

noOfSample= size(featureVectorInCol,2); 

[principalComponents, tmpEigVal, meanVector, ~] = SemiPCAD(featureVectorInCol,featureLabel, eigenThreshold);

eigenVectors = principalComponents;
eigenValues = tmpEigVal;