function [principalComponents, eigenValues, meanVector, projectedData] = SemiPCA(featureVectorInCol, eigenThreshold)
% -----------------------------------------------------------------------
% Low-dimensional batch eigen-computation when M>>N
%
% written by T-K, Kim and S-F. Wong, 2007
% -----------------------------------------------------------------------

% Input:
% featureVectorInCol: MxN matrix - input data, M=dimension, N=noOfSample
% eigenThreshold: 1x1 value - the min value of eigenvalues to be selected

% Output:
% principalComponents: MxR matrix - each column is a PC, R is the reduced dimension
% eigenValues: RxR - diagonal matrix storing the eigenvalues which are > eigenThreshold
% meanVector: Mx1 vector - the mean vector of the input data
% projectedData: RxN matrix - the projected data organised in column
% -----------------------------------------------------------------------
beta1 = 1e-6; beta2 = 1e-3;
[M,N] = size(featureVectorInCol);
meanVector = mean(featureVectorInCol, 2);
data = featureVectorInCol - repmat(meanVector, 1, N);

% 使用knn与高斯核结合计算laplacian矩阵
method = 'exp'; option.k = 5;
L = build_laplacian(data,method,option);

% % 使用cos计算laplacian矩阵
% method = 'cos'; option.k = 5;
% L = build_laplacian(featureVectorInCol,method,option);

% Component selection
% 直接使用svd分解计算
% normCovMatrix = (1-beta1)*(data*data')+beta1*data*L*data'+beta2*eye(M);
normCovMatrix = data*data'+beta1*data*L*data'+beta2*eye(M);
[V_NN, S_NN,~] = svd(normCovMatrix,0);


% Component selection
testRow = diag(S_NN);
testIdx = find(testRow>eigenThreshold);
V_NN = V_NN(:,testIdx);
S_NN = diag( testRow(testIdx) );

principalComponents = V_NN; % O(NR^2+MNR)
eigenValues = S_NN;
projectedData = principalComponents'*data;
end