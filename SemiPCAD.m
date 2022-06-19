function [principalComponents, eigenValues, meanVector, projectedData] = SemiPCAD(featureVectorInCol,featureLabel, eigenThreshold)
% -----------------------------------------------------------------------
% Low-dimensional batch eigen-computation when M>>N
%
% written by T-K, Kim and S-F. Wong, 2007
% -----------------------------------------------------------------------

% Input:
% featureVectorInCol: MxN matrix - input data, M=dimension, N=noOfSample
% featureLabel: 1xN vector - each cell stores the class label of a sample
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
L = build_laplacianD(data,featureLabel,method,option);
L = sparse(L);%稀疏存储
% % 使用cos计算laplacian矩阵
% method = 'cos'; option.k = 5;
% L = build_laplacian(featureVectorInCol,method,option);

% Component selection 
if M>=N
    % 使用QR分解计算
    [Q,R] = qr(data,0);
    normCovMatrix = (1-beta1)*(R*R')+R*L*R'+beta2*speye(N);
    [V_NN, S_NN,~] = svd(normCovMatrix,0);
    testRow = diag(S_NN);
    testIdx = find(testRow>eigenThreshold);
    V_NN = V_NN(:,testIdx);
    S_NN = diag( testRow(testIdx) );
    
%     if size(V_NN,2) > N
%         fprintf('Need to change eigenThreshold larger than',testRow(N))
%     end
        
    principalComponents = Q*V_NN;
    eigenValues = S_NN;
    projectedData = principalComponents'*data;
else
    % 直接使用svd分解计算
    normCovMatrix = (1-beta1)*(data*data')+beta1*data*L*data'+beta2*eye(M);
    [V_NN, S_NN,~] = svd(normCovMatrix,0);
    
    
    % Component selection
    testRow = diag(S_NN);
    testIdx = find(testRow>eigenThreshold);
    V_NN = V_NN(:,testIdx);
    S_NN = diag( testRow(testIdx) );
    
    % dimension restriction & eigenThreshold choosing
%     if size(V_NN,2) > N
%         fprintf('Need to change eigenThreshold larger than',testRow(N))
%     end
    
    principalComponents = V_NN; % O(NR^2+MNR)
    eigenValues = sparse(S_NN);
    projectedData = principalComponents'*data;
end