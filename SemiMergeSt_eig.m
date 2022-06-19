function [outMean, outNSample, outEVect, outEVal] = SemiMergeSt_eig(eigenmodel1, eigenmodel2,X,Xinc,IncS, eigenThreshold)
% -----------------------------------------------------------------------
% Merging two "total scatter matrices" 
%
% The detailed algorithm here is the same as P.Hall and et al.'s method
% just except the covariance matrix is replaced by the scatter matrix.
% (P.Hall, D.Marshall and R.Martin, Merging and splitting eigenspace
% models, IEEE Trans. on PAMI, 2000.)
% -----------------------------------------------------------------------

% Input:
% eigenmodel1:
% inMean_1: Mx1 matrix - the mean vector of the old data 
% inNS_1: 1x1 value - the number of sample in the old data
% inEVect_1: MxR_1 matrix - each column is an eigenvector of the total scatter matrix of the old data, R_1 is the reduced dimension
% inEVal_1: R_1xR_1 matrix - diagonal matrix storing the eigenvalues of the total scatter matrix of the old data
% eigenmodel2:
% inMean_2: Mx1 matrix - the mean vector of the new data 
% inNS_2: 1x1 value - the number of sample in the new data
% inEVect_2: MxR_2 matrix - each column is an eigenvector of the total scatter matrix of the new data, R_2 is the reduced dimension
% inEVal_2: R_2xR_2 matrix - diagonal matrix storing the eigenvalues of the total scatter matrix of the new data
% X:  dxn1
% Xinc: dxn2
% IncS: N1xN2 - incremental similarity matrix
% Output:
% outMean: Mx1 vector - the updated mean vector of all raw data, M is the dimension
% outNSample: 1x1 value - the number of sample in the raw data
% outEVect: MxR matrix - each column is an eigenvector of the total scatter matrix, R is the reduced dimension
% outEVal: RxR matrix - diagonal matrix storing the eigenvalues of the total scatter matrix
% -----------------------------------------------------------------------
inMean_1 = eigenmodel1.inMean_1;
inNS_1 = eigenmodel1.inNS_1;
inEVect_1 = eigenmodel1.inEVect_1;
inEVal_1 = eigenmodel1.inEVal_1;

inNS_2 = eigenmodel2.inNS_2;
inEVect_2 = eigenmodel2.inEVect_2;
inEVal_2 = eigenmodel2.inEVal_2;
inMean_2 = eigenmodel2.inMean_2;

% updating global mean
outNSample = inNS_1 + inNS_2;
outMean = (inMean_1 * inNS_1 + inMean_2 * inNS_2 ) / outNSample;


% QR:
residueThreshold = 0.0001; 

GMatrix = inEVect_1' *inEVect_2; % O(R_1MR_2)
meanDiff = inMean_1-inMean_2; 
residue = inEVect_2 - inEVect_1 * (GMatrix); % O(MR_1R_2)
residueSumRow = sum(abs(residue),1);
pureResidue = residue(:,residueSumRow>residueThreshold);
meanResidue = meanDiff - inEVect_1 * (inEVect_1' * meanDiff); % O(MR_1)
meanResidueSumRow = sum(abs(meanResidue),1);
meanResidue = meanResidue(:,meanResidueSumRow>residueThreshold);

[OrthSubMatrix,upperTri] = qr([pureResidue meanResidue],0); % O(MR_2^2) OrthSubMatrix->MxR

% QR redundancy removal:
qrThreshold = 0.0001;
upperTriSum = sum(abs(upperTri),2);
upperTriRowIndex = find(upperTriSum>qrThreshold)';
OrthSubMatrix = OrthSubMatrix(:,upperTriRowIndex);

%SVD
TMatrix = OrthSubMatrix' * inEVect_2; % O(MRR_2) TMatrix->RxR_2
mGMatrix = inEVect_1'*meanDiff; % O(MR_1) mGMatrix->R_1x1
mTMatrix = OrthSubMatrix'*meanDiff; % O(MR) mTMatrix->Rx1

reducedDim = size(inEVect_1, 2) + size(OrthSubMatrix,2);
fprintf('The dimension of merged problem is %8.5f\n',reducedDim)
beta1 = 1e-6; beta2 = 1e-3;

term1 = zeros(reducedDim, reducedDim);
detaD1 = diag(sum(IncS,2));
XTPhi = X'*[inEVect_1 OrthSubMatrix]; %n_1x(rt1+rt2+1)
term1(1:size(inEVect_1,2),1:size(inEVect_1,2)) = inEVal_1;
term1 = term1 + beta1*XTPhi'*detaD1*XTPhi;
detaD2 = diag(sum(IncS,1));
XincTPhi = Xinc'*[inEVect_1 OrthSubMatrix]; %n_2x(rt1+rt2+1)
term2 = [GMatrix*inEVal_2*GMatrix' GMatrix*inEVal_2*TMatrix'; TMatrix*inEVal_2*GMatrix' TMatrix*inEVal_2*TMatrix'];%*(inNS_2/outNSample); % O( max(R_1, R_2, R)^3) 
term2 = term2 + beta1*XincTPhi'*detaD2*XincTPhi;

term3 = [mGMatrix*mGMatrix' mGMatrix*mTMatrix'; mTMatrix*mGMatrix' mTMatrix*mTMatrix']  *((inNS_1*inNS_2)*(1-beta1)/(outNSample));%*outNSample)); % O( max(R_1, R)^2)

term4 = XTPhi'* IncS * XincTPhi;
term4 = -beta1*term4 - beta1*term4';

CompositeMatrix = term1 + term2 + term3 + term4 - beta2*speye(reducedDim); %  CompositeMatrix->RxR

CompositeMatrix = max(CompositeMatrix,CompositeMatrix');%保对称
[U, Sigma] = eig(CompositeMatrix);
[~, idx] = sort( abs(diag(Sigma)), 'descend');%取最大的d个特征值，按降序排列
U = U(:,idx);
Sigma = Sigma(idx,idx);

testRow = diag(Sigma);
testIdx = find(testRow>eigenThreshold);

U = U(:,testIdx);
Sigma = diag(testRow(testIdx));

outEVal = Sigma;
outEVect = [inEVect_1 OrthSubMatrix] * U; % O(M (R_1+R) R)

