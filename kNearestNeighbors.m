function [neighborIds neighborDistances,Dist] = kNearestNeighbors(dataMatrix,queryMatrix,k)
%--------------------------------------------------------------------------
% Program to find the k - nearest neighbors (kNN) within a set of points. 
% Distance metric used: Euclidean distance
% 
% Usage:
% [neighbors distances] = kNearestNeighbors(dataMatrix, queryMatrix, k);
% dataMatrix  (N x D) - N vectors with dimensionality D (within which we search for the nearest neighbors)
% queryMatrix (M x D) - M query vectors with dimensionality D
% k           (1 x 1) - Number of nearest neighbors desired
% Output:
%       neighborIds          ---- M x k
%       neighborDistances    ---- M x k
%
% Example:
% a = [1 1; 2 2; 3 2; 4 4; 5 6];
% b = [1 1; 2 1; 6 2];
% [neighbors distances] = kNearestNeighbors(a,b,2);
% 
% Output:
% neighbors =
%      1     2
%      1     2
%      4     3
% 
% distances =
%          0    1.4142
%     1.0000    1.0000
%     2.8284    3.0000
% Dist   -----储存整个距离矩阵
%--------------------------------------------------------------------------


neighborIds = zeros(size(queryMatrix,1),k);
neighborDistances = neighborIds;

numDataVectors = size(dataMatrix,1);
numQueryVectors = size(queryMatrix,1);
Dist = zeros(size(dataMatrix,1),size(queryMatrix,1));%为距离矩阵划分出空间

%% % 参考Deng Cai的欧式距离的形成
Dist = EuDist2(dataMatrix,queryMatrix,0);%欧氏距离不需要开根号
[sortval,sortpos] = sort(Dist,'ascend');
neighborIds = (sortpos(1:k,:))';
neighborDistances = (sortval(1:k,:))';

% %% % 参考Lu师姐的kNearestNeighbors
% for i=1:numQueryVectors
%     dist = sum((repmat(queryMatrix(i,:),numDataVectors,1)-dataMatrix).^2,2);
%     Dist(i,:) = dist;%储存整个距离矩阵
%     [sortval,sortpos] = sort(dist,'ascend');%按升序排列
%     neighborIds(i,:) = sortpos(1:k);
%     %neighborDistances(i,:) = sqrt(sortval(1:k));
%     neighborDistances(i,:) = sortval(1:k);
% end
