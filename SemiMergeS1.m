function [IncS] = SemiMergeS1(data,indata,Y,Yindata,k)
%   Input:
%       data:
%           data         ------原始样本矩阵D×N1
%           indata       ------增量样本矩阵D×N2
%           Y            ------原始样本样本标签
%           Yindata      ------增量样本样本标签
%           k            ------k近邻参数
%   Output:
%       IncS      ------增量部分相似度矩阵S：N1×N2
% 2020 PWR

% 增加样本对原始样本kNN
N1 = size(data,2);  N2 = size(indata,2);
data = data - repmat(mean(data,2),1,N1);
indata = indata - repmat(mean(indata,2),1,N2);
IncS = zeros(N1,N2);
[neighborIds,neighborDistances,~] = kNearestNeighbors(data',indata',k+1);
sigma = mean(mean(neighborDistances(:,2:k+1)));%只用选择k近邻后的距离求sigma
neighborDistances = exp(-neighborDistances/(2*sigma^2));
for i= 1:N2
    IncS(neighborIds(i,:),i) = neighborDistances(i,:);%从第二列开始取，去掉自身的距离
end
%有标记样本
labelindice1 = find(Y~=0);
labelindice2 = find(Yindata~=0);
M1 = size(labelindice1,2);
M2 = size(labelindice2,2);
YLabel = Y(labelindice1);
YindataLabel = Yindata(labelindice2);
for i = 1:M1
    for j = 1:M2
        if YindataLabel(j) == YLabel(i)
            IncS(labelindice1(i),labelindice2(j)) = 1;
        end
    end
end
end


