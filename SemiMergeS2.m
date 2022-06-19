function [IncS] = SemiMergeS2(YwholeSet,Yindata)
% 'IncCos':
%   Input:
%         YwholeSet    ------增量样本的样本标签(其中无标记样本标签记作0)
%         Yindata      ------增量样本的样本标签(其中增量数据为有标签数据)
%   Output:
%       IncS      ------增量部分相似度矩阵S：N1×N2
% PWR 2020

N1 = size(YwholeSet,2);  N2 = size(Yindata,2);
IncS = zeros(N1,N2);
for i = 1:N1
    for j = 1:N2
        if Yindata(j) == YwholeSet(i)
            IncS(i,j) = 1;
        end
    end
end
end


