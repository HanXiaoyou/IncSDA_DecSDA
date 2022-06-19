function [IncS] = SemiMergeS1(data,indata,Y,Yindata,k)
%   Input:
%       data:
%           data         ------ԭʼ��������D��N1
%           indata       ------������������D��N2
%           Y            ------ԭʼ����������ǩ
%           Yindata      ------��������������ǩ
%           k            ------k���ڲ���
%   Output:
%       IncS      ------�����������ƶȾ���S��N1��N2
% 2020 PWR

% ����������ԭʼ����kNN
N1 = size(data,2);  N2 = size(indata,2);
data = data - repmat(mean(data,2),1,N1);
indata = indata - repmat(mean(indata,2),1,N2);
IncS = zeros(N1,N2);
[neighborIds,neighborDistances,~] = kNearestNeighbors(data',indata',k+1);
sigma = mean(mean(neighborDistances(:,2:k+1)));%ֻ��ѡ��k���ں�ľ�����sigma
neighborDistances = exp(-neighborDistances/(2*sigma^2));
for i= 1:N2
    IncS(neighborIds(i,:),i) = neighborDistances(i,:);%�ӵڶ��п�ʼȡ��ȥ������ľ���
end
%�б������
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


