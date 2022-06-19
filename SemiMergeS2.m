function [IncS] = SemiMergeS2(YwholeSet,Yindata)
% 'IncCos':
%   Input:
%         YwholeSet    ------����������������ǩ(�����ޱ��������ǩ����0)
%         Yindata      ------����������������ǩ(������������Ϊ�б�ǩ����)
%   Output:
%       IncS      ------�����������ƶȾ���S��N1��N2
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


