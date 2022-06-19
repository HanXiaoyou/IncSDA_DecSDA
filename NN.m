function [accuracy,errorIndex,predictlabel]=NN(trainData,testData,trainLabel,testLabel)
%����ڷ�����(Nearest Neighbor, NN)
% output:
%     accuracy      ----ʶ�𾫶�
%     errorIndex    ----����������������
%%%����ʶ����accuracy
distance=Distance(trainData,testData);%%��ʱ����ŷʽ����
%[neighborIds,distance] = kNearestNeighbors(trainData,testData,1);
rightNo=0;
wrongNo=0;
accuracy=0;
%%%%%%%%%%%%%%   m��ѵ����������������������n:���Լ�����������������
[m,n]=size(distance);
errorIndex=[];
errorJudgeClass=[];
%%%%%%%%%%%%%%   �жϲ��Լ�testData���������
for i=1:n
    [ddistance,index]=sort(distance(:,i));
    if testLabel(i)==trainLabel(index(1))%%%��С���룬ע���������Ⱥ�
        rightNo=rightNo+1;
    else 
        wrongNo= wrongNo+1;
        errorIndex(wrongNo,1)=i;
        errorJudgeClass(errorJudgeClass)=trainLabel(index(1));
    end
    predictlabel(:,i) = trainLabel(index(1));
end
accuracy = rightNo/n;
end