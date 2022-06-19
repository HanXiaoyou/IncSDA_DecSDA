function [accuracy,errorIndex,predictlabel]=NN(trainData,testData,trainLabel,testLabel)
%最近邻分类器(Nearest Neighbor, NN)
% output:
%     accuracy      ----识别精度
%     errorIndex    ----分类错误的样本索引
%%%给出识别率accuracy
distance=Distance(trainData,testData);%%此时就是欧式距离
%[neighborIds,distance] = kNearestNeighbors(trainData,testData,1);
rightNo=0;
wrongNo=0;
accuracy=0;
%%%%%%%%%%%%%%   m：训练集的行数，即人类数；n:测试集的行数，即人类数
[m,n]=size(distance);
errorIndex=[];
errorJudgeClass=[];
%%%%%%%%%%%%%%   判断测试集testData所属的类别
for i=1:n
    [ddistance,index]=sort(distance(:,i));
    if testLabel(i)==trainLabel(index(1))%%%最小距离，注意是两个等号
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