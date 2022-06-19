function [accuracy,errorIndex,predictlabel] = TestEval(W,model,method)
% W ----投影向量 N×C
% method:
% Input:
% NNLu   ----考虑降维维数对识别率的影响，accuracy取最大值
% model：
%        X_train  ----训练集样本(label) D×N
%        X_Test  ----测试集样本D×Nt
%        Y_train ----训练集样本标签(label)
%        Y_Test  ----测试集样本标签
% Output:
%     accuracy   ----将按维度增的精确度储存在result中并将max的accuarcy储存成向量输出
% 核心思想：用W将测试样本降维后与降维后得训练样本作比较(欧氏距离),取最近距离得样本的标签作为测试得标签
% 将随维度变化的accuracy储存在相对路径为result的文件夹中

% NNCD ----使用Cai Deng的方法，同样也是最近邻
% Input:
% model：
%        X_Test       ----测试集样本D×Nt
%        Y_Test       ----测试集样本标签Nt×1
%        ClassCenter  ----训练集样本类中心矩阵D×C
%        ClassLabel   ----类别标签C×1
% Output:
%     accuracy      ----直接输出精确度向量
%     errorIndex    ----分类错误的样本索引

switch method
    case 'NNLu'
        high = size(W,2);
        accuracy = zeros(1, high); % 初始识别率
        for j = 1:high
            train = (W(:,1:j))' * model.X_train ;
            test = W(:,1:j)' * model.X_Test;
            trainLabel = model.Y_train;
            testLabel = model.Y_Test;
            [accuracy(j),errorIndex1] = NN(train',test',trainLabel,testLabel);%nearest neighbour分类器，行数为样本数
            %选择最大精度对应的错误样本索引
            if j > 1
                if  accuracy(j) >= accuracy(j-1)
                    errorIndex = errorIndex1;
                end
            else
                errorIndex = errorIndex1;
            end
        end
        accuracy = max(accuracy);
        %识别准确率结束
    case 'NNCD'
        ProXTest = W'*model.X_Test;
        D = EuDist2(ProXTest',(W'*model.ClassCenter)',0);
        [~,idx] = min(D,[],2);
        predictlabel = (model.ClassLabel(idx))';
        accuracy = 1 - length(find(predictlabel-model.Y_Test))/size(model.X_Test,2);
        errorIndex = find(predictlabel~=model.Y_Test);
end
end

