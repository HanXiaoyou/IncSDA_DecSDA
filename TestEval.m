function [accuracy,errorIndex,predictlabel] = TestEval(W,model,method)
% W ----ͶӰ���� N��C
% method:
% Input:
% NNLu   ----���ǽ�άά����ʶ���ʵ�Ӱ�죬accuracyȡ���ֵ
% model��
%        X_train  ----ѵ��������(label) D��N
%        X_Test  ----���Լ�����D��Nt
%        Y_train ----ѵ����������ǩ(label)
%        Y_Test  ----���Լ�������ǩ
% Output:
%     accuracy   ----����ά�����ľ�ȷ�ȴ�����result�в���max��accuarcy������������
% ����˼�룺��W������������ά���뽵ά���ѵ���������Ƚ�(ŷ�Ͼ���),ȡ�������������ı�ǩ��Ϊ���Եñ�ǩ
% ����ά�ȱ仯��accuracy���������·��Ϊresult���ļ�����

% NNCD ----ʹ��Cai Deng�ķ�����ͬ��Ҳ�������
% Input:
% model��
%        X_Test       ----���Լ�����D��Nt
%        Y_Test       ----���Լ�������ǩNt��1
%        ClassCenter  ----ѵ�������������ľ���D��C
%        ClassLabel   ----����ǩC��1
% Output:
%     accuracy      ----ֱ�������ȷ������
%     errorIndex    ----����������������

switch method
    case 'NNLu'
        high = size(W,2);
        accuracy = zeros(1, high); % ��ʼʶ����
        for j = 1:high
            train = (W(:,1:j))' * model.X_train ;
            test = W(:,1:j)' * model.X_Test;
            trainLabel = model.Y_train;
            testLabel = model.Y_Test;
            [accuracy(j),errorIndex1] = NN(train',test',trainLabel,testLabel);%nearest neighbour������������Ϊ������
            %ѡ����󾫶ȶ�Ӧ�Ĵ�����������
            if j > 1
                if  accuracy(j) >= accuracy(j-1)
                    errorIndex = errorIndex1;
                end
            else
                errorIndex = errorIndex1;
            end
        end
        accuracy = max(accuracy);
        %ʶ��׼ȷ�ʽ���
    case 'NNCD'
        ProXTest = W'*model.X_Test;
        D = EuDist2(ProXTest',(W'*model.ClassCenter)',0);
        [~,idx] = min(D,[],2);
        predictlabel = (model.ClassLabel(idx))';
        accuracy = 1 - length(find(predictlabel-model.Y_Test))/size(model.X_Test,2);
        errorIndex = find(predictlabel~=model.Y_Test);
end
end

