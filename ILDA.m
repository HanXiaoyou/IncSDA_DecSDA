function [result,t] = ILDA(data,option)
%ILDA 
% Input:
%       data:D��N
%             X_Train      ----ѵ����
%             Y_Train      ----ѵ������ǩ
%             init_ind     ----��ʼѵ��������
%             incr_ind     ----����ѵ��������
%             X_Test       ----������������
%             Y_Test       ----����������ǩ
%        option:
%              n                           ----��������
%              p                   ------ÿ��������������
%              eigenThreshold              ----PCA��ά��ֵ
%        method:
%              incr_in_train  ----����������ѵ������
%              incr_in_test   ----���������ڲ��Լ���
% Output:          
%      result:
%               V            ----��ά����D��M(MΪ���ͺ��ά��)
%               index        ----ԭʼ���������ֽ�ά����������
%               ClassCenter  ----ÿ������D��C
% written by T-K. Kim, 2007
% PWR 2020
X_init = data.X_init; Y_L_init = data.Y_L_init; 
result.index = zeros(option.n+1,2);%nΪ��������
t = zeros(option.n+1,1);%��¼ÿ�������ʱ�䣬1�γ�ʼ+n������
eigenThreshold = option.eigenThreshold;
for nupdate=0:option.n   
    if(nupdate==0)
        % init proc
        dataset_1 = X_init; label_1 = Y_L_init;
        tic;
        [m_1, M_1, TeigenVect_1, TeigenVal_1] = fGetStModel(dataset_1, eigenThreshold);
        [m_1, M_1, BeigenVect_1, BeigenVal_1, samplePerClass_1, meanPerClass_1] = fGetSbModel(dataset_1, label_1, eigenThreshold);
        [DiscriminativeComponents D] = fGetDiscriminativeComponents(TeigenVect_1, TeigenVal_1, BeigenVect_1, BeigenVal_1, eigenThreshold);   
        t(nupdate+1,1)=toc;%��¼����ʱ��
        result.V = DiscriminativeComponents;
        result.index(nupdate+1,:) = [1,size(DiscriminativeComponents,2)];
    else
        % for new data
        p = option.p;
        dataset_2 = data.X_incr(:,(nupdate-1)*p+1:nupdate*p); label_2 = data.Y_L_incr(nupdate,:);
        tic
        [m_2, M_2, TeigenVect_2, TeigenVal_2] = fGetStModel(dataset_2, eigenThreshold);
        [m_2, M_2, BeigenVect_2, BeigenVal_2, samplePerClass_2, meanPerClass_2] = fGetSbModel(dataset_2, label_2, eigenThreshold);
        % update
        [outMean, outNSample, outEVect_t, outEVal_t] = fMergeSt(m_1, M_1, TeigenVect_1, TeigenVal_1, m_2, M_2, TeigenVect_2, TeigenVal_2, eigenThreshold);
        [outMean, outNSample, outEVect_b, outEVal_b, outSamplePerClass, outMeanPerClass] = fMergeSb(m_1, M_1, BeigenVect_1, BeigenVal_1, samplePerClass_1, meanPerClass_1, label_1, m_2, M_2, BeigenVect_2, BeigenVal_2, samplePerClass_2, meanPerClass_2, label_2, eigenThreshold);
        [DiscriminativeComponents D] = fGetDiscriminativeComponents(outEVect_t, outEVal_t, outEVect_b, outEVal_b, eigenThreshold);
        t(nupdate+1,1)=toc;%��¼����ʱ��
        % update variables
        m_1 = outMean; M_1 = outNSample; TeigenVect_1=outEVect_t; TeigenVal_1=outEVal_t;
        BeigenVect_1=outEVect_b; BeigenVal_1=outEVal_b; 
        samplePerClass_1=outSamplePerClass; meanPerClass_1=outMeanPerClass;
        label_1 = horzcat(label_1,label_2);       
        result.V = [result.V,DiscriminativeComponents];
        result.index(nupdate+1,:) = [result.index(nupdate,2)+1,result.index(nupdate,2)+size(DiscriminativeComponents,2)];
    end    
end
        result.ClassCenter = repmat(m_1,1,size(samplePerClass_1,2))+BeigenVect_1*meanPerClass_1;
end

