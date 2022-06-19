% Copy of decSDA.m
% Difference:Change the formation of the laplacian matrix L to the SRDA method
function [result,t] = decSDAL(data,option,eigenThreshold)
% Input:
%     data:
%           X_init     ------��ʼѵ������������D��N
%           X_decr     ------����ѵ������������D��(n��Ni)
%           Y_LU_init     ------��ʼ������ǩ����1��N
%           Y_LU_decr     ------����������ǩ����n��N
%     option: 
%             n                   ------��������
%             p                   ------ÿ�μ�����������
%             decr_ind            ------index of decremental data 1*N_dec
% Output:
%       result:
%               V            ----��ά����D��M(MΪ���ͺ��ά��)
%               index        ----ԭʼ�ͼ������ֽ�ά����������
%               ClassCenter  ----ÿ������D��C
%               t1           ----ÿ�μ����γ�S�����ʱ��
%        t      ----����ʱ��(n+1)��1
X_init = data.X_init; Y_LU_init = data.Y_LU_init; 
result.index = zeros(option.n+1,2);%nΪ��������
t = zeros(option.n+1,1);%��¼ÿ�������ʱ�䣬1�γ�ʼ+n�μ���
for nupdate=0:option.n
    % init proc
    if(nupdate==0)
        dataset_1 = X_init; label_1 = Y_LU_init;
        dataset_label_1 = X_init(:,label_1~=0);label_label_1 = label_1(1,label_1~=0);
        tic;
        [m1_1, M1_1, TeigenVect_1, TeigenVal_1] = GetStL(dataset_1,label_1,option,eigenThreshold);
        [m2_1, M2_1, BeigenVect_1, BeigenVal_1, samplePerClass_1, meanPerClass_1] = GetSb_eig(dataset_label_1, label_label_1,eigenThreshold);
        [DiscriminativeComponents,~] = getV_eig(TeigenVect_1, TeigenVal_1, BeigenVect_1, BeigenVal_1,eigenThreshold);
        t(nupdate+1,1)=toc;% record time
        result.V = DiscriminativeComponents;
        result.index(nupdate+1,:) = [1,size(DiscriminativeComponents,2)];
        result.ClassCenterI = repmat(m2_1,1,size(samplePerClass_1,2))+BeigenVect_1*meanPerClass_1;
    else
        % for deremental data
        p = option.p;
        dataset_2 = data.X_decr(:,(nupdate-1)*p+1:nupdate*p); label_2 = data.Y_LU_decr(nupdate,:);
        dataset_label_2 = dataset_2(:,label_2~=0); label_label_2 = label_2(1,label_2~=0);
        tic
        [m1_2, M1_2, TeigenVect_2, TeigenVal_2] = GetStL(dataset_2,label_2, option,eigenThreshold);
        [m2_2, M2_2, BeigenVect_2, BeigenVal_2, samplePerClass_2, meanPerClass_2] = GetSb_eig(dataset_label_2, label_label_2, eigenThreshold);
        
        % update
        dataset_3 = dataset_1; dataset_3(:,option.decr_ind) = []; % Xnew  
        label_3 = label_1; label_3(:,option.decr_ind) = []; % Ynew --delete wrong tag
        DecS = SemiMergeS1(dataset_2,dataset_3,label_2,label_3,5);%knn = 5; N_dec*N_new
        [outMean, outNSample, outEVect_t, outEVal_t] = SplitSt(m1_1, M1_1, TeigenVect_1, TeigenVal_1, m1_2, M1_2, TeigenVect_2, TeigenVal_2,DecS, eigenThreshold);
        [outMean2, outNSample2, outEVect_b, outEVal_b, outSamplePerClass, outMeanPerClass] = SplitSb(m2_1, M2_1, BeigenVect_1, BeigenVal_1, samplePerClass_1, meanPerClass_1, label_label_1, m2_2, M2_2, BeigenVect_2, BeigenVal_2, samplePerClass_2, meanPerClass_2, label_label_2, eigenThreshold);
        [DiscriminativeComponents,~] = getV_eig(outEVect_t, outEVal_t, outEVect_b, outEVal_b, eigenThreshold);
        
        t(nupdate+1,1)=toc;% record time
        
        % update variables
        
        m1_1 = outMean; M1_1 = outNSample; TeigenVect_1=outEVect_t; TeigenVal_1=outEVal_t;
        m2_2 = outMean2; M2_2 = outNSample2; BeigenVect_1=outEVect_b; BeigenVal_1=outEVal_b;
        samplePerClass_1=outSamplePerClass; meanPerClass_1=outMeanPerClass;
%         label_label_1 = horzcat(label_label_1,label_label_2);
%         label_1 = horzcat(label_1,label_2);
%         dataset_1 = [dataset_1,dataset_2];
        result.V = [result.V,DiscriminativeComponents];
        result.index(nupdate+1,:) = [result.index(nupdate,2)+1,result.index(nupdate,2)+size(DiscriminativeComponents,2)];
        
    end
end
% result.ClassCenter = repmat(m2_2,1,size(samplePerClass_1,2))+BeigenVect_1*meanPerClass_1;
result.ClassCenter = outMeanPerClass;
end



    

    



