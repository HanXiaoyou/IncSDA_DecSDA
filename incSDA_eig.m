function [result,t] = incSDA_eig(data,option,eigenThreshold)
% Input:
%     data:
%           X_init     ------初始训练样本集矩阵D×N
%           X_incr     ------增量训练样本集矩阵D×(n×Ni)
%           Y_LU_init     ------初始样本标签向量1×N
%           Y_LU_incr     ------增量样本标签矩阵n×N
%     option: 
%             n                   ------增量次数
%             p                   ------每次增量样本个数
%             DataBase            ------使用的数据集，用于最后给结果命名，输出结果到相对路径result文件夹中
% Output:
%       result:
%               V            ----降维向量D×M(M为降低后的维数)
%               index        ----原始和增量部分降维向量的索引
%               ClassCenter  ----每类中心D×C
%               t1           ----每次增量形成S所需的时间
%        t      ----运算时间(n+1)×1
X_init = data.X_init; Y_LU_init = data.Y_LU_init; 
result.index = zeros(option.n+1,2);%n为增量次数
t = zeros(option.n+1,1);%记录每次运算的时间，1次初始+n次增量
for nupdate=0:option.n
    % init proc
    if(nupdate==0)
        dataset_1 = X_init; label_1 = Y_LU_init;
        dataset_label_1 = X_init(:,label_1~=0);label_label_1 = label_1(1,label_1~=0);
        tic;
        [m1_1, M1_1, TeigenVect_1, TeigenVal_1] = GetStbar_eig(dataset_1,label_1,option,eigenThreshold);
        [m2_1, M2_1, BeigenVect_1, BeigenVal_1, samplePerClass_1, meanPerClass_1] = GetSb_eig(dataset_label_1, label_label_1,eigenThreshold);
        [DiscriminativeComponents,~] = getV_eig(TeigenVect_1, TeigenVal_1, BeigenVect_1, BeigenVal_1,eigenThreshold);
        t(nupdate+1,1)=toc;%记录运算时间
        result.V = DiscriminativeComponents;
        result.index(nupdate+1,:) = [1,size(DiscriminativeComponents,2)];
    else
        % for new data
        p = option.p;
        dataset_2 = data.X_incr(:,(nupdate-1)*p+1:nupdate*p); label_2 = data.Y_LU_incr(nupdate,:);
        dataset_label_2 = dataset_2(:,label_2~=0); label_label_2 = label_2(1,label_2~=0);
        tic
        [m1_2, M1_2, TeigenVect_2, TeigenVal_2] = GetStbar_eig(dataset_2,label_2, option,eigenThreshold);
        [m2_2, M2_2, BeigenVect_2, BeigenVal_2, samplePerClass_2, meanPerClass_2] = GetSb_eig(dataset_label_2, label_label_2, eigenThreshold);
        
        % update
        IncS = SemiMergeS1(dataset_1,dataset_2,label_1,label_2,5);%k近邻取5
         %assign values to SemiMergeSt_eig.m
        eigenmodel1.inMean_1=m1_1; eigenmodel1.inNS_1=M1_1; eigenmodel1.inEVect_1=TeigenVect_1; eigenmodel1.inEVal_1=TeigenVal_1;
        eigenmodel2.inMean_2=m1_2; eigenmodel2.inNS_2=M1_2; eigenmodel2.inEVect_2=TeigenVect_2; eigenmodel2.inEVal_2=TeigenVal_2;
         %merge process
        [outMean, outNSample, outEVect_t, outEVal_t] = SemiMergeSt_eig(eigenmodel1,eigenmodel2,dataset_1,dataset_2,IncS, eigenThreshold);
        [outMean2, outNSample2, outEVect_b, outEVal_b, outSamplePerClass, outMeanPerClass] = SemiMergeSb_eig(m2_1, M2_1, BeigenVect_1, BeigenVal_1, samplePerClass_1, meanPerClass_1, label_label_1, m2_2, M2_2, BeigenVect_2, BeigenVal_2, samplePerClass_2, meanPerClass_2, label_label_2, eigenThreshold);
        [DiscriminativeComponents,~] = getV_eig(outEVect_t, outEVal_t, outEVect_b, outEVal_b, eigenThreshold);
        
        t(nupdate+1,1)=toc;%记录运算时间
        
        % update variables
        
        m1_1 = outMean; M1_1 = outNSample; TeigenVect_1=outEVect_t; TeigenVal_1=outEVal_t;
        m2_2 = outMean2; M2_2 = outNSample2; BeigenVect_1=outEVect_b; BeigenVal_1=outEVal_b;
        samplePerClass_1=outSamplePerClass; meanPerClass_1=outMeanPerClass;
        label_label_1 = horzcat(label_label_1,label_label_2);
        label_1 = horzcat(label_1,label_2);
        dataset_1 = [dataset_1,dataset_2];
        result.V = [result.V,DiscriminativeComponents];
        result.index(nupdate+1,:) = [result.index(nupdate,2)+1,result.index(nupdate,2)+size(DiscriminativeComponents,2)];
        
    end
end
result.ClassCenter = repmat(m1_1,1,size(samplePerClass_1,2))+BeigenVect_1*meanPerClass_1;
end



    

    



