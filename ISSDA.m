function[result,t] = ISSDA(X_Unlable,X_Train,Y_Train,Ninit,Nincr,option)
% Input:
%     X_Train     ------训练样本集矩阵D×N
%     Y_Train     ------训练集标签向量1×N
%     Ninit       ------初始训练集个数
%     Nincr       ------每次增量训练集个数
%     option:      
%             n          ------增量次数
%             DataBase   ------使用的数据集，用于最后给结果命名，输出结果到相对路径result文件夹中
% Output:
%       result:
%               V            ----降维向量D×M(M为降低后的维数)
%               index        ----原始和增量部分降维向量的索引
%               ClassCenter  ----每类中心D×C
%        t      ----运算时间(n+1)×1


t = zeros(option.n+1,1);%记录每次运算的时间，1次初始+n次增量
for nupdate=0:option.n
    eigenThreshold = 0.01;
    if(nupdate==0)
        % init proc
        dataset_1 = X_Unlable; dataset_label_1 = X_Train(:,1:Ninit); label_label_1 = Y_Train(1:Ninit); 
        tic;
        [m1_1, M1_1, TeigenVect_1, TeigenVal_1] = SemiGetSt(dataset_1, eigenThreshold);
        [m2_1, M2_1, BeigenVect_1, BeigenVal_1, samplePerClass_1, meanPerClass_1] = SemiGetSb(dataset_label_1, label_label_1, eigenThreshold);
        [DiscriminativeComponents,~] = fGetDiscriminativeComponents(TeigenVect_1, TeigenVal_1, BeigenVect_1, BeigenVal_1,eigenThreshold);     
        t(nupdate+1,1)=toc;%记录运算时间
        result.V = DiscriminativeComponents;
        result.index(nupdate+1,:) = [1,size(DiscriminativeComponents,2)];
    else
        % for new data
        tic;
        dataset_label_2 = X_Train(:,(Ninit+1+(nupdate-1)*Nincr):(Ninit+nupdate*Nincr)); label_label_2 = Y_Train((Ninit+1+(nupdate-1)*Nincr):(Ninit+nupdate*Nincr)); 
        [m2_2, M2_2, BeigenVect_2, BeigenVal_2, samplePerClass_2, meanPerClass_2] = SemiGetSb(dataset_label_2, label_label_2, eigenThreshold);
        % update
        [outMean2, outNSample2, outEVect_b, outEVal_b, outSamplePerClass, outMeanPerClass] = SemiMergeSb(m2_1, M2_1, BeigenVect_1, BeigenVal_1, samplePerClass_1, meanPerClass_1, label_label_1, m2_2, M2_2, BeigenVect_2, BeigenVal_2, samplePerClass_2, meanPerClass_2, label_label_2, eigenThreshold);
        [DiscriminativeComponents,~] = fGetDiscriminativeComponents(TeigenVect_1, TeigenVal_1, outEVect_b, outEVal_b, eigenThreshold);
        % update variables
        t(nupdate+1,1)=toc;%记录运算时间
        m2_2 = outMean2; M2_2 = outNSample2; BeigenVect_1=outEVect_b; BeigenVal_1=outEVal_b; 
        samplePerClass_1=outSamplePerClass; meanPerClass_1=outMeanPerClass;
        label_label_1 = horzcat(label_label_1,label_label_2);   
        result.V = [result.V,DiscriminativeComponents];
        result.index(nupdate+1,:) = [result.index(nupdate,2)+1,result.index(nupdate,2)+size(DiscriminativeComponents,2)];   

    end
end
result.ClassCenter = repmat(m1_1,1,size(samplePerClass_1,2))+BeigenVect_1*meanPerClass_1;
end



