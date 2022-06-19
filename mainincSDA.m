%% %----------------------------数据库提示-------------------------------%

%% %----------------------------参数的设定-------------------------------%
% DataBase = 'CAS_PEAL'; train_num = 16;
DataBase = 'YaleB100x100'; train_num = 51;
% DataBase = 'YouTubeFace_320x320'; train_num = 72;
type = 'Normalize';
incunlabel = 0.6;%增量过程无标记样本所占比例
%% %-----------------------------导入数据--------------------------------%
[fea,gnd]=DataProcess(DataBase,type);
%%
%随机序列的生成方法见DataBase里面的MainCreatTrainTest.m
for group = 1:1
    fprintf(1,strcat('第',num2str(group),'次实验','\n'));
    %-----------------------------导出索引数据--------------------------------%
%     % --只有YouTubeFace数据集才需要%***%代码
%     DataBase = 'YouTubeFace'; train_num = 72;
%     j = num2str(group);
%     databasegroup = [DataBase,j];
%     type = 'Normalize';
%     incunlabel = 0.6;
%     [fea,gnd]=DataProcess(databasegroup,type);
%     % --
    eval(['load '  'DataBase_Index\',DataBase '\' int2str(train_num) 'Train\'  int2str(group) '.mat']);   %
    X_train = (fea(trainIdx,:))';  %D×N 
    Y_train = (gnd(trainIdx,:))';  %D×1
    X_test = (fea(testIdx,:))';    %D×N
    Y_test = (gnd(testIdx,:))';    %D×1
    clear fea gnd
    %-----------------------------incSDA 训练部分-----------------------------%
    Y_LU_init = zeros(1,size([init_ind;init_unlabel],1));
    data2.X_init = [X_train(:,init_ind),X_train(:,init_unlabel)];%[XL,XU]
    Y_LU_init(1,1:size(init_ind,1)) = Y_train(init_ind);
    data2.Y_LU_init = Y_LU_init;
    
    nupdate = size(incr_ind,1);  photo = size(incr_ind,2);
    data2.X_incr = [];
    data2.Y_LU_incr = zeros(nupdate,size(incr_ind,2));
    for n = 1:nupdate
        data2.X_incr = [data2.X_incr,X_train(:,incr_ind(n,:))];
        data2.Y_LU_incr(n,:) = Y_train(1,incr_ind(n,:));
        
%         subindices = randperm(size(incr_ind,2));
%         data2.Y_LU_incr(n,subindices(1,1:floor(photo*incunlabel))) = 0; %增量unlabel数据标签为0
        data2.Y_LU_incr(n,1:floor(photo*incunlabel)) = 0; %增量unlabel数据标签为0
    end
    option2.n = nupdate; option2.p = photo;
    eigenThreshold = 0.01;
    [result,t_incSDA] = incSDA(data2,option2,eigenThreshold);
    %-----------------------------incSDA 测试部分-----------------------------%  
    %使用Cai Deng的最近邻分类方法NNCD
    accuracy_incSDA = zeros(nupdate+1,1);
    method = 'NNCD';
    model.X_Test = X_test;model.Y_Test = Y_test;
    %     for i = 1:Class_Number
    %         ClassCenter(:,i) = mean(X_train(:,find(Y_train==i)),2);
    %     end
    model.ClassCenter = result.ClassCenter;
    model.ClassLabel = union([],Y_train);
    
    for i = 1:nupdate+1
        W = result.V(:,result.index(i,1):result.index(i,2));
        option.nupdate = i-1;
        accuracy_incSDA(i,1) = TestEval(W,model,method);
    end
    %储存结果
    alg = 'incSDA';
    file = strcat(DataBase,alg,'_',num2str(group),'.mat');
    save(['RepeatResult\', file],'accuracy_incSDA','t_incSDA','-v7.3');
end

% %%
% % 另外增加的比较算法：IESDE (I+R*L*R')进行增量QR处理
% % DataBase = 'ORLnor64x64';
% DataBase = 'YaleBnor64x64';
% eval(['load ' 'DataBase\' DataBase '.mat'])
% fea = fea';gnd = gnd';
% for group = 1:10
%     Nexp = num2str(group);
%     DataBase = 'YaleB64x64_t2n5p160p015_';
%     eval(['load ' 'expDataBase\YaleB64\' DataBase,Nexp '.mat'])
%     X_train = fea(:,Train_indices);
%     X_test = fea(:,Test_indices);
%     %% % ESDEI_L % 陆迎弟师姐的算法和代码
%     %训练部分
%     %初始训练部分
%     trainNo = 52;%对算法没什么影响
%     trainSetL = (X_train(:,init_ind))';trainSetU = (X_train(:,init_unlabel))';%N×D
%     options=[]; options.gnd = Y_train(:,init_ind); options.train_No = trainNo;
%     k = length(unique(Y_train));
%     [QS,RS,LwS,LbS,LS,eigvector,ESDEI_Utime]= YESDEA(trainSetL,trainSetU,options,k);
%     t_ESDEI_U(1,1) = ESDEI_Utime;
%     result.V = eigvector; result.index(1,:) = [1,size(eigvector,2)];
%     %增量训练部分
%     for n = 1:nupdate
%         updataSet = (X_train(:,incr_ind(n,:)))';
%         [eigvector,ESDEI_Utime,QS,RS,LwS,LbS,LS]= ESDEI_U(trainSetL,trainSetU,updataSet,options,k,QS,RS,LwS,LbS,LS);
%         result.V = [result.V,eigvector]; result.index(n+1,:) = [result.index(n,2)+1,result.index(n,2)+size(eigvector,2)];
%         trainSetU=[trainSetU;updataSet];
%         t_ESDEI_U(1,n+1) = ESDEI_Utime;
%     end
%     
%     %测试部分 NNCD
%     accuracy_ESDEI_U = zeros(nupdate+1,1);
%     method = 'NNCD';
%     model.X_Test = X_test;model.Y_Test = Y_test;
%     for i = 1:Class_Number
%         ClassCenter(:,i) = mean(X_train(:,find(Y_train==i)),2);
%     end
%     model.ClassCenter = ClassCenter;
%     model.ClassLabel = union([],Y_train);
%     
%     for i = 1:nupdate+1
%         W = result.V(:,result.index(i,1):result.index(i,2));
%         option.nupdate = i-1;
%         accuracy_ESDEI_U(i,1) = TestEval(W,DataBase,model,method,option);
%     end
%     %储存结果
%     alg = 'ESDEI_U';
%     file = strcat(DataBase,alg,'_',Nexp,'.mat');
%     save(['RepeatResult\', file],'accuracy_ESDEI_U','t_ESDEI_U','-v7.3');
% end
% 

