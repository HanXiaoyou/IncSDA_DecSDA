%% %----------------------------参数的设定----------------------------%
% DataBase = 'CAS_PEAL'; train_num = 16;
% DataBase = 'YaleB100x100'; train_num = 51;
DataBase = 'YaleB_100x100'; train_num = 51; DataBase_tantamount = 'YaleB100x100';
% DataBase = 'YouTubeFace_320x320'; train_num = 72;
type = 'Normalize';
incunlabel = 0.5;%增量过程无标记样本所占比例
%% %-----------------------------导入数据-----------------------------%
[fea,gnd]=DataProcess(DataBase,type);
%%
%随机序列的生成方法见DataBase里面的MainCreatTrainTest.m
for group = 1:1
    fprintf(1,strcat('第',num2str(group),'次实验','\n'));
    %-----------------------------导出索引数据-----------------------------%
%     % --只有YouTubeFace数据集才需要%***%代码
%     DataBase = 'YouTubeFace'; train_num = 72;
%     j = num2str(group);
%     databasegroup = [DataBase,j];
%     type = 'Normalize';
%     incunlabel = 0.6;
%     [fea,gnd]=DataProcess(databasegroup,type);
    eval(['load '  'DataBase_Index\',DataBase '\' int2str(train_num) 'Train\'  int2str(group) '.mat']);   %
%     % --
%     eval(['load '  'DataBase_Index\',DataBase_tantamount '\' int2str(train_num) 'Train\'  int2str(group) '.mat']);   %
    X_train = fea(trainIdx,:);  %%N×D 
    Y_train = gnd(trainIdx,:);  %%N×1
    X_test = fea(testIdx,:);    %%N×D
    Y_test = gnd(testIdx,:);    %%N×1
%     clear fea gnd
    %-----------------------------IESDE 初始训练部分-----------------------%
    trainSetL = X_train(init_ind,:);
    trainSetU = X_train(init_unlabel,:);
    trainSet = [trainSetL;trainSetU];
    testSet = X_test;
    options.gndL =Y_train(init_ind,:);
    trainLabel = [Y_train(init_ind,:);Y_train(init_unlabel,:)];%[Y_L;Y_U] 原始程序Extractunlabelset.m是按照类别顺序取的
    testLabel = Y_test;
    options.gnd = trainLabel;
    options.train_No = train_num;
    k = length(unique(trainLabel)); % Projected dimension
    
    T = 2*size(incr_ind,1);
    ESDEI_LUtime = zeros(T,1);
    [QS2,RS2,LwS2,LbS2,LS2,eigvector,ESDEI_LUtime(1,1)]= YESDEA(trainSetL,trainSetU,options,k);
    high = size(eigvector,2);
    accuracy = zeros(1, high); % 初始识别率
    trainSet = trainSetL;trainLabel = Y_train(init_ind,:); %Unlabel数据的标签信息不可用于测试
    for j = 1:high
       train = trainSet * eigvector(:,1:j);
      test = testSet * eigvector(:,1:j);
      accuracy(j) = NN(train,test,trainLabel,testLabel);%nearest neighbour分类器
    end
    accuracy_YESDEA = zeros(T+1,1);
    accuracy_YESDEA(1,1) = max(accuracy);
    fprintf(1,strcat('第',num2str(group),'次实验初始训练','accuracy_YESDEA为',num2str(accuracy_YESDEA(1,1)),'\n'));

    %---------------------------IESDE 增量训练部分-------------------------%
    incr_ind_LU = incr_ind(:,1:0.5*size(incr_ind,2));
    incr_ind_LU = [incr_ind_LU;incr_ind(:,0.5*size(incr_ind,2)+1:end)];%均匀拆分增量中label和unlabel数据
    trainSetL1 = trainSetL;
    trainSetU1 = trainSetU;
    for t = 1:T
        updataSet = X_train(incr_ind_LU(t,:),:);       
        options.gndt = Y_train(incr_ind_LU(t,:),1);%增量label data的标签
        [eigvector,ESDEI_LUtime(t+1,1),QS2,RS2,LwS2,LbS2,LS2]= ESDEI_LU(t,trainSetL1,trainSetU1,updataSet,options,k,QS2,RS2,LwS2,LbS2,LS2);
        high = size(eigvector,2);
        %%----------分类识别---------%
        if (mod(t,2)==1)
            trainSetN = [trainSet;updataSet];
            trainLabelN=[options.gnd;options.gndt];
        else
            trainSetN = trainSet;
            trainLabelN = trainLabel;
        end
        accuracy = zeros(1, high);
        for j = 1:high
            train = trainSetN * eigvector(:,1:j);
            test = testSet * eigvector(:,1:j);
            accuracy(j) = NN(train,test,trainLabelN,testLabel);%nearest neighbour分类器
        end
        accuracy_YESDEA(t+1,1) = max(accuracy);
        fprintf(1,strcat('第',num2str(group),'次实验第',num2str(t), '次增量训练','accuracy_YESDEA为',num2str(accuracy_YESDEA(t+1,1)),'\n'));
        %-------------------------------更新变量------------------------------%
        if (mod(t,2)==1)
            trainSet=[trainSet;updataSet];
            trainSetL1=[trainSetL1;updataSet];
            trainSetU1=trainSetU1;
            
            options.gnd=[options.gnd;options.gndt];
            options.gndL=[options.gndL;options.gndt];
            trainLabel=options.gnd;
        else
            trainSet=[trainSet;updataSet];
            trainSetL1=trainSetL1;
            trainSetU1=[trainSetU1;updataSet];
            
            options.gnd=[options.gnd;options.gndt];
            options.gndL=[options.gndL;options.gndt];
            trainLabel=options.gnd;
        end
    end
    %储存结果
    alg = 'IESDE';
    file = strcat(DataBase,alg,'_',num2str(group),'.mat');
    save(['RepeatResult\', file],'accuracy_YESDEA','ESDEI_LUtime','-v7.3');
end