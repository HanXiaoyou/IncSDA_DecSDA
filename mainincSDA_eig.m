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
        data2.Y_LU_incr(n,1:floor(photo*incunlabel)) = 0; %增量unlabel数据标签为0
    end
    option2.n = nupdate; option2.p = photo; option2.beta1=1e-6; option2.beta2=1e-3; 
    eigenThreshold = 0.01;
    [result,t_incSDA] = incSDA_eig(data2,option2,eigenThreshold);
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
    alg = 'incSDA_eig';
    file = strcat(DataBase,alg,'_',num2str(group),'.mat');
    save(['RepeatResult\', file],'accuracy_incSDA','t_incSDA','-v7.3');
end




