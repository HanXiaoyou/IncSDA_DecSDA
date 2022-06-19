%% %---------------------Parameter setting---------------------------%
% DataBase = 'YaleB100x100'; train_num = 44; DataBaseIndex = 'YaleB_dec'; %只用于调�?
% DataBase = 'YaleB100x100'; train_num = 38; DataBaseIndex = 'YaleB_dec'; %只用于调�?
% DataBase = 'YaleB100x100'; train_num = 51; DataBaseIndex = 'YaleB_dec'; %只用于调�?
% DataBase = 'YaleB32x32'; train_num = 44; DataBaseIndex = 'YaleB_dec';
% DataBase = 'NIR128x128'; train_num = 12; DataBaseIndex = 'NIRdec'; %只用于调�?
% DataBase = 'NIR128x128'; train_num = 12100; DataBaseIndex = 'NIRdec'; %只用于调�?
% DataBase = 'NIR128x128'; train_num = 12150; DataBaseIndex = 'NIRdec'; %只用于调�?
% DataBase = 'NIR128x128'; train_num = 14150; DataBaseIndex = 'NIRdec'; %只用于调�?
DataBase = 'NIR128x128'; train_num = 16150; DataBaseIndex = 'NIRdec'; %只用于调�?
type = 'Normalize';
% all decremental data are labled samples
%% %-----------------------------Import Data-------------------------%
[fea,gnd]=DataProcess(DataBase,type);
%%
for group = 1:10
    fprintf(1,strcat('The'," ",num2str(group)," ",'group','\n'));
    %-----------------------------Export index--------------------------------%
%     % --only YouTubeFace need %***% code
%     DataBase = 'YouTubeFace'; train_num = 72;
%     j = num2str(group);
%     databasegroup = [DataBase,j];
%     type = 'Normalize';
%     incunlabel = 0.6;
%     [fea,gnd]=DataProcess(databasegroup,type);
%     % --
    eval(['load '  'DataBase_Index\',DataBaseIndex '\' int2str(train_num) 'Train\'  int2str(group) '.mat']);   %
    X_train = (fea(trainIdx,:))';  %D*N 
    Y_train = (gnd(trainIdx,:))';  %1*N
    X_test = (fea(testIdx,:))';    %D*N
    Y_test = (gnd(testIdx,:))';    %1*N
    Y_trainorigin = Y_train;       %1*N
    % Set error label 
    nupdate = size(decr_ind,1);
    class = max(Y_train);%number of classes
    for r = 1:nupdate
        Y_train(:,decr_ind(nupdate,:)) = Y_train(:,decr_ind(nupdate,:))+1;% class+1
        Y_train(Y_train==class+1) = 1;% The last class is set to the first class 
    end  
%     clear fea gnd
    %-----------------------decSDA initial train---------------------------%
    Y_LU_init = zeros(1,size([init_ind;init_unlabel],1));
    data.X_init = [X_train(:,init_ind),X_train(:,init_unlabel)];%[XL,XU]
    Y_LU_init(1,1:size(init_ind,1)) = Y_train(init_ind);
    data.Y_LU_init = Y_LU_init;
    
    photo = size(decr_ind,2);
    data.X_decr = [];
    data.Y_LU_decr = zeros(nupdate,size(decr_ind,2));
    for n = 1:nupdate
        data.X_decr = [data.X_decr,X_train(:,decr_ind(n,:))];
        data.Y_LU_decr(n,:) = Y_train(1,decr_ind(n,:));       
    end
    option.n = nupdate; option.p = photo; 
    option.decr_ind = decr_ind; % 1*N_dec
    option.beta1=1e-6; option.beta2=1e-3; 
    eigenThreshold = 0.01;
    [result,t_decSDA] = decSDA(data,option,eigenThreshold);
    %-----------------------------decSDA test-----------------------------%  
    % Cai Deng -- ClassCenter
    accuracy_decSDA = zeros(nupdate+1,1);
    method = 'NNCD';
    model.X_Test = X_test;model.Y_Test = Y_test;
    model.ClassLabel = union([],Y_train);
    %initial
    model.ClassCenter = result.ClassCenterI;
    W = result.V(:,result.index(1,1):result.index(1,2));
    option.nupdate = 0;
    accuracy_decSDA(1,1) = TestEval(W,model,method);
    %decremental
    Class_Number =  size(union([],Y_train),1);
    for i = 1:Class_Number
        ClassCenter(:,i) = mean(X_train(:,Y_trainorigin==i),2);
    end
    model.ClassCenter = ClassCenter;
    for i = 2:nupdate+1
        W = result.V(:,result.index(i,1):result.index(i,2));
        option.nupdate = i-1;
        accuracy_decSDA(i,1) = TestEval(W,model,method);
    end
    %save results
    alg = 'decSDA_16_150';% eig
    file = strcat(DataBase,alg,'_',num2str(group),'.mat');
    save(['RepeatResult\', file],'accuracy_decSDA','t_decSDA','-v7.3');
end
 



