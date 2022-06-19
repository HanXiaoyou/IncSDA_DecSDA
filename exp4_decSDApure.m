% Copy of exp4_decSDA.m
% Difference:The wrong label is directly removed, 
% and no correction is made during classification  
%% %---------------------Parameter setting---------------------------%
% DataBase = 'YaleB32x32'; train_num = 44; DataBaseIndex = 'YaleB_dec'; 
% DataBase = 'YaleB32x32'; train_num = 38; DataBaseIndex = 'YaleB_dec'; 
% DataBase = 'YaleB32x32'; train_num = 51; DataBaseIndex = 'YaleB_dec'; 
% DataBase = 'YaleB32x32'; train_num = 51; DataBaseIndex = 'YaleBdec_wronglabel'; 
% DataBase = 'YaleB100x100'; train_num = 51; DataBaseIndex = 'YaleBdec_wronglabel'; 
% DataBase = 'NIR128x128'; train_num = 12; DataBaseIndex = 'NIRdec_wronglabel'; 
% DataBase = 'YaleB100x100'; train_num = 51; DataBaseIndex = 'YaleB_dec';
% DataBase = 'YouTubeFace_320x320'; train_num = 72; DataBaseIndex = 'YouTubeFacedec_wronglabel'; 
DataBase = 'YaleB100x100'; train_num = 38; DataBaseIndex = 'YaleB_dec';
% DataBase = 'NIR128x128'; train_num = 12; DataBaseIndex = 'NIRdec';
type = 'Normalize';
% all decremental data are labled samples
%% %-----------------------------Import Data-------------------------%
[fea,gnd]=DataProcess(DataBase,type);
%%
for group = 1:1
    fprintf(1,strcat('The'," ",num2str(group)," ",'group','\n'));
    %-----------------------------Export index--------------------------------%
    eval(['load '  'DataBase_Index\',DataBaseIndex '\' int2str(train_num) 'Train\'  int2str(group) '.mat']);   %
    X_train = (fea(trainIdx,:))';  %D*N 
    Y_train = (gnd(trainIdx,:))';  %1*N
    X_test = (fea(testIdx,:))';    %D*N
    Y_test = (gnd(testIdx,:))';    %1*N
    Y_trainorigin = Y_train;       %1*N
    
    Class_Number =  size(union([],Y_train),1);
    [nupdate,decnum] = size(decr_ind);
    if ~exist('wronglabel','var')
        % Set error label
        class = max(Y_train);%number of classes
        for r = 1:nupdate
          Y_train(:,decr_ind(nupdate,:)) = Y_train(:,decr_ind(nupdate,:))+1;% class+1
          Y_train(Y_train==class+1) = 1;% The last class is set to the first class
%             wrongidx = decr_ind(nupdate,:);
%             for m = 1:size(wrongidx,2)
%                 Y_train(:,wrongidx(m)) = randperm(Class_Number,1);
%             end
        end
    else
        Y_train(:,decr_ind) = wronglabel;
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
    Y_train(init_unlabel) = 0;    
    for i = 1:Class_Number
        ClassCenter(:,i) = mean(X_train(:,Y_train==i),2);
    end
    %initial
    model.ClassCenter = ClassCenter;
    W = result.V(:,result.index(1,1):result.index(1,2));
    option.nupdate = 0;
    [accuracy_decSDA(1,1),~,predictlabel(1,:)] = TestEval(W,model,method);
    %decremental
    Y_train = Y_trainorigin;
    index = [decr_ind';init_unlabel];%wrong tag index and unlabel index
%     Y_train(init_unlabel) = 0;
    Y_train(index) = 0;
    Class_Number =  size(union([],Y_train),1);
    for i = 1:Class_Number
        ClassCenter(:,i) = mean(X_train(:,Y_train==i),2);
    end
    model.ClassCenter = ClassCenter;
    for i = 2:nupdate+1
        W = result.V(:,result.index(i,1):result.index(i,2));
        option.nupdate = i-1;
        [accuracy_decSDA(i,1),~,predictlabel(i,:)] = TestEval(W,model,method);
    end
    %save results
%     alg = 'decSDApure_wronglabel';% eig  %pure + wronglabel
    alg = strcat('decSDA_',num2str(train_num));% eig
    file = strcat(DataBase,alg,'_',num2str(group),'.mat');
    save(['RepeatResult\', file],'accuracy_decSDA','t_decSDA','predictlabel','-v7.3');
clearvars -except fea gnd DataBaseIndex train_num DataBase
end
%%
