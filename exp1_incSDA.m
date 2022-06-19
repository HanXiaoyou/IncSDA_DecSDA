DataBaseName = 'VIS128x128';
% DataBaseName = 'VIS32x32';
% DataBaseName = 'NIR128x128'; % NxD
% DataBaseName = 'YaleB32x32';
% DataBaseName = 'YaleB64x64';
% DataBaseName = 'YaleB100x100';
type = 'Normalize';
[fea,gnd]=DataProcess(DataBaseName,type);
fea = fea'; %DxN
gnd = gnd'; %1xN
for nexp = 1:1
    Nexp = num2str(nexp);
    fprintf(1,strcat('The',Nexp,'times of experiment','\n'));
    eval(['load ' 'DataBase_Index\',num2str(DataBaseName),'\',Nexp '.mat'])
    X_test = fea(:,pro_idx(:,6)~=0);    %D×N
    Y_test = gnd(:,pro_idx(:,6)~=0);    %1×N
    %% % incSDA
    %训练部分
    init_ind = find(pro_idx(:,2)==1);
    init_unlabel = find(pro_idx(:,1)==0);
    nupdate = size(pro_idx,2) - 3;
    for i = 1:nupdate
        incr_ind(i,:) = (find(pro_idx(:,i+2)==1))'; 
    end
    
    Y_LU_init = zeros(1,size([init_ind;init_unlabel],1));
    data2.X_init = [fea(:,init_ind),fea(:,init_unlabel)];
    Y_LU_init(1,1:size(init_ind,1)) = gnd(init_ind);
    data2.Y_LU_init = Y_LU_init;
    data2.X_incr = [];
    photo = size(incr_ind,2);
    for n = 1:nupdate
        data2.X_incr = [data2.X_incr,fea(:,incr_ind(n,:))];
        data2.Y_LU_incr(n,:) = gnd(1,incr_ind(n,:));
    end
    option2.n = nupdate; option2.p = photo;
    option2.beta1=1e-6; option2.beta2=1e-3;
    eigenThreshold = 0.01;
    [result,t_incSDA] = incSDA_eig(data2,option2,eigenThreshold);
    %测试部分(NNCD)
    accuracy_incSDA = zeros(nupdate+1,1);
    method = 'NNCD';
    model.X_Test = X_test;model.Y_Test = Y_test;
    model.ClassCenter = result.ClassCenter;
    model.ClassLabel = union([],Y_test);
    for i = 1:nupdate+1
        W = result.V(:,result.index(i,1):result.index(i,2));
        option.nupdate = i-1;
        [accuracy_incSDA(i,1),~,predictlabel(i,:)] = TestEval(W,model,method);
    end
    %储存结果
    alg = 'incSDA';
    file = strcat(num2str(DataBaseName),alg,'_',Nexp,'.mat');
    save(['RepeatResult\', file],'accuracy_incSDA','t_incSDA','predictlabel','-v7.3');
    clearvars -except fea gnd type DataBaseName accuracy_incSDA
end