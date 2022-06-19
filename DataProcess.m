function [fea,gnd]=DataProcess(DataBase,type)
% DataProcess.m是对全数据做预处理
% 相比于LoadFace.m相当于做重复实验室只对数据预处理做一次，省时。
%加载数据
%     Input:
%          DataBase     -----选择数据库用于实验，参数选择为'ORL','Yale','YaleB' or 'PIE';
%          train_num    -----每个人选择用于训练的图片张数;
%          group        -----生成的随机标签，即随机选取不同的人脸用于训练，总共有50组;
%          type         -----数据加载的方式，参数包括：'Original','Scale','Normalize'.
%              'Original'    ----加载原始图像灰度值
%              'Scale'       ----将灰度值映射到[0,1]上
%              'Normalize'   ----对每张人脸进行归一化
%     Output:
%          fea   -----处理后的全数据集，其中每行代表1个人脸;
%          gnd   -----处理后的全数据集标签;
%    Written By PWR
switch DataBase
    case 'CAS_PEAL'
        eval(['load ' 'DataBase\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'YaleB100x100'
        eval(['load ' 'DataBase\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'YaleB_32x32'
        eval(['load ' 'DataBase\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'YaleB32x32'
        eval(['load ' 'DataBase\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gn
%     case 'YaleB64x64'
%         eval(['load ' 'DataBase\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gn
    case 'YaleB_100x100'
        eval(['load ' 'DataBase\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'ExtendedYaleB_crop_169x192'
        eval(['load ' 'DataBase\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'ExtendedYaleB_crop_100x100'
        eval(['load ' 'DataBase\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'VIS32x32'
        eval(['load ' 'DataBase\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'VIS128x128'
        eval(['load ' 'DataBase\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'NIR32x32'
        eval(['load ' 'DataBase\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'NIR128x128'
        eval(['load ' 'DataBase\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'PIE_32x32'
        eval(['load ' 'DataBase\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'PIE_486x640_800'
        eval(['load ' 'DataBase\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'YouTubeFace_320x320'
        eval(['load ' 'DataBase\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'YouTubeFace1'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'YouTubeFace2'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'YouTubeFace3'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'YouTubeFace4'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'YouTubeFace5'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'YouTubeFace6'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'YouTubeFace7'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'YouTubeFace8'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'YouTubeFace9'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'YouTubeFace10'
        eval(['load ' 'DataBase\YouTubeFaceClassiid\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
    case 'YouTubeFace_320x320_crop_160x160'
        eval(['load ' 'DataBase\' DataBase '.mat']);%加载人脸数据库,加载后有数据集fea和标签gnd
        
end
fea = double(fea);
[nSmp,nFea] = size(fea);    %nSmp:人脸的张数； nFea:人脸的维数，例ORL_32x32:nSmp=400, nFea=32x32=1024
if (~exist('type','var'))
    type = 'original';
    %     type='normalize';
end
switch lower(type)
    case 'scale'
        maxValue = max(max(fea));                                 %除以最大值(整个矩阵的最大)，将像素值映射到[0,1]上
        fea = fea/maxValue;
    case 'normalize'
        for i=1:nSmp
            fea(i,:) = fea(i,:)./ max(1e-12,norm(fea(i,:)));     %防止除以0，进行向量归一化运算
        end
    case 'original'
    otherwise
        error('请选取正确的数据加载方式！');
end

