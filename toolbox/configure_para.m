function [ param ] = configure_para( M,bits,iter,data)
%参数配置函数
%   M   PQ划分份数
%   bits    转化的位数
%   iter    迭代次数
%   data    输入数据
%   dim     数据原始维度
param.pos = [1 1e1:1e1:5e2];
param.M = M;
param.bits = bits;
param.iter = iter;
param.dim = size(data,2);
param.dataMean=mean(data);
[pc,~,latent,~] = princomp(data);
%pc     投影矩阵
%latent     dim*1的特征值向量，从大向小排序
%balance特征重要性不一样
[dim_ordered ]=balanced_partition(latent,M);
param.R = pc(:,1:64);
end

