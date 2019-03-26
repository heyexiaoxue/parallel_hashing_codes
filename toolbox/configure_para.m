function [ param ] = configure_para( M,bits,iter,data)
%�������ú���
%   M   PQ���ַ���
%   bits    ת����λ��
%   iter    ��������
%   data    ��������
%   dim     ����ԭʼά��
param.pos = [1 1e1:1e1:5e2];
param.M = M;
param.bits = bits;
param.iter = iter;
param.dim = size(data,2);
param.dataMean=mean(data);
[pc,~,latent,~] = princomp(data);
%pc     ͶӰ����
%latent     dim*1������ֵ�������Ӵ���С����
%balance������Ҫ�Բ�һ��
[dim_ordered ]=balanced_partition(latent,M);
param.R = pc(:,1:64);
end
