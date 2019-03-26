function [bin] = compressAOH( data,params,npart )
%压缩
bin =[];
for m=1:npart
    param = params{m};
    X_sub = data((m-1)*param.sub_Dim+1:m*param.sub_Dim,:)';
    %中心化
    X_sub = bsxfun(@minus,X_sub,param.dataMean);
    X_sub = X_sub*param.R;
    %原始空间的C
    [idx,~] = yael_nn(param.C,single(X_sub'));
    bin = [bin param.codebook(idx)];
end
end