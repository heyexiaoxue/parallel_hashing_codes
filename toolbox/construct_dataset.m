function exp_data = construct_dataset(n, dataset, gnd_num)

addpath(genpath('../toolbox/'));
rpath = '../data/';

rand('state', sum(100*clock));
train_num = n; % number of traning data

basedir = [rpath dataset '/']; % modify this directory to fit your configuration
fbase = [basedir dataset '_base.fvecs'];
fquery = [basedir dataset '_query.fvecs'];
ftrain = [basedir dataset '_learn.fvecs'];
fgroundtruth = [basedir dataset '_groundtruth.ivecs'];

%Read the vectors
vtrain = fvecs_read (ftrain);
vbase  = fvecs_read (fbase);
vquery = fvecs_read (fquery);

ntrain = size (vtrain, 2);
nbase = size (vbase, 2);
nquery = size (vquery, 2);

% Load the groundtruth
ids = ivecs_read (fgroundtruth);
ids_gnd = ids + 1;  % matlab indices start at 1

train_idx = randsample(ntrain, train_num);
train_data = vtrain(:, train_idx);

meanVec = mean(train_data, 2);
train_data = train_data - repmat(meanVec, 1, train_num);

data_num = nbase;
db_data = vbase - repmat(meanVec, 1, data_num);

test_num = nquery;
test_idx = randsample(nquery, test_num);
test_data = vquery(:, test_idx);
test_data = test_data - repmat(meanVec, 1, test_num);

for i = 1:test_num
    groundtruth(i,:) = ids_gnd(1:gnd_num, test_idx(i))';
end

exp_data.train_data = train_data;
exp_data.test_data = test_data;
exp_data.groundtruth = groundtruth;
exp_data.db_data = db_data;
