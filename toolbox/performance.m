function evaluation_info = performance(B_tst, B_db, groundtruth, param)

test_num = size(B_tst, 1);
pos = param.pos;
poslen = length(pos);
label_r = zeros(1, poslen);
label_p = zeros(1, poslen);
label_ap = zeros(1, 1);
label_ph1 = zeros(1, 1);
label_ph2 = zeros(1, 1);
for n = 1:test_num
    D_code = hammingDist(B_tst(n,:),B_db);
    D_truth = groundtruth(n,:);
     
    [P, R, AP, PH1, PH2] = precall2(D_code, D_truth, pos);

    label_r = label_r + R(1:poslen);
    label_p = label_p + P(1:poslen);
    label_ap = label_ap + AP;
    label_ph1 = label_ph1 + PH1;
    label_ph2 = label_ph2 + PH2;
    
end
evaluation_info.recall=label_r/test_num;
evaluation_info.precision=label_p/test_num;
evaluation_info.AP=label_ap/test_num;
evaluation_info.PH1=label_ph1/test_num;
evaluation_info.PH2=label_ph2/test_num;
evaluation_info.param=param;