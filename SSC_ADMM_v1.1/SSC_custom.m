% load (preprocessed) data to cluster
stage = 2;
if (stage == 1)
    load("C:/Users/aleks_000/Desktop/Mentorship/1-helper/PCA_norm/YaleB.mat")
    X = double(X.');
    alpha = 20;
    maxIter = 6;
elseif (stage == 2)
    load("C:/Users/aleks_000/Desktop/Mentorship/1-DSCwSP/HM2/YaleB.mat")
    X = double(HM2.');
    alpha = 18;
    maxIter = 10;
end
Y = double(Y); % ground-truth
k = size(unique(Y), 2);

r = 0; % PCA number of dimensions (0 to not perform)
affine = false;
outlier = false;
rho = 1;
thr = 1*10^-20;
tic
[grps, C] = SSC_modified(X, k, r, affine, alpha, outlier, rho, thr, maxIter);
toc
if (stage == 1)
    save ./../C_mats/YaleB.mat C
elseif (stage == 2)
    save ./../C_finals/YaleB.mat C
end
save ./../Labels_pred/YaleB.mat grps