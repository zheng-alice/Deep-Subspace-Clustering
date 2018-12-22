%--------------------------------------------------------------------------
% This is the function to call the sparse optimization program, to call the 
% spectral clustering algorithm and to compute the clustering error.
% r = projection dimension, if r = 0, then no projection
% affine = use the affine constraint if true
% s = clustering ground-truth
% missrate = clustering error
% CMat = coefficient matrix obtained by SSC
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function [grps] = SSC_modified(k,r,affine,alpha,outlier,rho,thr,maxIter,cluster,seed)

if (nargin < 10)
    seed = -1;
end
if (nargin < 9)
    cluster = true;
end
if (nargin < 8)
    maxIter = 150;
end
if (nargin < 7)
    thr = 2*10^-4;
end
if (nargin < 6)
    rho = 1;
end
if (nargin < 5)
    outlier = true;
end
if (nargin < 4)
    alpha = 20;
end
if (nargin < 3)
    affine = false;
end
if (nargin < 2)
    r = 0;
end

%still the fastest way to pass large ndarrays
%the alternative takes 37 seconds per (2350x300) array
load ./../temp.mat X;
X = double(X.');
Xp = DataProjection(X,r);

if (~outlier)
    CMat = admmLasso_mat_func(Xp,affine,alpha,thr,maxIter);
    C = CMat;
else
    CMat = admmOutlier_mat_func(Xp,affine,alpha,thr,maxIter);
    N = size(Xp,2);
    C = CMat(1:N,:);
end

grps = [];
if (cluster)
    disp("Clustering...")
    CKSym = BuildAdjacency(thrC(C,rho));
    if (seed >= 0)
        if(isOctave)
            rand('state', 0);
	else
            rng(seed);
	end
    end
    grps = SpectralClustering(CKSym,k);
end

save ./../temp.mat C -mat
