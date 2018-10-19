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

function [grps,C] = SSC_modified(X,k,r,affine,alpha,outlier,rho,thr,maxIter)

if (nargin < 9)
    maxIter = 150;
end
if (nargin < 8)
    thr = 2*10^-4;
end
if (nargin < 7)
    rho = 1;
end
if (nargin < 6)
    outlier = true;
end
if (nargin < 5)
    alpha = 20;
end
if (nargin < 4)
    affine = false;
end
if (nargin < 3)
    r = 0;
end

Xp = DataProjection(X,r);

if (~outlier)
    CMat = admmLasso_mat_func(Xp,affine,alpha,thr,maxIter);
    C = CMat;
else
    CMat = admmOutlier_mat_func(Xp,affine,alpha,thr,maxIter);
    N = size(Xp,2);
    C = CMat(1:N,:);
end

CKSym = BuildAdjacency(thrC(C,rho));
grps = SpectralClustering(CKSym,k);