A = randn(1000);

svd_driver("gesvd");
tic; [~,~,~] = svd(A); gesvd = toc;
fprintf("gesvd took %d seconds\n", gesvd);

svd_driver("gesdd");
tic; [~,~,~] = svd(A); gesdd = toc;
fprintf("gesdd took %d seconds\n", gesdd);

if gesvd < gesdd
	svd_driver("gesvd");
	disp "Using gesvd"
else
	disp "Using gesdd"
end
