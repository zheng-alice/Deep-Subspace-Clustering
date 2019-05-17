clear variables;
load ../saved/rescaled/Coil20
X = double(X');

result = zeros(10, 3);
for p = 1:10
    N = 2^p;
    n = 30;
    err_sum = 0;
    err_mean = 0;
    for i = 1:n
        idx = randperm(length(X), N); 
        x = X(:, idx);
        y = Y(:, idx);
        [miss,C] = SSC(x, 0, false, 20, false, 1, y);
        err_sum = err_sum + sum(sum(C.^2));
        err_mean = err_mean + mean(mean(C.^2));
    end
    err_mean = err_mean / n;
    err_sum = err_sum / n;
    fprintf('N: %i, mean: %d, sum: %d\n', N, err_mean, err_sum);
    result(p, :) = [p, log2(err_mean), log2(err_sum)];
end

plot(result(:, 1), result(:, 2), result(:, 1), result(:, 3));
fprintf('Mean: log2(E_m) = %f*log2(N) + %f\n', polyfit(result(:, 1), result(:, 2), 1));
fprintf('Sum: log2(E_s) = %f*log2(N) + %f\n', polyfit(result(:, 1), result(:, 3), 1));

% n=30
% p=1:10

% yaleB (processed):
% Mean: log2(E_m) = -0.066119*log2(N) + -10.781865
% Sum: log2(E_s) = 1.933881*log2(N) + -10.781865
% Coil20 (processed):
% Mean: log2(E_m) = -0.686837*log2(N) + -4.337824
% Sum: log2(E_s) = 1.313163*log2(N) + -4.337824
% yaleB (rescaled):
% ...
% ...
% Coil20 (rescaled):
% Mean: log2(E_m) = -0.695614*log2(N) + -3.998875
% Sum: log2(E_s) = 1.304386*log2(N) + -3.998875
% MNIST (rescaled):
% ...
% ...
% CIFAR10 (rescaled):
% ...
% ...