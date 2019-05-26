clear variables;
load ../saved/rescaled/CIFAR10
X = double(X');

range = 5:10;
n = 30;

result = zeros(max(range), 3);
for p = range
    N = 2^p;
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

plot(result(range, 1), result(range, 2), result(range, 1), result(range, 3));
fprintf('Mean: log2(E_m) = %f*log2(N) + %f\n', polyfit(result(range, 1), result(range, 2), 1));
fprintf('Sum: log2(E_s) = %f*log2(N) + %f\n', polyfit(result(range, 1), result(range, 3), 1));

% rescaled
% range = 5:10;
% n = 30;

% yaleB:
% Mean: log2(E_m) = -0.891821*log2(N) + -2.995528
% Sum: log2(E_s) = 1.108179*log2(N) + -2.995528
% Coil20:
% Mean: log2(E_m) = -0.890098*log2(N) + -2.511976
% Sum: log2(E_s) = 1.109902*log2(N) + -2.511976
% MNIST:
% Mean: log2(E_m) = -1.015780*log2(N) + -2.891591
% Sum: log2(E_s) = 0.984220*log2(N) + -2.891591
% CIFAR10:
% Mean: log2(E_m) = -0.918135*log2(N) + -5.456893
% Sum: log2(E_s) = 1.081865*log2(N) + -5.456893