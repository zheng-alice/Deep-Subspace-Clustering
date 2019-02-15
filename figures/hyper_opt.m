load hyper_opt.mat;
axxes = containers.Map(fieldnames(axxes), struct2cell(axxes));

if axxes.Count < 2
    error("Data has too few dimensions: " + axxes.Count);
end
idx_x = 1;
idx_y = 2;


M = size(truth.y, 1);
best = truth.x(truth.best, :);
for i = 1:size(best, 2)
    % convert to surrogate indices
    [~, idx] = min(abs(axxes(deblank(truth.names(i, :)))-best(i)));
    best(i) = idx;
end
slice = num2cell(best);
slice{idx_x} = ':';
slice{idx_y} = ':';


% create colors and size
% size decreases with distance to visible plane
c = linspace(0, 1, M);
s = 50.* ones(M, 1);
hidden = 1:axxes.Count;
hidden([idx_x, idx_y]) = [];
plane = 1:size(hidden, 2);
range = 1:size(hidden, 2);
values = truth.x(:, hidden);
for i = plane
    axis = axxes(deblank(truth.names(i, :)));
    plane(i) = axis(slice{hidden(i)});
    range(i) = axis(size(axis, 1));
    if strcmp(deblank(truth.priors(i, :)), 'log')
       plane(i) = log10(plane(i));
       range(i) = log10(range(i)) - log10(axis(1));
       values(:, i) = log10(values(:, 1));
    else
       range(i) = range(i) - axis(1);
    end
end
dist = sqrt(sum(((values - plane)./ range).^2, 2));
s = s.* (1.- (dist./sqrt(max(1, size(hidden, 2)))));
plot_truth = scatter3(truth.x(:, idx_x), truth.x(:, idx_y), truth.y, s, c, 'filled');

hold on;

means = surrogate.mean(slice{:});
stds = surrogate.std(slice{:});
if idx_x < idx_y
    means = means.';
    stds = stds.';
end
plot_surrogate = surf(axxes(deblank(truth.names(idx_x, :))), axxes(deblank(truth.names(idx_y, :))), means);
set(plot_surrogate, 'EdgeAlpha', 'interp', 'FaceColor', 'interp', 'FaceAlpha', 'interp');
set(plot_surrogate, 'AlphaData', 10.^-stds);


if strcmp(deblank(truth.priors(idx_x, :)), 'log')
    set(gca, 'Xscale', 'log');
end
if strcmp(deblank(truth.priors(idx_y, :)), 'log')
    set(gca, 'Yscale', 'log');
end
set(gca, 'Zscale', 'log');

% Create figure window and components

% fig = uifigure('Position',[100 100 350 275]);
% 
% cg = uigauge(fig,'Position',[100 100 120 120]);
% 
% sld = uislider(fig,...
%     'Position',[100 75 120 3],...
%     'ValueChangingFcn',@(sld,event) updateGauge(event,cg));
% 
% % Create ValueChangedFcn callback
% function updatePlot(event,cg)
%     cg.Value = event.Value;
% end