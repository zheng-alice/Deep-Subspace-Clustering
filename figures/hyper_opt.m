clf('reset');
if exist('fig', 'var') && ishghandle(fig)
    close(fig);
end
clear all;

global axxes;
global truth;
global surrogate;
load hyper_opt.mat;
axxes = containers.Map(fieldnames(axxes), struct2cell(axxes));
truth.names = cellstr(truth.names);
truth.priors = cellstr(truth.priors);

idx_x = 1;
idx_y = 2;
if axxes.Count < 2
    error("Data has too few dimensions: " + axxes.Count);
end
if idx_x == idx_y
    error("Visualized dimension indices must differ");
end


% Default position
global M;
M = size(truth.y, 1);
best = truth.x(truth.best, :);
for i = 1:size(best, 2)
    % convert to surrogate indices
    [~, idx] = min(abs(axxes(truth.names{i})-best(i)));
    best(i) = idx;
end
slice = num2cell(best);
slice{idx_x} = ':';
slice{idx_y} = ':';


% Create actual plots
c = linspace(0, 1, M);
plot_truth = scatter3(truth.x(:, idx_x), truth.x(:, idx_y), truth.y, 50, c, 'filled');

hold on;

axis_x = axxes(truth.names{idx_x});
axis_y = axxes(truth.names{idx_y});
plot_surrogate = surf(axis_x, axis_y, zeros(size(axis_y, 1), size(axis_x, 1)));
set(plot_surrogate, 'EdgeAlpha', 'interp', 'FaceColor', 'interp', 'FaceAlpha', 'interp');


if strcmp(truth.priors{idx_x}, 'log')
    set(gca, 'Xscale', 'log');
end
if strcmp(truth.priors{idx_y}, 'log')
    set(gca, 'Yscale', 'log');
end
set(gca, 'Zscale', 'log');


updatePlot(plot_truth, plot_surrogate, slice, idx_x, idx_y);

fig = uifigure('Position', [100 100 400 300]);

asdf = 3;
axisName = truth.names{asdf};
label = uilabel(fig,...
    'Position', [25, 30+25, 350, 22],...
    'Text', axisName);
set(label, 'FontSize', 14);
axis = axxes(axisName);
sld = uislider(fig,...
    'Position', [25, 25+25, 350, 3],...
    'Value', best(asdf),...
    'Limits', [1, size(axis, 1)],...
    'MajorTicks', 1:size(axis, 1),...
    'MajorTickLabels', cellstr(num2str(axis, '%-.4G')),...
    'ValueChangedFcn', @(sld, event) updateSlice(sld, event, plot_truth, plot_surrogate, slice, idx_x, idx_y));
set(sld, 'MinorTicks', []);

asdf = 4;
axisName2 = truth.names{asdf};
label2 = uilabel(fig,...
    'Position', [25, 60+30+25, 350, 22],...
    'Text', axisName2);
set(label2, 'FontSize', 14);
axis2 = axxes(axisName2);
sld2 = uislider(fig,...
    'Position', [25, 60+25+25, 350, 3],...
    'Value', best(asdf),...
    'Limits', [1, size(axis2, 1)],...
    'MajorTicks', 1:size(axis2, 1),...
    'MajorTickLabels', cellstr(num2str(axis2, '%-.4G')),...
    'ValueChangedFcn', @(sld2, event) updateSlice(sld2, event, plot_truth, plot_surrogate, slice, idx_x, idx_y));
set(sld2, 'MinorTicks', []);

% Slider callback
function updateSlice(sld, event, plot_truth, plot_surrogate, slice, idx_x, idx_y)
    sld.Value = round(event.Value);
    if sld.Value ~= event.PreviousValue
        slice{3} = sld.Value;
        updatePlot(plot_truth, plot_surrogate, slice, idx_x, idx_y);
    end
end

% Update surface, scatter size
function updatePlot(plot_truth, plot_surrogate, slice, idx_x, idx_y)
    global axxes;
    global truth;
    global surrogate;
    global M;
    
    % update scatter size - decreases with distance to visible plane
    s = 50.* ones(M, 1);
    hidden = 1:axxes.Count;
    hidden([idx_x, idx_y]) = [];
    plane = 1:size(hidden, 2);
    range = 1:size(hidden, 2);
    values = truth.x(:, hidden);
    for i = plane
        axis = axxes(truth.names{hidden(i)});
        plane(i) = axis(slice{hidden(i)});
        range(i) = axis(size(axis, 1));
        if strcmp(truth.priors{hidden(i)}, 'log')
           plane(i) = log10(plane(i));
           range(i) = log10(range(i)) - log10(axis(1));
           values(:, i) = log10(values(:, i));
        else
           range(i) = range(i) - axis(1);
        end
    end
    dist = sqrt(sum(((values - plane)./ range).^2, 2));
    s = s.* 10.^ -dist;
    set(plot_truth, 'SizeData', s);
    
    % update surface slice
    means = squeeze(surrogate.mean(slice{:}));
    stds = squeeze(surrogate.std(slice{:}));
    if idx_x < idx_y
        means = means.';
        stds = stds.';
    end
    set(plot_surrogate, 'ZData', means, 'AlphaData', 10.^-stds, 'AlphaDataMapping', 'none');
end