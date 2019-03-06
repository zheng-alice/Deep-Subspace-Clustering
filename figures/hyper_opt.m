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

if axxes.Count < 2
    error("Data has too few dimensions: " + axxes.Count);
end

for i = 1:4
    signif(i) = mean2(std(surrogate.mean, 0, i));
end
[signif, order] = sort(signif, 'descend');
idx_x = order(1);
idx_y = order(2);


% Default position
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
c = linspace(0, 1, size(truth.y, 1));
plot_truth = scatter3(truth.x(:, idx_x), truth.x(:, idx_y), truth.y, 50, c, 'filled');
clear c;

hold on;

axis_x = axxes(truth.names{idx_x});
axis_y = axxes(truth.names{idx_y});
plot_surrogate = surf(axis_x, axis_y, zeros(size(axis_y, 1), size(axis_x, 1)));
plot_surrogate.EdgeAlpha = 'interp';
plot_surrogate.FaceColor = 'interp';
plot_surrogate.FaceAlpha = 'interp';
clear axis_x axis_y;

ax = gca;
if strcmp(truth.priors{idx_x}, 'log')
    ax.XScale = 'log';
end
if strcmp(truth.priors{idx_y}, 'log')
    ax.YScale = 'log';
end
ax.ZScale = 'log';
ax.Parent.Position = [650 60 700 685];

string = replace(truth.names{idx_x}, '_', '-');
string(1) = upper(string(1));
ax.XLabel.String = string;
string = replace(truth.names{idx_y}, '_', '-');
string(1) = upper(string(1));
ax.YLabel.String = string;
ax.ZLabel.String = "Error";
clear string;


% Package variables
fig = uifigure('Position', [25 75 600 65+65*(size(order, 2)-2)]);
data = guihandles(fig);
data.scatter = plot_truth;
data.surf = plot_surrogate;
data.best = best;
data.slice = slice;
data.idx_x = idx_x;
data.idx_y = idx_y;
guidata(fig, data);
clear plot_truth plot_surrogate;
clear best slice;
clear idx_x idx_y;


% Create GUI
for i = 3:size(order, 2)
    idx = order(i);
    axisName = truth.names{idx};
    label = uilabel(fig,...
        'Position', [25, 65*(size(order, 2)-i)+25+30, 550, 22],...
        'Text', axisName);
    label.FontSize = 14;
    axis = axxes(axisName);
    majorIdx = uint8(linspace(1, size(axis, 1), 10));
    sld = uislider(fig,...
        'Position', [25, 65*(size(order, 2)-i)+25+25, 550, 3],...
        'Value', data.best(idx),...
        'Limits', [1, size(axis, 1)],...
        'MajorTicks', majorIdx,...
        'MajorTickLabels', replace(replace(cellstr(num2str(axis(majorIdx), '%-.3G')), 'E+0', 'E+'), 'E-0', 'E-'),...
        'ValueChangingFcn', @updateSlice,...
        'ValueChangedFcn', @discretize);
    sld.MinorTicks = 1:size(axis, 1);
    sld.UserData = idx;
end
clear i idx majorIdx;
clear axis axisName;
clear label sld;

[~] = uibutton(fig,...
    'Position', [25 65*(size(order, 2)-2)+25 100 25],...
    'Text', "Reset",...
    'ButtonPushedFcn', @(button, event) resetSlice(fig));
updatePlot(data.scatter, data.surf, data.slice, data.idx_x, data.idx_y);
clear data;



% Callbacks
function updateSlice(sld, event)
    data = guidata(sld.Parent);
    if data.slice{sld.UserData} ~= round(event.Value)
        data.slice{sld.UserData} = round(event.Value);
        
        updatePlot(data.scatter, data.surf, data.slice, data.idx_x, data.idx_y);
        guidata(sld.Parent, data);
    end
end

function discretize(sld, event)
    sld.Value = round(event.Value);
end

function resetSlice(fig)
    data = guidata(fig);
    for sld = fig.Children'
        if strcmp(sld.Type, 'uislider')
            idx = sld.UserData;
            data.slice{idx} = data.best(idx);
            sld.Value = data.best(idx);
        end
    end
    
    updatePlot(data.scatter, data.surf, data.slice, data.idx_x, data.idx_y);
    guidata(fig, data);
end

% Update surface, scatter size
function updatePlot(plot_truth, plot_surrogate, slice, idx_x, idx_y)
    global axxes;
    global truth;
    global surrogate;
    
    % update scatter size - decreases with distance to visible plane
    s = 50.* ones(size(plot_truth.CData, 2), 1);
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
    plot_truth.SizeData = s;
    
    % update surface slice
    means = squeeze(surrogate.mean(slice{:}));
    stds = squeeze(surrogate.std(slice{:}));
    if idx_x < idx_y
        means = means.';
        stds = stds.';
    end
    plot_surrogate.ZData = means;
    plot_surrogate.AlphaData = 32.^-stds;
    plot_surrogate.AlphaDataMapping = 'none';
end