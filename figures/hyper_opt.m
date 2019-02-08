load hyper_opt.mat;

colormap(jet);

c = linspace(0, 1, size(y, 2));
truth = scatter3(log(x(:, 1)), x(:, 2), log(y), size(y, 2), c, 'filled');

hold on;

maxIter = 10.^linspace(1, 26/20+1, 27);
alpha = 10.^linspace(-1, 40/20-1, 41);
surrogate = surf(log(alpha), maxIter, log(surface));

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