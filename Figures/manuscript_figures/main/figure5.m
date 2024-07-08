%% INITIALISE GENERAL PLOT VARS
clc
clearvars

linewidth_line = 1.5; % linewidth for plot lines
line_width = 0.5; % linewidth for axes
font_name = 'Arial'; % font name
font_size = 6; % font size
xlim_vals = [-0.1 1]; % x limits
ylim_vals = [-0.1 0.8]; % y limits
load("Data/estimation error analysis/lm_abs_esterror_signed_lr.mat","lm"); % estimated model fit to estimation errors
load("Data/estimation error analysis/partialrsq_signed.mat","partial_rsq"); % partial R2 values
[~,~,~,~,~,~,~,~,~,~,~,~,~,~,reg_color,~,~,~,~] = colors_rgb(); % colors

pvals = lm.Coefficients.pValue; % get p-vals for estimated coefficients
pvals_cell = {'','','','','',''}; % initalise empty cell array to store p-values
for p = 1:length(pvals)
    if pvals(p) < 0.001
        pvals_cell{p} = '\itp\rm < 0.001';
    elseif pvals(p) < 0.01
        pvals_cell{p} = '\itp\rm < 0.01';
    elseif pvals(p) < 0.05 % for less than 0.05, use exact p-value
        pvals_cell{p} = strcat("\itp\rm = ",num2str(round(pvals(p),2)));
    else
        pvals_cell{p} = strcat("\itp\rm = ",num2str(round(pvals(p),2)));
    end
end
%% INITIALISE TILE LAYOUT

figure
set(gcf,'Position',[100 100 600 200])
t = tiledlayout(1,3);
t.Padding = 'compact';
t.TileSpacing = 'compact';
ax2 = nexttile(2,[1,1]);
ax3 = nexttile(3,[1,1]);
ax1 = nexttile(1,[1,1]);
%% ADDED VARIABLE PLOTS

variables = [2,3,6]; % variables to be plotted
axes_variables = [ax1,ax2,ax3]; % axes array
xlabels_variables = ["Fixed LR","Belief-state-adapted LR","Confirmation bias","Salience","Congruence"]; % x-axis labels
ylabels_variables = ["Absolute estimation error"," "," "," "," "]; % y-axis labels

for v = 1:length(variables)
    hold on
    p = plotAdded(axes_variables(v),lm,[variables(v)],'Marker','o','MarkerSize', ...
        3,'MarkerFaceColor',[220, 220, 220]./255, ...
        'MarkerEdgeColor',[184, 184, 184]./255);

    % PLOT PROPERTIES
    p(2).Color = reg_color;
    p(3).Color = reg_color;

    p(2).LineWidth = linewidth_line;
    p(1).LineWidth = 0.7;
    p(3).LineWidth = 0.2;

    p(3).LineStyle = '--';

    legend(axes_variables(v),'off')
    xlabel(axes_variables(v),xlabels_variables(v))
    ylabel(axes_variables(v),ylabels_variables(v))
    title(axes_variables(v),strcat("Partial \itR^2\rm =",{' '},num2str(partial_rsq(variables(v)-1),'%.3f')) + newline + strcat({' '},...
        pvals_cell{variables(v)}), ...
    'FontWeight','normal','Interpreter','tex')
    adjust_figprops(axes_variables(v),font_name,font_size,line_width,xlim_vals,ylim_vals)
    box(axes_variables(v),"off")
end
%% ADD SUBPLOT LABELS

ax1_pos = ax1.Position;
adjust_x = -0.055; % adjusted x-position for subplot label
adjust_y = ax1_pos(4) + 0.1; % adjusted y-position for subplot label
[label_x,label_y] = change_plotlabel(ax1,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'a','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(ax2,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'b','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(ax3,adjust_x,adjust_y);
%% SAVE FIGURE

fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, 'estimation_error.png', '-dpng', '-r600') 