%% INITIALISE GENERAL PLOT VARS
clc
clearvars

colors_manuscript; % colors for plot
linewidth_line = 1.5; % linewidth for plot lines
line_width = 0.5; % linewidth for axes
font_name = 'Arial'; % font name
font_size = 6; % font size
xlim_vals = [0 1]; % x limits
ylim_vals = [0 0.8]; % y limits
load("lm_esterror_abs_lr.mat","lm"); % estimated model fit to estimation errors
load("partialrsq_signed.mat","partial_rsq"); % partial R2 values
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

variables = [2,3,6];
axes_variables = [ax1,ax2,ax3];
xlabels_variables = ["Adjusted fixed LR","Adjusted BS adapted LR","Adjusted confirmation LR"];
ylabels_variables = ["Adjusted estimation error"," "," "," "," "];

for v = 1:length(variables)
    hold on
    p = plotAdded(axes_variables(v),lm,[variables(v)],'Marker','o','MarkerSize', ...
        3,'MarkerFaceColor',gray_dots, ...
            'MarkerEdgeColor',dots_edges);
    p(1).Color = reg_color;
    p(2).Color = reg_color;
    p(3).Color = reg_color;
    p(2).LineWidth = linewidth_line;
    legend(axes_variables(v),'off')
    xlabel(axes_variables(v),xlabels_variables(v))
    ylabel(axes_variables(v),ylabels_variables(v))
    title(axes_variables(v),strcat('Partial R^2 =',{' '},num2str(sprintf('%.2f', partial_rsq(v))),{' '},...
        'p = ',{' '},num2str(sprintf('%.2f', lm.Coefficients.pValue(variables(v))))), ...
    'FontWeight','normal','Interpreter','tex')
    adjust_figprops(axes_variables(v),font_name,font_size,line_width,xlim_vals,ylim_vals)
    box(axes_variables(v),"off")
end
%% ADD SUBPLOT LABELS

ax1_pos = ax1.Position;
adjust_x = -0.06;
adjust_y = ax1_pos(4) + 0.05;
[label_x,label_y] = change_plotlabel(ax1,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'a','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(ax2,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'b','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(ax3,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'c','FontSize',12,'LineStyle','none','HorizontalAlignment','center')
%% SAVE FIGURE

fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, 'figure4_est2.png', '-dpng', '-r600') 