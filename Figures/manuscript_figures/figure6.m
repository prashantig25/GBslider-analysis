%% INITIALISE VARS
clc
clearvars
colors_manuscript; % colors for plot

% LOAD DATA
load("lm_esterror_signed.mat","lm"); % estimated model fit to estimation errors
load("partialrsq_signed.mat","partial_rsq"); % partial R2 values

linewidth_line = 1.5; % linewidth for plot lines
line_width = 0.5; % linewidth for axes
font_name = 'Arial'; % font name
font_size = 6; % font size
xlim_vals = [0.5 1.5]; % x limits
ylim_vals = [0.5 0.9]; % y limits

%% INITIALISE FIGURES

figure 
set(gcf,'Position',[100 100 400 200])
t = tiledlayout(1,2);
t.Padding = 'compact';
ax2 = nexttile(2,[1,1]);
ax1 = nexttile(1,[1,1]);

%% PLOT

% ILLUSTRATION OF OVER-ESTIMATION DRIVEN BY CONFIRMATION BIAS
mu = [0.73 0.77]; % example estimated mu to show over-estimation

line(ax1,[0 4],[0.8 0.8],'Color',dots_edges,'LineWidth',1.5) % actual mu
b = bar(ax1,1,mu,0.7,"grouped","EdgeColor",darkblue_muted, ...
    FaceColor=darkblue_muted,LineWidth=1); % estimate mu bars

% ADJUST PLOT PROPERTIES
b(1).FaceAlpha = 0.3;
b(2).FaceAlpha = 0.6;
set(ax1,'XTick',[]);
adjust_figprops(ax1,font_name,font_size,line_width,xlim_vals,ylim_vals)
box(ax1,"off")
l = legend("Actual \mu","Estimated \mu (no confirmation-bias LR)", ...
    "Estimated \mu (low confirmation-bias LR)","Location","best","EdgeColor","none","Color","none");
l.ItemTokenSize = [7,7];
ylabel('\mu')

% RELATIONSHIP BETWEEN SIGNED EST. ERROR vs. CONFIRMATION-BIAS LR
p = plotAdded(ax2,lm,[5],'Marker','o','MarkerSize', ...
        3,'MarkerFaceColor',gray_dots, ...
            'MarkerEdgeColor',dots_edges);

% ADJUST PLOT PROPERTIES
p(1).Color = reg_color;
p(2).Color = reg_color;
p(3).Color = reg_color;
p(2).LineWidth = linewidth_line;
legend(ax2,'off')
xlabel(ax2,"Adjusted confirmation-bias LR")
ylabel(ax2,"Adjusted signed estimation error")
title(ax2,strcat('Partial R^2 =',{' '},num2str(sprintf('%.2f', partial_rsq(3))), ...
    {' '},'p = ',{' '},num2str(sprintf('%.2f', lm.Coefficients.pValue(5)))), ...
'FontWeight','normal','Interpreter','tex')
adjust_figprops(ax2,font_name,font_size,line_width,xlim_vals,ylim_vals)
box(ax2,"off")

%% ADD SUBPLOT LABELS

ax1_pos = ax1.Position;
adjust_x = -0.08;
adjust_y = ax1_pos(4) + 0.052;
[label_x,label_y] = change_plotlabel(ax1,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'a','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(ax2,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'b','FontSize',12,'LineStyle','none','HorizontalAlignment','center')
%% SAVE FIGURE

fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, 'figure4_signed_est_bars.png', '-dpng', '-r600') 