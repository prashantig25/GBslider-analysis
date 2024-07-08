clc
clearvars

line_width = 0.5; % line width for axes
font_size = 6; % font size
font_name = 'Arial'; % font name
linewidth_plot = 1; % line width for plot lines
fontsize_label = 12; % font size for subplot labels
dot_size = 5; % size of dots for bars
[~,~,~,~,~,~,darkblue_muted,~,~,~,~,~,~,~,~,~,~,fits_colors,~] = colors_rgb(); % colors

% INITIALISE VARS
load("Data/LR analyses/betas_signed_obj.mat","betas_all"); betas_signed = betas_all; % participant betas from signed analysis
load("Data/LR analyses/betas_abs_obj.mat","betas_all"); betas_abs = betas_all; % participant betas from signed analysis
load("Data/LR analyses/betas_signed_salience.mat","betas_all"); betas_signed_salience = betas_all; % participant betas from signed analysis of salience bias in learning
load("Data/LR analyses/betas_abs_salience.mat","betas_all"); betas_abs_salience = betas_all; % participant betas from absolute analysis of salience bias in learning
num_subjs = 98; % number of subjects
selected_regressors = [1,2,6,3,4,5]; % list of regressors
%% INITIALISE TILE LAYOUT

figure
set(gcf,'Position',[100 100 400 250])
t = tiledlayout(1,2);
ax1 = nexttile(1,[1,1]);
ax2 = nexttile(2,[1,1]);
%% PLOT ALL BETA COEFFICIENTS

% TILE
position_change = [-0.06, 0.1, 0.4, -0.1]; % change position
new_pos = change_position(ax1,position_change); % new position
ax1_new = axes('Units', 'Normalized', 'Position', new_pos); % updated position
box(ax1_new, 'off'); % box off
delete(ax1); % delete old axis

% GET MEAN AND SEM FOR BETAS
[mean_signed,sem_signed,coeffs_signed] = prepare_betas(betas_signed,selected_regressors,num_subjs);
[mean_abs,sem_abs,coeffs_abs] = prepare_betas(betas_abs,selected_regressors,num_subjs);
coeffs_subjs = [coeffs_signed,coeffs_abs];
mean_avg = [mean_signed,mean_abs];
mean_sd = [sem_signed,sem_abs];

% GET STARS FOR CORRESPONDING REGRESSOR'S P-VALUES
bar_labels = {'','','','','',''};
pstars = bar_labels;

% OTHER FIGURE PROPERTIES
xticks = [1:length(selected_regressors)]; % x-axis ticks
row1 = {'Fixed LR' 'Belief-state' 'Confirmation' 'Salience' 'Congruence' 'Risk '};
row2 = {'' '-adapted LR' 'bias' '' '' ''};
labelArray = [row1; row2;]; 
xticklabs = {'','','','','',''}; % x-axis tick labels
title_name = {''}; % title string
xlabelname = {''}; % x-axis label name
ylabelname = {'Mean beta coefficients'}; % y-axisx label name
colors_name = [fits_colors;darkblue_muted]; % colors for bars
y_label = repelem(1,1,length(selected_regressors)); % location for p-values stars
disp_pval = 0; % if p-values should be displayed

% PLOT
hold on
bar_plots_pval(coeffs_subjs,mean_avg,mean_sd,num_subjs, ...
    length(selected_regressors),2,{'Signed','Absolute'}, ...
    xticks,xticklabs,title_name,xlabelname, ...
    ylabelname,disp_pval,1,dot_size,1,font_size,line_width,font_name,1,colors_name,pstars, ...
    y_label,[NaN,NaN,NaN,NaN,NaN,NaN],[0.5,6.5]) 

% PLOT PROPERTIES
set(gca,'Color','none')
ax = gca;
ax.XTick = [1 2 3 4 5 6];
ax.XTickLabel = '';
myLabels = labelArray;
for i = 1:length(myLabels)
    text(i, ax.YLim(1), sprintf('%s\n%s\n%s', myLabels{:,i}), ...
        'horizontalalignment', 'center', 'verticalalignment', 'top','FontSize',font_size);    
end
%% PLOT SALIENCE BIAS IN LEARNING

% CHANGE TILE POSITION
position_change = [0.26, 0.1, -0.17, -0.1];
new_pos = change_position(ax2,position_change);
ax2_new = axes('Units', 'Normalized', 'Position', new_pos);
box(ax2_new, 'off');
delete(ax2);

% GET MEAN AND SEM FOR BETAS
selected_regressors = 3;
[mean_signed,sem_signed,coeffs_signed] = prepare_betas(betas_signed_salience,selected_regressors,num_subjs);
[mean_abs,sem_abs,coeffs_abs] = prepare_betas(betas_abs_salience,selected_regressors,num_subjs);
coeffs_subjs = [coeffs_signed,coeffs_abs];
mean_avg = [mean_signed,mean_abs];
mean_sd = [sem_signed,sem_abs];

% GET STARS FOR CORRESPONDING REGRESSOR'S P-VALUES
bar_labels = {''};
pstars = bar_labels;

% OTHER FIGURE PROPERTIES
xticks = [1:length(selected_regressors)]; % x-axis ticks
row1 = {'Salience-bias'};
row2 = {''};
labelArray = [row1; row2;]; 
xticklabs = {''}; % x-axis tick labels
title_name = {''}; % title string
xlabelname = {''}; % x-axis label name
ylabelname = {''}; % y-axisx label name
colors_name = [fits_colors;darkblue_muted]; % colors for bars
y_label = repelem(1,1,length(selected_regressors)); % location for p-values stars
disp_pval = 0; % if p-values should be displayed

% PLOT
hold on
bar_plots_pval(coeffs_subjs,mean_avg,mean_sd,num_subjs, ...
    length(selected_regressors),2,{'',''}, ...
    xticks,xticklabs,title_name,xlabelname, ...
    ylabelname,disp_pval,1,dot_size,1,font_size,line_width,font_name,0,colors_name,pstars, ...
    y_label,[NaN,NaN],[0.5,1.5]) 
set(gca,'Color','none')
ylim(gca,[-0.25 0.25])


ax = gca;
ax.XTick = [1];
ax.XTickLabel = '';
myLabels = labelArray;
for i = 1
    text(i, ax.YLim(1), sprintf('%s\n%s\n%s', myLabels{:,i}), ...
        'horizontalalignment', 'center', 'verticalalignment', 'top','FontSize',font_size);    
end
%% ADD SUBPLOT LABELS

ax1_pos = ax2_new.Position;
adjust_x = [- 0.07,-0.08];
adjust_y = ax1_pos(4)+0.05;

all_axes = [ax1_new,ax2_new];
subplot_labels = {'a','b'};
for i = 1:2
    [label_x,label_y] = change_plotlabel(all_axes(i),adjust_x(i),adjust_y);
    annotation("textbox",[label_x label_y .05 .05],'String', ...
        subplot_labels{i},'FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment','center')
end
%%
fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, 'figure3_SM2.png', '-dpng', '-r600') 