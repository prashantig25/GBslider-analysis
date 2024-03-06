%% INITIALISE GENERAL PLOT VARS
clc
clearvars

colors_manuscript; % colors for plot
linewidth_line = 1.5; % linewidth for plot lines
line_width = 0.5; % linewidth for axes
linewidth_single = 0.05;
font_name = 'Arial'; % font name
font_size = 6; % font size
xlim_vals = [0 1]; % x limits
ylim_vals = [0 1.2]; % y limits
num_subjs = 98; % number of subjects
num_conds = 4; % number of condition
load("ecoperf_hh.mat"); % economic performance for pilot study for both uncertainties condition
load("ecoperf_hl.mat"); % perceptual condition
load("ecoperf_lh.mat"); % reward condition
load("ecoperf_ll.mat"); % no uncertainty condition
load("betas_salience.mat"); betas_study2 = betas_all; % beta coefficient from salience bias analysis from main task
load("betas_salience_study1.mat"); betas_all = [betas_all;NaN(5,3)]; % beta coefficient from salience bias analysis from pilot study
%% INITIALISE TILE LAYOUT

figure
set(gcf,'Position',[100 100 600 250])
t = tiledlayout(1,2);
t.Padding = 'compact';
t.TileSpacing = 'compact';
%% PLOT SALIENCE BIAS

% BIAS
bias_hh = ecoperf_hh(:,2)-ecoperf_hh(:,1);
bias_hl = ecoperf_hl(:,2)-ecoperf_hl(:,1);
bias_lh = ecoperf_lh(:,2)-ecoperf_lh(:,1);
bias_ll = ecoperf_ll(:,2)-ecoperf_ll(:,1);

% MEAN AND SEM
y = [bias_hh;bias_hl;bias_lh;bias_ll];
hh_avg = nanmean(bias_hh,1);
hl_avg = nanmean(bias_hl,1);
lh_avg = nanmean(bias_lh,1);
ll_avg = nanmean(bias_ll,1);
mean_avg = [hh_avg; hl_avg; lh_avg; ll_avg];

hh_sd = nanstd(bias_hh,1)/sqrt(num_subjs);
hl_sd = nanstd(bias_hl,1)/sqrt(num_subjs);
lh_sd = nanstd(bias_lh,1)/sqrt(num_subjs);
ll_sd = nanstd(bias_ll,1)/sqrt(num_subjs);
mean_sd = [hh_sd; hl_sd; lh_sd; ll_sd];

% SIGNIFICANCE TESTING
p_vals = NaN(1,num_conds);
[~,p_vals(1,1),~,~] = ttest(bias_hh,0);
[~,p_vals(1,2),~,~] = ttest(bias_hl,0);
[~,p_vals(1,3),~,~] = ttest(bias_lh,0);
[~,p_vals(1,4),~,~] = ttest(bias_ll,0);
p_vals = p_vals(:,:);
bar_labels = {'','','',''};
bar_labels = pvals_stars(p_vals,1:4,bar_labels,0);

% PLOT PROPERTIES
xticks = [1:length(mean_sd)]; % array of x-axis ticks
colors_name = [reg_color]; % colors for bars
y_label = repelem(0.55,num_conds,1); % y-axis position for p-val stars
disp_pval = 1; % whether to display p-val stars
xticklabs = {'Both','Perceptual','Reward','None'}; % x-axis tick labels
title_name = {''}; % title string for plot
legend_names = {'Low-contrast','High-contrast'}; % legend names
xlabelname = {'Type of uncertainty'}; % x-axis label
ylabelname = {'Mean economic performance'}; % y-axis label

% PLOT
ax1 = nexttile(1,[1,1]);
hold on
h = bar_plots_pval(y,mean_avg,mean_sd,num_subjs,length(mean_avg),1, ...
    legend_names,xticks,xticklabs,title_name,xlabelname,ylabelname,disp_pval,1,3,1, ...
    font_size,line_width,font_name,0, ...
    colors_name,bar_labels(1:num_conds),y_label);
hold on 

% PLOT PROPERTIES
h.BarWidth = 0.6;
ylim([-0.4,0.6])
xlim([0.5,4.5])
set(gca,'Color','none')
%% PLOT BETA COEFFICIENTS FROM SALIENCE BIAS ANALYSIS

num_vars = 3; % number of regressors
% SIGNIFICANCE TESTING
h_vals = nan(1,num_vars);
p_vals = nan(1,num_vars); 
t_vals = nan(1,num_vars);
for i = 1:num_vars
    [h_vals(1,i),p_vals(1,i),~,stats] = ttest(betas_all(:,i));
    t_vals(1,i) = stats.tstat;
end

% INITIALISE VARIABLES
selected_regressors = [3,2,1];
coeffs = [betas_all(:,selected_regressors); betas_study2(:,selected_regressors)]; 
coeffs_subjs = [];
for n = selected_regressors
    coeffs_subjs = [coeffs_subjs; betas_all(:,n), betas_study2(:,n)];
end

coeffs_avg = [nanmean(betas_all(:,selected_regressors),1); nanmean(betas_study2(:,selected_regressors),1)];
coeffs_sd = [nanstd(betas_all(:,selected_regressors),1); nanstd(betas_study2(:,selected_regressors),1)];
coeffs_sem = coeffs_sd./sqrt(num_subjs);

% MEAN and SEM
mean_avg_subjs = coeffs_avg.';
mean_sd_subjs = coeffs_sem.';

% PLOT PROPERTIES
xticks = 1:length(selected_regressors); % x-axis ticks
row1 = {'High contrast' 'High perceptual' 'High reward'};
row2 = {'' 'uncertainty' 'uncertainty'};
labelArray = [row1; row2;]; 
xticks_label = {'','',''};
p_vals_regressor = p_vals(:,selected_regressors);
bar_labels = {'','',''};
title_name = {''}; % plot title string
colors_name = [barface_green;study2_green]; % bar color
y_label = []; % y-axis label for p-val stars
for i = xticks
    if mean_avg(i) > 0
        y_label = [y_label,mean_avg_subjs(i)+mean_sd_subjs(i)];
    else
        y_label = [y_label,mean_avg_subjs(i)-mean_sd_subjs(i)-0.02];
    end
end
y_label = repelem(0.4,num_vars,1);
disp_pval = 1; % whether p-val stars should be plotted
agent_means = NaN(length(selected_regressors),1).'; % betas for agent

% PLOT
ax2 = nexttile(2,[1,1]);
hold on
h = bar_plots_pval(coeffs_subjs,mean_avg_subjs,mean_sd_subjs,num_subjs, ...
    length(selected_regressors),2,{'Pilot study','Main task'}, ...
    xticks,xticks_label,title_name,xlabelname,'Beta coefficient', ...
    disp_pval,1,5,1,6,line_width,'arial',1,colors_name, ...
    bar_labels,y_label,agent_means,[0.5,3.5]);
h(1).BarWidth = 0.8; 
set(gca,'Color','none')
ylim([-0.4,0.5])

ax = gca;
ax.XTick = [1 2 3];
ax.XTickLabel = '';
myLabels = labelArray;
for i = 1:length(myLabels)
    text(i, ax.YLim(1), sprintf('%s\n%s\n%s', myLabels{:,i}), ...
        'horizontalalignment', 'center', 'verticalalignment', 'top','FontSize',font_size);    
end
ax.XLabel.String = sprintf('\n\n%s', '');
%% ADD SUBPLOT LABELS

ax1_pos = ax1.Position;
adjust_x = [-0.06,-0.06]; % adjusted x-position for subplot label
adjust_y = ax1_pos(4) + 0.03; % adjusted y-position for subplot label
[label_x,label_y] = change_plotlabel(ax1,adjust_x(1),adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'a','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(ax2,adjust_x(2),adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'b','FontSize',12,'LineStyle','none','HorizontalAlignment','center')
%% SAVE FIGURE

fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, 'figure6_SM5.png', '-dpng', '-r600') 