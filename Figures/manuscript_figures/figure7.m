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
num_conds = 3; % number of condition
load("ecoperf_mix.mat"); % economic performance for mixed condition
load("ecoperf_perc.mat");
load("ecoperf_rew.mat");
load("betas_salience.mat") % beta coefficient from salience bias analysis
load("betas_salience_grouped.mat") % beta coefficient from grouped salience bias analysis
%% INITIALISE TILE LAYOUT

figure
set(gcf,'Position',[100 100 600 250])
t = tiledlayout(1,2);
t.Padding = 'compact';
t.TileSpacing = 'compact';
%% PLOT SALIENCE BIAS

% MEAN AND SEM
y = [ecoperf_mix;ecoperf_perc;ecoperf_rew];
mix_avg = nanmean(ecoperf_mix,1);
perc_avg = nanmean(ecoperf_perc,1);
rew_avg = nanmean(ecoperf_rew,1);
mean_avg = [mix_avg; perc_avg; rew_avg];

mix_sd = nanstd(ecoperf_mix,1)/sqrt(num_subjs);
perc_sd = nanstd(ecoperf_perc,1)/sqrt(num_subjs);
rew_sd = nanstd(ecoperf_rew,1)/sqrt(num_subjs);
mean_sd = [mix_sd; perc_sd; rew_sd];

% SIGNIFICANCE TESTING
p_vals = NaN(1,num_conds);
[~,p_vals(1,1),~,~] = ttest2(ecoperf_mix(:,2),ecoperf_mix(:,1));
[~,p_vals(1,2),~,~] = ttest2(ecoperf_perc(:,2),ecoperf_perc(:,1));
[~,p_vals(1,3),~,~] = ttest2(ecoperf_rew(:,2),ecoperf_rew(:,1));
p_vals = p_vals(:,:);
bar_labels = {'','',''};
bar_labels = pvals_stars(p_vals,1:3,bar_labels,0);

% PLOT PROPERTIES
xticks = [1:length(mean_sd)]; % array of x-axis ticks
colors_name = [lowsal_colors;highsal_colors]; % colors for bars
y_label = repelem(1.01,num_conds,1); % y-axis position for p-val stars
disp_pval = 1; % whether to display p-val stars
xticklabs = {'Both','Perceptual','Reward'}; % x-axis tick labels
title_name = {''}; % title string for plot
legend_names = {'Low contrast','High contrast'}; % legend names
xlabelname = {'Type of uncertainty'}; % x-axis label
ylabelname = {'Mean economic performance'}; % y-axis label

ax1 = nexttile(1,[1,1]);
hold on
h = bar_plots_pval(y,mean_avg,mean_sd,num_subjs,length(mean_avg),length(legend_names), ...
    legend_names,xticks,xticklabs,title_name,xlabelname,ylabelname,disp_pval,0,0,1, ...
    font_size,line_width,font_name,1, ...
    colors_name,bar_labels(1:num_conds),y_label);
hold on 

% PLOT LINES CONNECTING SINGLE DOTS
adjust_lines = 0.15; % adjusted for single plot lines
start = 1; end_num = num_subjs; % start and end for y-data

for c = 1:num_conds
    bar_points = [c - adjust_lines, c + adjust_lines]; % adjust x-axis location
    x_rep = repmat(bar_points,num_subjs,1);
    y_bar = y(start:end_num,:); % y-data
    hold on
    for i = 1:length(y_bar)
        plot(x_rep(i,:),y_bar(i,:), '-', 'color', light_gray, 'linewidth', linewidth_single,'Marker','none');
        hold on
    end
    start = end_num + 1; % for next bar
    end_num = end_num + num_subjs;
end
ylim([0.5,1.05])
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
[mean_avg,mean_sd,coeffs_subjs] = prepare_betas(betas_all,selected_regressors,num_subjs);

% PLOT PROPERTIES
xticks = 1:length(selected_regressors); % x-axis ticks
xticks_label = {'High contrast','High PU','High RU',}; % x-axis tick labels
p_vals_regressor = p_vals(:,selected_regressors); % p-value for 
bar_labels = {'','',''};
bar_labels = pvals_stars(p_vals,1:3,bar_labels,0); % stars for p-vals
title_name = {''}; % plot title string
colors_name = barface_green; % bar color
y_label = []; % y-axis label for p-val stars
for i = xticks
    if mean_avg(i) > 0
        y_label = [y_label,mean_avg(i)+mean_sd(i)];
    else
        y_label = [y_label,mean_avg(i)-mean_sd(i)-0.02];
    end
end
y_label = repelem(0.4,num_vars,1);
disp_pval = 1; % whether p-val stars should be plotted
agent_means = NaN(length(selected_regressors),1).'; % betas for agent

% PLOT
ax2 = nexttile(2,[1,1]);
hold on
h = bar_plots_pval(coeffs_subjs,mean_avg,mean_sd,num_subjs, ...
    length(selected_regressors),1,{'Empirical data'}, ...
    xticks,xticks_label,title_name,xlabelname,'Mean beta coefficients', ...
    disp_pval,1,5,1,6,line_width,'arial',0,colors_name, ...
    bar_labels,y_label,agent_means,[0.5,3.5]);
h.BarWidth = 0.4; 
set(gca,'Color','none')
ylim([-0.4,0.5])

%% ADD SUBPLOT LABELS

ax1_pos = ax1.Position;
adjust_x = -0.06;
adjust_y = ax1_pos(4) + 0.03;
[label_x,label_y] = change_plotlabel(ax1,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'a','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(ax2,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'b','FontSize',12,'LineStyle','none','HorizontalAlignment','center')
%% SAVE FIGURE

fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, 'figure6_1.png', '-dpng', '-r600') 