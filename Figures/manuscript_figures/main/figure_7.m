%% INITIALISE GENERAL PLOT VARS
clc
clearvars

linewidth_line = 1.5; % linewidth for plot lines
line_width = 0.5; % linewidth for axes
linewidth_single = 0.005;
font_name = 'Arial'; % font name
font_size = 6; % font size
xlim_vals = [0 1]; % x limits
ylim_vals = [0 1.2]; % y limits
num_subjs = 98; % number of subjects
num_conds = 3; % number of condition
[~,~,~,~,~,~,darkblue_muted,~,~,~,~,~,~,~,~,~,~,~,~] = colors_rgb(); % colors

% directory specification
current_Dir = pwd;
save_dir = fullfile("saved_figures",filesep,"main");
mkdir(save_dir)

ecoperf_mix = importdata("Data/descriptive data/main study/ecoperf_mix.mat"); % economic performance for mixed condition
ecoperf_perc = importdata("Data/descriptive data/main study/ecoperf_perc.mat");
ecoperf_rew = importdata("Data/descriptive data/main study/ecoperf_rew.mat");
%% INITIALISE TILE LAYOUT

figure
set(gcf,'Position',[100 100 250 200])
t = tiledlayout(1,1);
t.Padding = 'compact';
t.TileSpacing = 'compact';
%% PLOT SALIENCE BIAS

% COMPUTE SALIENCE BIAS
bias_mix = ecoperf_mix(:,2) - ecoperf_mix(:,1);
bias_perc = ecoperf_perc(:,2) - ecoperf_perc(:,1);
bias_rew = ecoperf_rew(:,2) - ecoperf_rew(:,1);

% MEAN AND SEM OF SALIENCE BIAS
y = [bias_mix;bias_perc;bias_rew];
mix_avg = nanmean(bias_mix,1);
perc_avg = nanmean(bias_perc,1);
rew_avg = nanmean(bias_rew,1);
mean_avg = [mix_avg; perc_avg; rew_avg];

mix_sd = nanstd(bias_mix,1)/sqrt(num_subjs);
perc_sd = nanstd(bias_perc,1)/sqrt(num_subjs);
rew_sd = nanstd(bias_rew,1)/sqrt(num_subjs);
mean_sd = [mix_sd; perc_sd; rew_sd];

% SIGNIFICANCE TESTING
p_vals = NaN(1,num_conds);
[~,p_vals(1,1),~,~] = ttest(bias_mix);
[~,p_vals(1,2),~,~] = ttest(bias_perc);
[~,p_vals(1,3),~,~] = ttest(bias_rew);

p_vals = p_vals(:,:);
bar_labels = {'','',''};
bar_labels = pvals_stars(p_vals,1:3,bar_labels,0);

% PLOT PROPERTIES
xticks = [1:length(mean_sd)]; % array of x-axis ticks
colors_name = [darkblue_muted]; % colors for bars
y_label = repelem(1.05,num_conds,1); % y-axis position for p-val stars
disp_pval = 1; % whether to display p-val stars
xticklabs = {'Both','Perceptual','Reward'}; % x-axis tick labels
title_name = {''}; % title string for plot
legend_names = {'Low-contrast blocks','High-contrast blocks'}; % legend names
xlabelname = {'Type of uncertainty'}; % x-axis label
ylabelname = {'Salience bias'}; % y-axis label

ax1 = nexttile(1,[1,1]);
hold on
h = bar_plots_pval(y,mean_avg,mean_sd,num_subjs,length(mean_avg),1, ...
    legend_names,xticks,xticklabs,title_name,xlabelname,ylabelname,disp_pval,1,5,1, ...
    font_size,line_width,font_name,0, ...
    colors_name,bar_labels(1:num_conds),y_label);
hold on 
h.BarWidth = 0.4; 
ylim([-0.5,1.1])
xlim([0.5,3.5])
set(gca,'Color','none')

%% SAVE FIGURE

fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, fullfile(save_dir,filesep,'bias.png'), '-dpng', '-r600') 