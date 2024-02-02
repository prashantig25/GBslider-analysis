clc
clearvars
colors_manuscript; % colors for the plot
line_width = 0.5; % line width for axes
font_size = 6; % font size
font_name = 'Arial'; % font name
linewidth_plot = 1; % line width for plot lines
fontsize_label = 12; % font size for subplot labels
dot_size = 5; % size of dots for bars

% INITIALISE VARS
load("betas_signed_recoding_wo_subjest.mat","betas_all"); betas_signed = betas_all; % participant betas from signed analysis
load("betas_abs_recoding_wo_subjest.mat","betas_all"); betas_abs = betas_all; % participant betas from signed analysis
num_subjs = 99; % number of subjects
selected_regressors = [1,2,6,3,4,5]; % list of regressors
%% INITIALISE TILE LAYOUT

figure
set(gcf,'Position',[100 100 400 200])
t = tiledlayout(1,1);

%% PLOT ALL BETA COEFFICIENTS

% TILE
ax1 = nexttile(1,[1,1]);

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
row1 = {'Fixed LR' 'BS adapted LR' 'Confirmation' 'Salience' 'Congruence' 'Risk '};
row2 = {'' '' 'bias' 'adapted LR' 'adapted LR' 'adapted LR'};
labelArray = [row1; row2;]; 
tickLabels = strtrim(sprintf('%s\\newline%s\n', labelArray{:}));
xticklabs = tickLabels; % x-axis tick labels
title_name = {''}; % title string
xlabelname = {'Regressor'}; % x-axis label name
ylabelname = {'Mean beta coefficients'}; % y-axisx label name
colors_name = [fits_colors;darkblue_muted]; % colors for bars
y_label = repelem(1,1,length(selected_regressors)); % location for p-values stars
disp_pval = 0; % if p-values should be displayed

% PLOT
hold on
bar_plots_pval(coeffs_subjs,mean_avg,mean_sd,num_subjs, ...
    length(selected_regressors),2,{'Relative','Absolute'}, ...
    xticks,xticklabs,title_name,xlabelname, ...
    ylabelname,disp_pval,1,dot_size,1,font_size,line_width,font_name,1,colors_name,pstars, ...
    y_label,[NaN,NaN,NaN,NaN,NaN,NaN],[0.5,6.5]) 
set(gca,'Color','none')
ylim(gca,[-0.5 0.5])
%%
fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, 'figureSM_rewunc.png', '-dpng', '-r600') 