clc
clearvars
colors_manuscript; % colors for the plot
line_width = 0.5; % line width for axes
font_size = 6; % font size
font_name = 'Arial'; % font name
linewidth_plot = 1; % line width for plot lines
fontsize_label = 12; % font size for subplot labels

% INITIALISE VARS
load("betas_signed_recoding_wo_rewunc.mat","betas_all"); betas_signed = betas_all; % participant betas from signed analysis
load("betas_abs_recoding_wo_rewunc.mat","betas_all"); betas_abs = betas_all; % participant betas from signed analysis
data_subjs = readtable("data_recoding_signed.xlsx");
num_subjs = 99; % number of subjects
selected_regressors = [1,2,5,3,4];
dot_size = 10;

load("betas_signed_recoding_wo_rewunc.mat","betas_all"); betas_signed = betas_all;
load("betas_abs_recoding_wo_rewunc.mat","betas_all"); betas_abs = betas_all;
num_vars = 5;
xlim_vals = [0.5,5.5];
ylim_vals = [0.5,5.5];
colorbar_pos = [0.5,0.1,0.035,0.35];
%% INITIALISE TILE LAYOUT

figure
set(gcf,'Position',[100 100 400 400])
t = tiledlayout(2,2);
% t.TileSpacing = 'compact';
t.Padding = 'compact';

%% PLOT ALL BETA COEFFICIENTS

% TILE
ax1 = nexttile(1,[1,2]);

% GET MEAN AND SEM FOR BETAS
[mean_signed,sem_signed,coeffs_signed] = prepare_betas(betas_signed,selected_regressors,num_subjs);
[mean_abs,sem_abs,coeffs_abs] = prepare_betas(betas_abs,selected_regressors,num_subjs);
coeffs_subjs = [coeffs_signed,coeffs_abs];
mean_avg = [mean_signed,mean_abs];
mean_sd = [sem_signed,sem_abs];

% GET STARS FOR CORRESPONDING REGRESSOR'S P-VALUES
bar_labels = {'','','','',''};
pstars = bar_labels;

% OTHER FIGURE PROPERTIES
xticks = [1:length(selected_regressors)]; % x-axis ticks
row1 = {'Fixed LR' 'BS adapted LR' 'Confirmation' 'Salience' 'Congruence'};
row2 = {'' '' 'bias' 'adapted LR' 'adapted LR'};
labelArray = [row1; row2;]; 
tickLabels = strtrim(sprintf('%s\\newline%s\n', labelArray{:}));
xticklabs = tickLabels; % x-axis tick labels
title_name = {''}; % title
xlabelname = {'Regressor'}; % x-axis label
ylabelname = {'Mean beta coefficients'}; % y-axis label
colors_name = [fits_colors;darkblue_muted]; % colors for bars
y_label = repelem(1,1,length(selected_regressors)); % location for p-value stars
disp_pval = 0; % if p-value should be displayed

% PLOT
hold on
bar_plots_pval(coeffs_subjs,mean_avg,mean_sd,num_subjs, ...
    length(selected_regressors),2,{'Relative','Absolute'}, ...
    xticks,xticklabs,title_name,xlabelname, ...
    ylabelname,disp_pval,1,5,1,font_size,line_width,font_name,1,colors_name,pstars, ...
    y_label,[NaN,NaN,NaN,NaN,NaN],[0.5,5.5]) 
set(gca,'Color','none')

%% PLOT CORELATION BETWEEN PARAMETERS

% COMPUTE CORRELATION
correlation_signed = corrcoef(betas_signed);
correlation_abs = corrcoef(betas_abs);

% INITIALIZE TILES
ax3 = nexttile(3,[1 1]);
ax4 = nexttile(4,[1 1]);

% PLOT
imagesc(ax3,correlation_signed) % correlation matrix 
colormap(ax3,flipud(bone)); % set colormap
cb = colorbar(ax3); % set colorbar 
cb.Position = colorbar_pos; % set position
xticklabels(ax3,{'Fixed LR','Belief states','Salience','Congruence','Confirmation'}) % x-axis tick labels
yticklabels(aX3,{'Fixed LR','Belief states','Salience','Congruence','Confirmation'}) % y-axis tick labels
title(ax3,'Signed LR analysis','FontWeight','normal') % title
adjust_figprops(ax3,font_name,font_size,line_width,xlim_vals,ylim_vals) % adjust figure properties
box(ax3,'off')

imagesc(ax4,correlation_abs)
colormap(ax4,flipud(bone));
adjust_figprops(ax4,font_name,font_size,line_width,xlim_vals,ylim_vals)
xticklabels(ax4,{'Fixed LR','Belief states','Salience','Congruence','Confirmation'})
title(ax4,'Absolute LR analysis','FontWeight','normal')
set(ax4,'YTickLabel', [])
box(ax4,'off')
%% ADD SUBPLOT LABELS

ax1_pos = ax1.Position;
adjust_x = - 0.06;
adjust_y = ax1_pos(4);

all_axes = [ax1,ax3,ax4];
subplot_labels = {'a','b','c'};
for i = 1:3
    [label_x,label_y] = change_plotlabel(all_axes(i),adjust_x,adjust_y);
    annotation("textbox",[label_x label_y .05 .05],'String', ...
        subplot_labels{i},'FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment','center')
end
%%
fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, 'figure2_corr_SM1.png', '-dpng', '-r600') 