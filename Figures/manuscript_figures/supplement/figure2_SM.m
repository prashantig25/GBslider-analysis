clc
clearvars

line_width = 0.5; % line width for axes
font_size = 6; % font size
font_name = 'Arial'; % font name
linewidth_plot = 1; % line width for plot lines
fontsize_label = 12; % font size for subplot labels
[~,~,~,~,~,~,darkblue_muted,~,~,~,~,~,~,~,...
    ~,~,~,fits_colors,~] = colors_rgb(); % colors

% INITIALISE VARS
load("Data/LR analyses/betas_signed_wo_rewunc_obj.mat","betas_all"); betas_signed = betas_all; % participant betas from signed analysis
load("Data/LR analyses/betas_abs_wo_rewunc_obj.mat","betas_all"); betas_abs = betas_all; % participant betas from absolute analysis
num_subjs = 98; % number of subjects
selected_regressors = [1,2,5,3,4]; % regressors to be plotted
num_vars = 5; % number of variables
xlim_vals = [0.5,5.5]; % x-limit values
ylim_vals = [0.5,5.5]; % y-axis limit values
colorbar_pos = [0.5,0.1,0.02,0.35]; % position for colorbar
%% INITIALISE TILE LAYOUT

figure
set(gcf,'Position',[100 100 400 400])
t = tiledlayout(2,2);
t.Padding = 'compact';

%% PLOT ALL BETA COEFFICIENTS

% TILE
ax1 = nexttile(1,[1,2]);

% GET MEAN AND SEM FOR BETAS
[mean_signed,sem_signed,coeffs_signed] = prepare_betas(betas_signed,selected_regressors,num_subjs); % signed
[mean_abs,sem_abs,coeffs_abs] = prepare_betas(betas_abs,selected_regressors,num_subjs); % absolute
coeffs_subjs = [coeffs_signed,coeffs_abs];
mean_avg = [mean_signed,mean_abs];
mean_sd = [sem_signed,sem_abs];

% GET STARS FOR CORRESPONDING REGRESSOR'S P-VALUES
bar_labels = {'','','','',''};
pstars = bar_labels;

% OTHER FIGURE PROPERTIES
xticks = [1:length(selected_regressors)]; % x-axis ticks
row1 = {'Fixed LR' 'Belief-state' 'Confirmation' 'Salience' 'Congruence'};
row2 = {'' '-adapted LR' 'bias' '' ''};
labelArray = [row1; row2;]; 
xticklabs = {'','','','',''}; % x-axis tick labels
title_name = {''}; % title
xlabelname = {'Regressor'}; % x-axis label
ylabelname = {'Mean beta coefficients'}; % y-axis label
colors_name = [fits_colors;darkblue_muted]; % colors for bars
y_label = repelem(1,1,length(selected_regressors)); % location for p-value stars
disp_pval = 0; % if p-value should be displayed

% PLOT
hold on
bar_plots_pval(coeffs_subjs,mean_avg,mean_sd,num_subjs, ...
    length(selected_regressors),2,{'Signed','Absolute'}, ...
    xticks,xticklabs,title_name,xlabelname, ...
    ylabelname,disp_pval,1,5,1,font_size,line_width,font_name,1,colors_name,pstars, ...
    y_label,[NaN,NaN,NaN,NaN,NaN],[0.5,5.5]) 

% PLOT PROPERTIES
set(gca,'Color','none')
ax = gca;
ax.XTick = [1 2 3 4];
ax.XTickLabel = '';
myLabels = labelArray;
for i = 1:length(myLabels)
    text(i, ax.YLim(1), sprintf('%s\n%s\n%s', myLabels{:,i}), ...
        'horizontalalignment', 'center', 'verticalalignment', 'top','FontSize',font_size);    
end
ax.XLabel.String = sprintf('\n\n%s', 'Regressor');
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
xticklabels(ax3,{'Fixed LR','Belief state','Salience','Congruence','Confirmation'}) % x-axis tick labels
yticklabels(ax3,{'Fixed LR','Belief state','Salience','Congruence','Confirmation'}) % y-axis tick labels
title(ax3,'Signed LR analysis','FontWeight','normal') % title
adjust_figprops(ax3,font_name,font_size,line_width,xlim_vals,ylim_vals) % adjust figure properties
box(ax3,'off')

imagesc(ax4,correlation_abs)
colormap(ax4,flipud(bone));
adjust_figprops(ax4,font_name,font_size,line_width,xlim_vals,ylim_vals)
xticklabels(ax4,{'Fixed LR','Belief-state','Salience','Congruence','Confirmation'})
title(ax4,'Absolute LR analysis','FontWeight','normal')
set(ax4,'YTickLabel', [])
box(ax4,'off')
%% ADD SUBPLOT LABELS

ax1_pos = ax1.Position;
adjust_x = - 0.08; % adjusted x-position for subplot label
adjust_y = ax1_pos(4); % adjusted y-position for subplot label

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
print(fig, 'figure2_corr_SM3.png', '-dpng', '-r600') 