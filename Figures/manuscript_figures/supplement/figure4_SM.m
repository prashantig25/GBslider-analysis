clc
clearvars

line_width = 0.5; % line width for axes
font_size = 6; % font size
font_name = 'Arial'; % font name
linewidth_axes = 0.5; % line width for plot lines
fontsize_label = 12; % fontsize for subplot labels
dot_size = 10;
[~,~,~,~,~,~,~,~,~,~,~,~,~,barface_green,...
    ~,~,~,fits_colors,~] = colors_rgb(); % colors

% INITIALISE VARS
load("Data/LR analyses/rsquared_wo_rewunc_obj.mat","rsquared_full"); % r-squared values
load("Data/LR analyses/posterior_up_wo_rewunc_obj.mat","posterior_up_subjs"); % posterior updates
data_subjs = readtable("Data/LR analyses/preprocessed_subj.xlsx"); % single-trial updates, prediction errors
num_subjs = 98; % number of subjects
%% INITIALISE TILE LAYOUT

figure
set(gcf,'Position',[100 100 400 200])
t = tiledlayout(1,2);
t.TileSpacing = 'compact';
t.Padding = 'compact';
ax1 = nexttile(1,[1,1]);

%% PLOT R-SQUARED VALUES

% CHANGE TILE POSITION
ax2 = nexttile(2,[1,1]);
position_change = [0.17, 0, -0.17, 0]; % change in position
new_pos = change_position(ax2,position_change); % new position
ax2_new = axes('Units', 'Normalized', 'Position', new_pos); % updated position
box(ax2_new, 'off'); % box off
delete(ax2); % delete old axis

% PLOT
title_name = {''}; 
bar_plots_pval(rsquared_full,mean(rsquared_full),std(rsquared_full)./sqrt(num_subjs),num_subjs, ...
    1,1,{''},1,{''},title_name,{''},{''},0,1,dot_size,1,font_size,line_width,font_name,0,barface_green) 
hold on

% PLOT PROPERTIES
xlabel('')
set(gca,'Color','none')
ylabel('Mean \itR^2\rm values','Interpreter','tex')
title('Model fit','FontWeight','normal')
%% PLOT POSTERIOR DISTRIBUTION

% GET EMPIRICAL AND POSTERIOR UPDATES
y = data_subjs.up(data_subjs.pe ~= 0); % empirical updates
y_hat = posterior_up_subjs; % regression model estimated updates
nbins = 75; % number of bins in a distribution

% CHANGE POSITION
position_change = [0.03, 0, 0.15, 0];
new_pos = change_position(ax1,position_change);
ax1_new = axes('Units', 'Normalized', 'Position', new_pos);
box(ax1_new, 'off');
delete(ax1);

% PLOT
h1 = histfit(y_hat,nbins);
hold on
h = histfit(y,nbins);

h1(1).FaceAlpha = 1; % face alpha for distributions
h(1).FaceAlpha = 0.7;

h(2).Color = fits_colors; % colors for distributions
h1(2).Color = [37, 50, 55]/255;

h(1).EdgeColor = fits_colors; % edge color for bars
h1(1).EdgeColor = [37, 50, 55]/255;

h(1).FaceColor = fits_colors; % face color for bars
h1(1).FaceColor = [37, 50, 55]/255;

% PLOT PROPERTIES
set(ax1_new,'Color','none','FontName',font_name,'FontSize',font_size)
set(ax1_new,'LineWidth',linewidth_axes)
l = legend('Regression fits','','Empirical updates','','EdgeColor','none','Color','none');
l.ItemTokenSize = [7 7];
xlabel('Update')
ylabel('Frequency (x 10^3)','Interpreter','tex')
yticklabels({'0','1','2','3','4','5'})
title('Posterior and empirical distribution','FontWeight','normal')
set(gca,'Color','none')
box off
%% ADD SUBPLOT LABELS

ax1_pos = ax2_new.Position;
adjust_x = [- 0.06,-0.075]; % adjusted x-position for subplot label
adjust_y = ax1_pos(4)+0.05; % adjusted y-position for subplot label

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
print(fig, 'figure4_SM2.png', '-dpng', '-r600') 