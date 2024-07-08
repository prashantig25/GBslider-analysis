%% INITIALISE VARIABLES

clc
clearvars
linewidth_arrow = 0.5; % arrow width
headlength_arrow = 5; % arrow headlength
fontsize_trial = 7.5; % font size
font_name = 'Arial'; % font name
horz_align = 'center'; % alignment
vert_align = 'middle';
bg_color = 'none'; % background color for text boxes
face_alpha = 0; % face alpha
edge_color = 'none'; % edge color for text boxes
trialtext_width = 0.1; % width for trial text
trialtext_height = 0.1;
screen_width = 3; % screen dimensions for trial
screen_height = 2;
linewidth_screens = 0.7; % line width for screens
fontsize_title = 9; % font size for plot titles
linewidth_axes = 0.5; % line width for axes
linewidth_box = 0.25; % line width for boxes
font_size = 6; % font size
fontsize_label = 12; % font size for subplot labels
line_style = '-'; % line style
[~,~,~,~,color_screen,fb_green,darkblue_muted,mix,perc,rew,~,~,~,~,...
    ~,~,~,~,~] = colors_rgb(); % colors

% load all required data
load("Data/descriptive data/main study/ecoperf_mix.mat","ecoperf_mix") % economic performance
load("Data/descriptive data/main study/ecoperf_perc.mat","ecoperf_perc")
load("Data/descriptive data/main study/ecoperf_rew.mat","ecoperf_rew")
load("Data/descriptive data/main study/mean_curves.mat","mean_curves") % learning across trials for each condition
load("Data/descriptive data/main study/sem_curves.mat","sem_curves")
load("Data/descriptive data/main study/mix_curves_study2.mat","mix_curve"); % learning in a block for each condition
load("Data/descriptive data/main study/perc_curves_study2.mat","perc_curve");
load("Data/descriptive data/main study/rew_curves_study2.mat","rew_curve");
load("Data/descriptive data/main study/ecoperf.mat","ecoperf"); % mean economic performance
load("Data/descriptive data/main study/esterror.mat","esterror"); % mean estimation error
%% INITIALISE TILE LAYOUT

figure
set(gcf,'Position',[100 100 600 400])
t = tiledlayout(3,4);
t.Padding = 'compact';
ax3 = nexttile(3,[2 2]);
ax5 = nexttile(9,[1 1]);
ax10 = nexttile(10,[1 1]);
ax12 = nexttile(12,[1,1]);
ax11 = nexttile(11,[1,1]);
ax1 = nexttile(1,[2,2]);
%% PLOT TRIAL PROCEDURE

% axes position adjustments
position_change = [0, 0, -0.05, 0]; % change in position
new_pos = change_position(ax1,position_change); 
ax1_new = axes('Units', 'Normalized', 'Position', new_pos); % new position
box(ax1_new, 'on'); % box on
delete(ax1); % delete old axis

start_x = 0.7; % start-x for screens
start_y = 8;
adjust_x = 2.3; % adjustments for screens
adjust_y = -1.8;
num_screens = 4; % number of screens
screens_y = [];

% PLOT SCREENS
for n = 1:num_screens
    rectangle('Position',[start_x start_y screen_width screen_height],'LineWidth',linewidth_screens, ...
        'FaceColor',color_screen)
    screens_y = [screens_y,start_y];
    start_x = start_x + adjust_x;
    start_y = start_y + adjust_y;
    axis([0 11 0 11]);
end

% PLOT SLIDER
line([7.8,10.5],[3,3],'color','k','LineWidth',2)
center_X = 9.5;  % X-coordinate of the center
center_Y = 3.0;  % Y-coordinate of the center
radius = 0.17;   % Radius of the circle
rectangle('Position', [center_X - radius, center_Y - radius, 2 * radius, 2 * radius], ...
    'Curvature', [1, 1], 'EdgeColor', 'k', 'FaceColor', 'w', 'LineWidth', 1);

% PLOT FIXATION CROSS
fix_width = 0.08;
fix_height = 0.2;
fix_xpos = [2,4.3,8.9];
fix_ypos = [9.2,7.45,3.8];
num_fix = 3;
for n = 1:num_fix
    fix1 = annotation("textbox",'String','+','FontSize',10,'LineStyle','none');
    fix1.Parent = ax1_new;
    fix1.Position = [fix_xpos(n) fix_ypos(n) fix_width fix_height];
end

% PLOT ARROW 
ar1 = annotation('arrow','LineWidth',linewidth_arrow,'HeadLength',headlength_arrow);
ar1.Parent = ax1_new;
ar1.X = [1 6];
ar1.Y = [5,1];

% ADD TEXTBOXES
all_strings = {{'Fixation (0.5s)', ''},{'Choice (1s)' ''},{'Reward', 'feedback (1s)'},{'Reported reward', 'probability (7s)'}};
num_strings = 4;
text_xpos = [0.7,3,5.3,7.6];
text_ypos = [7.8,6,4.2,2.3];
horzalign_trial = 'Left';
vertalign_trial = 'Top';
for n = 1:num_strings
    string = all_strings{1,n};
    position = [text_xpos(n) text_ypos(n) trialtext_width trialtext_height];
    annotate_textbox(ax1_new,position,string,font_name,5, ...
        horzalign_trial,vertalign_trial,bg_color,face_alpha,'none');
end

% ADD FEEDBACK TEXT
fb1 = annotation("textbox",'LineWidth',1.5,'String', ...
    'You win 1 point!','FontSize',fontsize_trial,'LineStyle','none', ...
    'Color',fb_green,'FontName','Arial','FontWeight','bold', ...
    'HorizontalAlignment','center');
fb1.Parent = ax1_new;
fb1.Position = [6.8 6.2 .07 .1];
set(gca, 'Color', 'None')
box off
axis off

% CREATE CONDITIONS TABLE AS INSET
line([5.3 10.8],[9.7 9.7],'LineWidth',0.5,'Color','k')
line([7.04 7.04],[10.5 8.3],'LineWidth',0.5,'Color','k')
line([8.95 8.95],[10.5 8.3],'LineWidth',0.5,'Color','k')

% ADD TEXT BOXES
all_strings = {'Both','Perceptual','Reward','0.7     0.3','0.9     0.1',...
    '0.7     0.3'}; % string for each text box
num_strings = 6; % number of strings
text_xpos = [6, 7.9, 9.8, 6, 7.9, 9.8]; % x-position
text_ypos = [10, 10, 10, 8.5, 8.5, 8.5]; % y-posisiton
box_width = 0.2; % text box width
box_height = 0.15; % text box height
for n = 1:num_strings
    string = all_strings(n);
    position = [text_xpos(n) text_ypos(n) box_width box_height];
    annotate_textbox(ax1_new,position,string,font_name,font_size-0.5, ...
        horz_align,vert_align,bg_color,face_alpha,edge_color);
end
%% PLOT S-A-R CONTINGENCY

pos = ax3.Position + [0, -0.03,0,0];
ax3_new = axes('Units', 'Normalized', 'Position', pos);
box(ax3_new, 'on');
delete(ax3);

axis([0 1 0 1])
title('Task contingency','FontWeight','normal',FontName=font_name,Position=[0.5,0.95], ...
    Parent=ax3_new,FontSize=fontsize_title)
line([0 1], [0.89 0.89],'Color','k','LineWidth',linewidth_axes);

% ADD TEXTBOXES
all_strings = {'State 0','State 1'};
num_strings = 2;
text_xpos = [0.23, 0.68] - 0.01;
text_ypos = [0.86, 0.86];
statebox_width = 0.1;
for n = 1:num_strings
    string = all_strings(n);
    position = [text_xpos(n) text_ypos(n) statebox_width statebox_width];
    annotate_textbox(ax3_new,position,string,font_name,font_size, ...
        horz_align,vert_align,bg_color,face_alpha,edge_color);
end

all_strings = {'Left stronger','Right stronger'};
text_xpos = text_xpos - 0.07;
text_ypos = text_ypos - 0.06;
box_width = 0.25; 
box_height = 0.08;
for n = 1:num_strings
    string = all_strings(n);
    position = [text_xpos(n) text_ypos(n) box_width box_height];
    annotate_textbox(ax3_new,position,string,font_name,font_size, ...
        horz_align,vert_align,bg_color,face_alpha,edge_color);
end

all_strings = {'Left   Right','Left    Right','μ = 0.9','μ = 0.9','0.9      0.1','0.1      0.9'};
num_strings = 6;
text_xpos = [0.145, 0.6, 0.145, 0.6, 0.145, 0.6, 0.145, 0.6];
text_ypos = [0.5, 0.5, 0.335, 0.335, 0.27, 0.27, 0.73, 0.73];
edge_colors = {'k','k','none','none','k','k'};
bg_colors = {bg_color, bg_color, [238, 238, 238]/256, [238, 238, 238]/256, bg_color, bg_color};
for n = 1:num_strings
    string = all_strings(n);
    position = [text_xpos(n) text_ypos(n) box_width box_height];
    annotate_textbox(ax3_new,position,string,font_name,font_size, ...
        horz_align,vert_align,bg_colors{n},face_alpha,edge_colors{n},linewidth_box,line_style);
end

% ADD ROTATED TEXT BOXES
xpos = 0.01;
ypos = [0.55,0.3,0.8];
allstrings = {{'Economic', 'choice'},{'Contingency' 'parameter'},{'', 'Stimulus'}};
num_strings = 3;
for n = 1:num_strings
    txt = text(xpos,ypos(n),allstrings{1,n});
    txt.FontSize = font_size;
    txt.FontWeight = 'normal';
    txt.Rotation = 90;
    txt.LineStyle = 'none';
    txt.FontName = font_name;
    txt.HorizontalAlignment = horz_align;
end

annotation("textbox",[0.145,0.73,0.25,0.08],'LineWidth',linewidth_box,'String', ...
    '','FontSize',font_size,'LineStyle','-','Color','k','FontName','Arial', ...
    'HorizontalAlignment','center',Parent=gca)

annotation("textbox",[0.6,0.73,0.25,0.08],'LineWidth',linewidth_box,'String', ...
    '','FontSize',font_size,'LineStyle','-','Color','k','FontName','Arial', ...
    'HorizontalAlignment','center',Parent=gca)

a1 = annotation('arrow',[0.5 0.5],[0.6 0.4],'LineWidth',0.7,'Color', ...
    'k','LineStyle','-','HeadLength',headlength_arrow);
a1.Parent = gca;
set(gca, 'Color', 'None','FontName','Arial')
box off
axis off
%% DESCRIPTIVE PLOTS

position_change = [0, 0.05, 0, 0];
new_pos = change_position(ax12,position_change);
ax12_new = axes('Units', 'Normalized', 'Position', new_pos);
box(ax12_new, 'on');
delete(ax12);

% PLOT EST. ERROR vs. ECOPERF
scatter(esterror,ecoperf,"filled","o",'MarkerFaceColor',[220, 220, 220]./255, ...
        'MarkerEdgeColor',[184, 184, 184]./255,'LineWidth',0.7,'MarkerFaceAlpha',1,SizeData=10)
l = lsline;
l.Color = 'k';
l.LineWidth = linewidth_axes;
ylabel('Mean economic performance')
xlabel("Mean estimation error")
title(strcat("\itr\rm = ",{''},num2str(round(corr(esterror,ecoperf),2))) + newline + "\itp\rm < 0.001",'FontWeight','normal')

% PLOT CONFIDENCE INTERVAL LINES
mdl = fitlm(esterror,ecoperf); % Fit a linear model
xrange = xlim;
xvalues = linspace(xrange(1), xrange(2), 100); % predictor values
[yfit, yci] = predict(mdl, xvalues'); % Compute and plot confidence interval lines
hold on
plot(xvalues, yfit,LineWidth=1.5, Color=darkblue_muted); % Plot the fitted line
plot(xvalues, yci, Color=darkblue_muted, LineWidth=0.2,LineStyle='--'); % Plot the upper confidence interval
plot(xvalues, yci(:,1), Color=darkblue_muted, LineWidth=0.2,LineStyle='--'); % Plot the lower confidence interval
set(gca,'color','none','FontName',font_name,'FontSize',font_size,'LineWidth', ...
    linewidth_axes,'YLim',[0.3,1],'YTick',[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], ...
    'YTickLabels',{'0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'})
 
position_change = [0, 0.05, -0.01, 0];
new_pos = change_position(ax11,position_change);
ax11_new = axes('Units', 'Normalized', 'Position', new_pos);
box(ax11_new, 'on');
delete(ax11);

% PLOT SLIDER DATA
colors_name = [mix;perc;rew]; % colors for plot lines
legend_names = {'Both','Perceptual','Reward'}; % legend names
title_name = {'Learning curve'}; % figure title
xlabelname = {'Trial'}; % x-axis label name
ylabelname = {'Slider response'}; % y-axis label name
x = 1:25; % x-axis
hold on
lg_curves(x,mean_curves./100,sem_curves./100,colors_name,legend_names,title_name,xlabelname,ylabelname)
xlim([1,25])
set(gca,'color','none','FontName',font_name,'FontSize',font_size,'LineWidth',linewidth_axes)
yline(0.9,'--',"Color",'k',LineWidth=0.5)
yline(0.7,'--',"Color",'k',LineWidth=0.5)
ylim([0.4,1])
annotation("textbox",[1,0.96,0.2,0.04],'LineWidth',linewidth_box,'String', ...
    ' Actual reward probability','FontSize',font_size,'LineStyle','none','Color','k','FontName','Arial', ...
    'HorizontalAlignment','left',Parent=gca)

new_pos = change_position(ax5,position_change);
ax5_new = axes('Units', 'Normalized', 'Position', new_pos);
box(ax5_new, 'on');
delete(ax5);

% MEAN ECONOMIC PERFORMANCE ACROSS SUBJECTS
y = [ecoperf_mix;ecoperf_perc;ecoperf_rew;];
num_subjs = length(ecoperf_mix);
mix_avg = nanmean(ecoperf_mix,1);
perc_avg = nanmean(ecoperf_perc,1);
rew_avg = nanmean(ecoperf_rew,1);
mean_avg = [mix_avg; perc_avg;rew_avg;];

% SEM ACROSS SUBJECTS
mix_sd = nanstd(ecoperf_mix,1)/sqrt(num_subjs);
perc_sd = nanstd(ecoperf_perc,1)/sqrt(num_subjs);
rew_sd = nanstd(ecoperf_rew,1)/sqrt(num_subjs);
mean_sd = [mix_sd; perc_sd;rew_sd;];
xticks = [1:length(mean_sd)];

% FIGURE PROPERTIES
xticklabs = {'Both','Perceptual','Reward'};% x-axis tick labels
title_name = {'Economic performance'}; % figure title
legend_names = {''}; % legend names
xlabelname = {''}; % x-axis label name
ylabelname = {'P(Correct)'};%{'Mean economic'; 'performance'}; % y-axis label name
colors_name = [92, 110, 129]./255; % bar colors

y = mean(y,2);
mean_avg = mean(mean_avg,2);
mean_sd = mean(mean_sd,2);
% PLOT CHOICE DATA
bar_plots(y,mean_avg,mean_sd,num_subjs,length(mean_avg),length(legend_names), ...
    legend_names,xticks,xticklabs,title_name,xlabelname,ylabelname,colors_name) 
set(gca,'color','none','FontName',font_name,'FontSize',font_size,'YLim',[0.3,1], ...
    'LineWidth',linewidth_axes,'YTick',[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], ...
    'YTickLabels',{'0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'})

new_pos = change_position(ax10,position_change);
ax10_new = axes('Units', 'Normalized', 'Position', new_pos);
box(ax10_new, 'on');
delete(ax10);

% SLIDER UPDATES ACROSS SUBJECTS
mix_avg = nanmean(mix_curve,2); % mean across trials
perc_avg = nanmean(perc_curve,2);
rew_avg = nanmean(rew_curve,2);
y = [mix_avg;perc_avg;rew_avg;];
mix_subjs = nanmean(mix_avg,1); % mean across participants
perc_subjs = nanmean(perc_avg,1);
rew_subjs = nanmean(rew_avg,1);
mu_subjs = [mix_subjs, perc_subjs, rew_subjs];

% SEM ACROSS SUBJECTS
mix_sd = nanstd(mix_avg,1)/sqrt(num_subjs);
perc_sd = nanstd(perc_avg,1)/sqrt(num_subjs);
rew_sd = nanstd(rew_avg,1)/sqrt(num_subjs);
mu_sd = [mix_sd; perc_sd;rew_sd;];

% FIGURE PROPERTIES
title_name = {'Learning behavior'}; % figure title
ylabelname = {'Mean slider response'};

% PLOT CHOICE DATA
bar_plots(y./100,mu_subjs.'./100,mu_sd./100,num_subjs,length(mu_subjs),length(legend_names), ...
    legend_names,xticks,xticklabs,title_name,xlabelname,ylabelname,colors_name) 
set(gca,'color','none','FontName',font_name,'FontSize',font_size,'YLim',[0.3,1], ...
    'LineWidth',linewidth_axes,'YTick',[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], ...
    'YTickLabels',{'0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'})
%% ADD EXTERNAL PNGs

patch_dim = 0.035; % dimensions for patch
pos_y = 0.7825; % position of patches on y-axis
pos_x = [0.655,0.605,0.845,0.795]; % position of patches on x-axis
image_png = {'lowcon_patch.png','highcon_patch.png','highcon_patch.png','lowcon_patch.png'};
num_pngs = 4;
for n = 1:num_pngs
    axes('pos',[pos_x(n) pos_y patch_dim patch_dim]);
    imshow(image_png{n});
    hold on
end

patch_dim = 0.03;
adjust_x = [0.032, 0.03,0.032,0.03,0.032,0];
pos_y = 0.85; pos_x = 0.237;
image_png = {'lowcon_patch.png','lowcon_patch.png','lowcon_patch.png','lowcon_patch.png','highcon_patch.png','lowcon_patch.png',};
num_pngs = 6;
for n = 1:num_pngs
    axes('pos',[pos_x pos_y patch_dim patch_dim]);
    imshow(image_png{n});
    hold on
    pos_x = pos_x + adjust_x(n);
end

patch_dim = 0.04;
pos_y = 0.744; pos_x = 0.17;
image_png = {'lowcon_patch.png','highcon_patch.png'};
num_pngs = 2;
adjust_x = 0.05;
for n = 1:num_pngs
    axes('pos',[pos_x pos_y patch_dim patch_dim]);
    imshow('lowcon_patch.png');
    hold on
    pos_x = pos_x + adjust_x;
end

%% SUBPLOT LABELS

ax1_pos = ax1_new.Position;
adjust_x = -0.06; % adjust x-position of subplot label
adjust_y = ax1_pos(4) - 0.05; % adjust y-position of subplot label
[label_x,label_y] = change_plotlabel(ax1_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'a','FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment',horz_align)
adjust_y = ax1_pos(4) - 0.02;
[label_x,label_y] = change_plotlabel(ax3_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'b','FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment',horz_align)

ax5_pos = ax5_new.Position;
adjust_y = ax5_pos(4) + 0.018;
adjust_x = -0.06;
[label_x,label_y] = change_plotlabel(ax5_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'c','FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment',horz_align)
[label_x,label_y] = change_plotlabel(ax10_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'd','FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment',horz_align)
[label_x,label_y] = change_plotlabel(ax11_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'e','FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment',horz_align)
[label_x,label_y] = change_plotlabel(ax12_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'f','FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment',horz_align)
%% SAVE AS PNG

fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, 'task.png', '-dpng', '-r600') 