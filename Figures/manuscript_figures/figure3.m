%% INITIALISE VARS

clc
clearvars

% general vars for plot
colors_manuscript; % colors for plot
font_name = 'Arial'; % font name
font_size = 6; % font size
horz_align = 'center'; % horizontal alignment for text
vert_align = 'middle'; % vertical alignment for text
line_width = 0.5; % linewidth for axes
linewidth_box = 1; % linewidth for box edges
linewidth_arrow = 0.5; % linewidth for arrows
linewidth_line = 1.5; % linewidth for plotted lines
headlength_arrow = 5; % headlength for arrows
tile_title = "Normative agent";
bs_string = "BS modulated computations";
bg_color = 'none'; % background color for textbox
face_alpha = 0; % face alpha for textbox
edge_color = 'none';

% agent related vars
contrast_diff = [0.08,0.02]; % contrast difference levels
sigma = [0.06,0.06]; % perceptual sensitivity of agent
colors_pu = [low_PU; high_PU]; % colors for low and high perceptual uncertainty data
ag = Agent(); % initialise Agent to generate belief states
pi_0_high = []; pi_1_high = []; % empty arrays to store belief states
mu = 0.8; % mu for EV computation
pe_vals = linspace(-1,1,10); % prediction error range
neg_up = [-0.5,-0.15]; % negative updates
pos_up = [0.5,0.15]; % positive updates
trial = 1:25; % trial numbers

%% INITIALISE TILE LAYOUT

f1 = figure;
set(gcf,'Position',[100 100 300 500])
t = tiledlayout(4,3);
ax1 = nexttile(2);
ax2 = nexttile(5);
ax3 = nexttile(7);
ax4 = nexttile(9);
ax5 = nexttile(10,[1,3]);

%% PLOT OBSERVATION PROBABILITIES

% ADJUST AXES POSITION
position_change = [-0.05, 0.05, 0.1, -0.03];
new_pos = change_position(ax1,position_change);
ax1_new = axes('Units', 'Normalized', 'Position', new_pos); 
box(ax1_new, 'on');
delete(ax1);

% PLOT
x = linspace(-0.3,0.3,40); % contrast difference range
xline(0,'LineStyle','-','LineWidth',1) % contrast difference = 0
hold on
for i = 1:length(contrast_diff)
    xline(contrast_diff(i),'LineStyle','-','LineWidth',linewidth_line,'Color',colors_pu(i,:))
    ag.p_s_giv_o(contrast_diff(i)); % generate belief states
    fit = normpdf(x, contrast_diff(i), sigma(i)); % generates pdf of the normal distribution 
    % with mean contrast difference and standard deviation of sigma
    p1 = plot(x,fit,'LineWidth',linewidth_line, 'Color',colors_pu(i,:)); 
    hold on
    pi_0_high = [pi_0_high ag.pi_0]; % store belief states
    pi_1_high = [pi_1_high ag.pi_1];
end

% ADJUST FIGURE PROPERTIES
xlim = [-0.3 0.3];
ylim = [0 17];
hold on
adjust_figprops(ax1_new,font_name,font_size,line_width,xlim,ylim);
title_obj = title('Observation probability','FontWeight','normal','Units','normalized');
xlabel({'Contrast difference'})
l = legend('','Low PU','','High PU','Location','best','Color','none', ...
    'EdgeColor','none','AutoUpdate','off','FontSize',5);
l.ItemTokenSize = [5, 5];
set(ax1_new,'YColor','none');

% ADD TEXT BOXES
position = [.1 5 .2 .2]; % position for textbox
string = 'State 1'; % string for textbox
annotate_textbox(ax1_new,position,string,font_name,font_size, ...
    'Left',vert_align,bg_color,face_alpha,edge_color)

position = [-0.35 5 .2 .2];
string = 'State 0';
annotate_textbox(ax1_new,position,string,font_name,font_size, ...
    'Left',vert_align,bg_color,face_alpha,edge_color)

position = [0.33 13 .19 .09];
string = 'Presented contrast difference';
bg_color = [210, 210, 210]./255;
annotate_textbox(ax1_new,position,string,font_name,font_size, ...
    horz_align,'Top',bg_color,face_alpha,edge_color,linewidth_box)

position = [-0.25 22 0.5 0];
string = tile_title;
annotate_textbox(ax1_new,position,string,font_name,10, ...
    horz_align,vert_align,bg_color,face_alpha,edge_color,linewidth_box)

% ADD ARROWS
ar = annotation('arrow','LineWidth',linewidth_arrow,'HeadLength',headlength_arrow,'Color','k');
ar.X = [0.67 0.58];
ar.Y = [0.885,0.885];

ar1 = annotation('arrow','LineWidth',linewidth_arrow,'HeadLength',headlength_arrow,'Color','k');
ar1.X = [0.52 0.52];
ar1.Y = [0.78,0.745];

ar2 = annotation('arrow','LineWidth',linewidth_arrow,'HeadLength',headlength_arrow,'Color','k');
ar2.X = [0.51 0.31];
ar2.Y = [0.55,0.45];

ar3 = annotation('arrow','LineWidth',linewidth_arrow,'HeadLength',headlength_arrow,'Color','k');
ar3.X = [0.53 0.74];
ar3.Y = [0.55,0.45];

ar = annotation('arrow','LineWidth',linewidth_arrow,'HeadLength',headlength_arrow,'Color','k');
ar.X = [0.38 0.63];
ar.Y = [0.35,0.35];
box off
hold on
%% PLOT BELIEF STATES

new_pos = change_position(ax2,position_change);
ax2_new = axes('Units', 'Normalized', 'Position', new_pos); 
box(ax2_new, 'on');
delete(ax2);

% PLOT BAR
hold on
b = bar([pi_0_high(1) pi_1_high(1);pi_0_high(2) pi_1_high(2)], ...
    'BarLayout','grouped','FaceColor','flat');
b(1).CData(1,:) = colors_pu(1,:);
b(2).CData(1,:) = colors_pu(1,:);
b(1).CData(2,:) = colors_pu(2,:);
b(2).CData(2,:) = colors_pu(2,:);
hatchfill2(b(1),'single','HatchAngle',45,'HatchDensity',25);

% ADJUST FIGURE PROPERTIES
ylim = [0,1.3];
xlim = [0.5,2.5];
adjust_figprops(ax2_new,font_name,font_size,line_width,xlim,ylim)
title({'Belief state (BS)'},'FontWeight','normal')
xlabel({'Perceptual uncertainty'})
set(ax2_new,'XTicklabels',["Low","","High"],'YColor','none')
box off
hold on

% ADD TEXT BOXES
position = [1.5 1.2 .2 .2];
string = 'State 0';
bg_color = 'none';
face_alpha = 0;
annotate_textbox(ax2_new,position,string,font_name,font_size, ...
    'Left','Top',bg_color,face_alpha,edge_color)
position = position - [0 0.15 0 0];
string = 'State 1';
annotate_textbox(ax2_new,position,string,font_name,font_size, ...
    'Left','Top',bg_color,face_alpha,edge_color)

% ADD PATCHES TO BARS
x = [1.35 1.5 1.5 1.35];
y = [1.3 1.3 1.2 1.2];
p1 = patch(x,y,colors_pu(2,:));

y = [1.15 1.15 1.05 1.05];
p2 = patch(x,y,colors_pu(2,:));

x = [1.1 1.25 1.25 1.1];
y = [1.3 1.3 1.2 1.2];
p1_dark = patch(x,y,colors_pu(1,:));

y = [1.15 1.15 1.05 1.05];
p2_dark = patch(x,y,colors_pu(1,:));

hatchfill2(p1, 'single', 'HatchAngle', 45, 'HatchDensity', 25, 'HatchColor', 'black');
hatchfill2(p1_dark, 'single', 'HatchAngle', 45, 'HatchDensity', 25, 'HatchColor', 'black');

% ADD TEXT TO BARS
groupOffset = [-0.2, 0.2];
barWidth = 0.6;
bar_text(b,groupOffset,barWidth,5);

%% PLOT EXPECTED VALUES
position_change = [-0.05, -0.06, 0.1, 0];
new_pos = change_position(ax3,position_change);
ax3_new = axes('Units', 'Normalized', 'Position', new_pos);
box(ax3_new, 'on');
delete(ax3);

% compute expected values
v_a_0 = (pi_0_high - pi_1_high) * mu + pi_1_high; 
v_a_1 = (pi_1_high - pi_0_high) * mu + pi_0_high;         
v_a_t = [v_a_0; v_a_1]; % concatenate action valences

% PLOT BARS
hold on
b = bar([v_a_t(1,1),v_a_t(2,1); v_a_t(1,2),v_a_t(2,2)],'BarLayout','grouped','FaceColor','flat');
b(1).CData(1,:) = colors_pu(1,:);
b(2).CData(1,:) = colors_pu(1,:);
b(1).CData(2,:) = colors_pu(2,:);
b(2).CData(2,:) = colors_pu(2,:);
hatchfill2(b(1),'single','HatchAngle',45,'HatchDensity',25);

% ADJUST FIGURE PROPERTIES
ylim = [0,1.1];
xlim;
adjust_figprops(ax3_new,font_name,font_size,line_width,xlim,ylim)
title({'Expected value (EV)'},'FontWeight','normal')
xlabel({'Perceptual uncertainty'})
set(ax3_new,'XTicklabels',["Low","","High"],'YColor','K')
box off
hold on

% ADD TEXT BOXES
position = [1.5 0.96 .2 .2];
string = 'Action 0';
bg_color = 'none';
face_alpha = 0;
annotate_textbox(ax3_new,position,string,font_name,font_size, ...
    'Left',vert_align,bg_color,face_alpha,edge_color);
position = position - [0 0.1 0 0];
string = 'Action 1';
annotate_textbox(ax3_new,position,string,font_name,font_size, ...
    'Left',vert_align,bg_color,face_alpha,edge_color);

% ADD PATCHES
x = [1.35 1.5 1.5 1.35];
y = [1.1 1.1 1.025 1.025];
p1 = patch(x,y,colors_pu(2,:));

y = [0.99 0.99 0.915 0.915];
p2 = patch(x,y,colors_pu(2,:));

x = [1.1 1.25 1.25 1.1];
y = [1.1 1.1 1.025 1.025];
p1_dark = patch(x,y,colors_pu(1,:));

y = [0.99 0.99 0.915 0.915];
p2_dark = patch(x,y,colors_pu(1,:));

hatchfill2(p1, 'single', 'HatchAngle', 45, 'HatchDensity', 25, 'HatchColor', 'black');
hatchfill2(p2_dark, 'single', 'HatchAngle', 45, 'HatchDensity', 25, 'HatchColor', 'black');

% ADD TEXT ON BARS
groupOffset = [-0.2, 0.2];
barWidth = 0.6;
bar_text(b,groupOffset,barWidth,font_size)

%% PLOT LEARNING RATES
position_change = [-0.05, -0.06, 0.1, 0]; 
new_pos = change_position(ax4,position_change);
ax4_new = axes('Units', 'Normalized', 'Position', new_pos);
box(ax4_new, 'on');
delete(ax4);

% PLOT
for i = 1:length(contrast_diff)
    hold on
    plot(pe_vals,linspace(neg_up(i),pos_up(i),10),'LineStyle','-','Color', ...
        colors_pu(i,:),'LineWidth',linewidth_line)
end
hold on
plot(pe_vals,repelem(0,1,10),'LineStyle',':','Color','k','LineWidth',0.7)
hold on
plot(pe_vals,pe_vals,'LineStyle',':','Color','k','LineWidth',0.7)

% ADJUST FIGURE PROPERTIES
xlim = [-1 1];
ylim = [-1 1];
adjust_figprops(ax4_new,font_name,font_size,line_width,xlim,ylim)
title('Update','FontWeight','normal')
xlabel({'Prediction error'})
l1 = legend('LR = 0.5','LR = 0.15','Location','best','Color','none', ...
    'EdgeColor','none','AutoUpdate','off');
l1.ItemTokenSize = [5, 5];
box off
%% PLOT LEARNING CURVES

% for normative agent
data_normative = readtable("data_model_ag1.xlsx");
data_normative = data_normative(data_normative.condition == 2,:);
data_normative = data_normative(26:50,:);

% for categorical learning
data_categorical = readtable("agent2_learning.xlsx");
data_categorical = data_categorical(1:25,:);

position_change = [-0.05, -0.06, 0.1, -0.02];
new_pos = change_position(ax5,position_change);
ax5_new = axes('Units', 'Normalized', 'Position', new_pos);
box(ax5_new, 'on');
delete(ax5);

% PLOT
hold on
s1 = scatter(trial,data_normative.reward,'filled','MarkerEdgeColor', ...
    dots_edges,'SizeData',15,'MarkerFaceColor',gray_dots);
hold on
s3 = plot(trial,data_normative.mu,"Color",'k','LineStyle','-','LineWidth',linewidth_line);
hold on
s4 = plot(trial,data_categorical.mu,"Color",light_gray,'LineStyle','-','LineWidth',linewidth_line);
l = legend([s1 s3 s4],'Reward','Reported \mu (normative)', ...
    'Reported \mu (categorical)','EdgeColor','none','Color','none','Location','best');
l.ItemTokenSize = [5 5];
title('Relationship between reported \mu and estimation uncertainty','FontWeight','normal')
xlabel('Trial')

% ADJUST FIGURE PROPERTIES
ylim = [0.4 1.1];
xlim = [1 25];
adjust_figprops(ax5_new,font_name,font_size,line_width,xlim,ylim);
box(ax5_new,'off')
%% import any external pngs

axes('pos',[.335 .92 .02 .02]);
imshow('lowcon_patch.png');
hold on

axes('pos',[.365 .92 .02 .02]);
imshow('highcon_patch.png');
hold on

axes('pos',[.335 .9 .02 .02]);
imshow('lowcon_patch.png');
hold on

axes('pos',[.365 .9 .02 .02]);
imshow('lowcon_patch.png');
hold on

all_strings = {'Reward','BS modulated computations'};
box_height = 0.025;
x_pos = [0.46, 0.32]; y_pos = [0.335, 0.48]; box_width = [0.09,0.4];
num_strings = 2;
for n = 1:num_strings
    text2 = annotation("textbox");
    text2.Position = [x_pos(n) y_pos(n) box_width(n) box_height];
    text2.String = all_strings{n};
    text2.FontName = font_name;
    text2.FontSize = 5;
    text2.HorizontalAlignment = horz_align;
    text2.VerticalAlignment = vert_align;
    text2.BackgroundColor = color_screen;
end
%% SUBPLOT LABELS

ax1_pos = ax1_new.Position;
adjust_x = -0.06;
adjust_y = ax1_pos(4) - 0.005;
[label_x,label_y] = change_plotlabel(ax1_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'a','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(ax2_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'b','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(ax3_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'c','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(ax4_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'd','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(ax5_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'e','FontSize',12,'LineStyle','none','HorizontalAlignment','center')
%% SAVE FIGURE

fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, 'agent_normative2.png', '-dpng', '-r600') 
