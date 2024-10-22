% Figure 1: Summary

% INITIALISE VARS

clc
clearvars

linewidth_axes = 0.5; % line width for axes
font_name = 'Arial'; % font
font_size = 6; % font size
horz_align = 'center'; % horizontal alignment for text
vert_align = 'middle'; % vertical alignment for text
headlength_arrow = 4; % head length for arrow
fontsize_label = 12; % font size for subplot labels
[pu_box,high_PU,mid_PU,low_PU,~,~,~,~,~,~,~,~,~,~,~,~,~,~,gray_arrow] = colors_rgb(); % colors

current_Dir = pwd;
save_dir = fullfile("saved_figures",filesep,"main");
mkdir(save_dir)

% INITIALISE TILE LAYOUT

figure("Position",[150,150,350,350])
t = tiledlayout(3,2);
t.TileSpacing = 'compact';
t.Padding = 'compact';
ax1 = nexttile(1);
ax2 = nexttile(2, [2,1]);
ax3 = nexttile(3);
ax5 = nexttile(5, [1,2]);

% H1: low perceptual uncertainty

% CHANGE AXES POSITION
position_change = [0, -0.05, 0, 0.015]; % adjusted position
new_pos = change_position(ax1,position_change);
ax1_new = axes('Units', 'Normalized', 'Position', new_pos); % new position
box(ax1_new, 'on'); % box on
delete(ax1); % delete old axis
axis([0 10 0 10]); % axis limits
set(ax1_new, 'YTick', [], 'XTick', [],'Color','none'); % remove tick labels
hold on

% ADD TITLEs
h1 = sgtitle('H1: Perceptual uncertainty drives speed of learning');
h1.FontName = font_name;
h1.FontSize = font_size;
h1.FontWeight = 'bold';

% ADD TEXT BOXES
all_strings = {'Distinct belief states (BS)','','Perception','Reward','State 0: pretzel','State 1: baguette'};
num_strings = 6; % number of strings
text_xpos = [0,3.9,0.9,0.6,1.3,6.7]; % x-position
text_ypos = [9,2,4,0.5,7.5,7.5]; % y-posiition
box_width = [10, 2, 2, 2,2,2]; % box width
box_height = [1, 1, 1, 1, 1 1]; % box height
bg_color = {pu_box,'none','none','none','none','none'}; % background color
edge_color = {'k','none','none','none','none','none'}; % edge color
face_alpha = [1,0,0,0,0,0];
fonts = [font_size,font_size,5,5,5,5];
for n = 1:num_strings
    string = all_strings(n);
    position = [text_xpos(n) text_ypos(n) box_width(n) box_height(n)];
    annotate_textbox(ax1_new,position,string,font_name,fonts(n), ...
        horz_align,vert_align,bg_color{n},face_alpha(n),edge_color{n},linewidth_axes,'-');
    hold on
end

a1 = annotation('arrow','HeadLength',headlength_arrow);
a1.Parent = gca;
a1.X = [5.4 6.87];
a1.Y = [2.5 5];
hold on

% H1: High perceptual uncertainty

% CHANGE AXES POSITION
position_change = [0, 0, 0, 0.015]; 
new_pos = change_position(ax3,position_change);
ax3_new = axes('Units', 'Normalized', 'Position', new_pos);
box(ax3_new, 'on');
delete(ax3);

% CREATE BOX AND ADD GRAY SHADING
axis([0 10 0 10])
set(ax3_new, 'YTick', [], 'XTick', [],'Color','none')

% ADD TEXT BOXES
all_strings = {'Similar belief states','','Perception','Reward','State 0: ciabatta','State 1: baguette'};
num_strings = 6;
text_xpos = [0,3.9,0.9,0.6,1.4,6.7]; % x-position
box_height = [1, 1, 1, 1,1,1];
face_alpha = [1,0,0,0,0,0];
for n = 1:num_strings
    string = all_strings(n);
    position = [text_xpos(n) text_ypos(n) box_width(n) box_height(n)];
    annotate_textbox(ax3_new,position,string,font_name,fonts(n), ...
        horz_align,vert_align,bg_color{n},face_alpha(n),edge_color{n},linewidth_axes,'-');
end

% ADD ARROW
a1 = annotation('arrow','HeadLength',headlength_arrow);
a1.Parent = gca;
a1.X = [5.4 6.87];
a1.Y = [2.5 5];
a1.Color = [0.5 0.5 0.5 0.85];

a1 = annotation('arrow','HeadLength',headlength_arrow);
a1.Parent = gca;
a1.X = [4.7 3.3];
a1.Y = [2.5 5];
a1.Color = [0.5 0.5 0.5 1];

% PLOT LEARNING RATES

% CHANGE AXES POSITION
position_change =  [0.02, 0.05, -0.02, -0.09]; 
new_pos = change_position(ax2,position_change);
ax2_new = axes('Units', 'Normalized', 'Position', new_pos);
box(ax2_new, 'on');
delete(ax2);

% INITIALISE VARS
axis([-1 1 0 1])
pe = linspace(-1,1,5); % prediction error array
up_highPU = pe * 0.1; % updates for high BS uncertainty
up_lowPU = pe * 0.4; % updates for low BS uncertainty
up_midPU = pe * 0.25; % updates for medium BS uncertainty
up = [up_highPU;up_midPU;up_lowPU;zeros(5,1).';pe;];
linestyles = {'-','-','-','--','--',}; % line-styles
linewidths = [1.5,1.5,1.5,0.5,0.5]; % line-width
linecolors = {high_PU,mid_PU,low_PU,'k','k',}; % colors

% PLOT
for i = 1:5
    plot(ax2_new,pe,up(i,:),"LineWidth",linewidths(1,i),"Color",linecolors{1,i}, ...
        'LineStyle',linestyles{1,i});
    hold on
end
box off
xlabel(ax2_new,'Prediction error')
ylabel(ax2_new,'Update')
adjust_figprops(ax2_new,font_name,font_size,linewidth_axes);
yticks([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
l = legend({'Low BS uncertainty','Medium BS uncertainty','High BS uncertainty'}, ...
    'FontSize',font_size,'Color','none','EdgeColor', ...
    'none','Location','best','FontName',font_name,'AutoUpdate','off');
l.ItemTokenSize = [7 7];

% ADD ROTATED TEXT BOXES
ypos = [0.7,-0.07]; % position on y-axis
xpos = 0.8; % position on x-axis
rotate = [55, 0]; % degree of rotation
allstrings = {'LR = 1','LR = 0'};
num_strings = 2;
for n = 1:num_strings
    txt = text(xpos,ypos(n),allstrings{1,n});
    txt.Parent = ax2_new;
    txt.FontSize = font_size - 1;
    txt.FontWeight = 'normal';
    txt.Rotation = rotate(n);
    txt.LineStyle = 'none';
    txt.FontName = font_name;
    txt.HorizontalAlignment = horz_align;
end

% H2 Box

% CHANGE AXES POSITION
position_change =  [0, -0.02, 0, 0]; 
new_pos = change_position(ax5,position_change);
ax5_new = axes('Units', 'Normalized', 'Position', new_pos);
box(ax5_new, 'on');
delete(ax5);

% CREATE BOX AND ADD GRAY SHADING
set(ax5_new,'xtick',[])
set(ax5_new,'ytick',[],'Color','none')
hold on
axis(ax5_new,[0 10 0 10])
hold on

% ADD TITLE
h2 = title(ax5_new,'H2: Visual salience drives choices under reward uncertainty');
h2.FontName = font_name;
h2.FontSize = font_size;
h2.FontWeight = 'bold';
h2_pos = h2.Position;
h2.Position = h2_pos + [0, 0.6, 0];

% ADD ARROWS
num_arrows = 2;
xpos = [1.7 3.5];
ypos = [5 8;5 2];
for n = 1:num_arrows
    a1 = annotation('arrow','HeadLength',headlength_arrow);
    a1.Parent = ax5_new;
    a1.Color = gray_arrow;
    a1.X = xpos;
    a1.Y = ypos(n,:);
end
a1 = annotation('arrow','HeadLength',5);
a1.Parent = ax5_new;
a1.X = [7 8.3];
a1.Y = [5 5];
a1.Color = 'k';
a1.LineWidth = 1;

% ADD LINES
linewidths = [1.5,0.5];
xpos = [6 7];
ypos = [7 5;3 5];
num_lines = 2;
for n = 1:num_lines
    a1 = annotation('line');
    a1.Parent = ax5_new;
    a1.X = xpos;
    a1.Y = ypos(n,:);
    a1.Color = 'k';
    a1.LineWidth = linewidths(n);
end

% ADD EXTERNAL PNGs

png_width = 0.06; % width of PNG
png_height = 0.06; % height of PNG
xpos = [0.08,0.14,0.84]; % x-position
ypos = 0.15; % y-position
num_pngs = 3; % number of pngs
image_pngs = {'bread_low.png','bread.png','bread.png'}; % path
for n = 1:num_pngs
    axes('pos',[xpos(n) ypos png_width png_height])
    [img, ~, tr] = imread(image_pngs{n}); % add image
    im = image('CData',img);
    im.AlphaData = tr;
    set(gca,'YDir','reverse'); % correct for reversed image
    set(gca,'color','none'); % make it transparent
    set(gca,'XColor', 'none','YColor','none')
end

png_width = 0.02;
png_height = 0.02;
xpos = [0.4825,0.515];
ypos = [0.235,0.305] - 0.02;
num_pngs = 2;
image_pngs = {'bread_low.png','bread.png'};
for n = 1:num_pngs
    axes('pos',[xpos(n) ypos(n) png_width png_height])
    [img, ~, tr] = imread(image_pngs{n});
    im = image('CData',img);
    im.AlphaData = tr;
    set(gca,'YDir','reverse');
    set(gca,'color','none')
    set(gca,'XColor', 'none','YColor','none')
end

axes('pos',[0.65 0.135 .08 .08])
set(gca,'XColor', 'none','YColor','none')
[img, ~, tr] = imread("integration.png");
im = image('CData',img);
im.AlphaData = tr;
set(gca,'YDir','reverse');
set(gca,'color','none')
colormap(gca,"bone"); 
set(gca,'XColor', 'none','YColor','none')

png_width = 0.05;
png_height = 0.05;
ypos = [0.39,0.665];
xpos = 0.24;
image_pngs = 'thumbs_up.png';
for n = 1:num_pngs
    axes('pos',[xpos ypos(n) png_width png_height])
    [img, ~, tr] = imread(image_pngs);
    im = image('CData',img);
    im.AlphaData = tr;
    set(gca,'YDir','reverse');
    set(gca,'color','none');
    set(gca,'XColor', 'none','YColor','none')
end

png_width = 0.07;
png_height = 0.07;
xpos = [0.135-0.01,0.345,0.135-0.01,0.345];
ypos = [0.807,0.807,0.532,0.532] - 0.015;
num_pngs = 4;
image_pngs = {'pretzel.png','baguette.png','bread.png','baguette.png'};
for n = 1:num_pngs
    axes('pos',[xpos(n) ypos(n) png_width png_height])
    [img, ~, tr] = imread(image_pngs{n});
    im = image('CData',img);
    im.AlphaData = tr;
    set(gca,'YDir','reverse');
    set(gca,'color','none');
    set(gca,'XColor', 'none','YColor','none')
end

png_width = 0.05;
png_height = 0.05;
xpos = [0.125,0.24];
ypos = [0.2,0.53];
image_pngs = 'ques_mark.png';
num_pngs = 2;
for n = 1:num_pngs
    axes('pos',[xpos(n) ypos(n) png_width png_height])
    [img, ~, tr] = imread(image_pngs);
    im = image('CData',img);
    im.AlphaData = tr;
    set(gca,'YDir','reverse');
    set(gca,'color','none');
    set(gca,'XColor', 'none','YColor','none');
end

% CREATE BARS
bar_width = 0.1; % width for bar plot
bar_height = 0.1; % height for bar plot
xpos = 0.46; % x-position
ypos = [0.225,0.08] - 0.025; % y-position
bar_data = [0.1,0.8;0.65,0.7]; % bar data
ylabel_strings = {'Salience','Value'}; % strings for y
num_plots = 2;
for n = 1:num_plots
    axes('pos',[.46 ypos(n) bar_width bar_height])
    set(gca,'XColor', 'none','YColor','none')
    bar(bar_data(n,:),'BarWidth',0.5,'FaceColor',[65, 111, 71]./255)
    ylim([0,1])
    yticks([0,1])
    yticklabels(["Low","High"])
    set(gca,'color','none','LineWidth',linewidth_axes,'FontName',font_name)
    set(gca,'Xticklabel',[],'FontSize',font_size)
    yl = ylabel(ylabel_strings(n));
    yl.Position = yl.Position;% + [-1,0,0];
    box off
end

% SUBPLOT LABELS

ax1_pos = ax1_new.Position;
adjust_x = -0.06; % adjust x-axis position of subplot label
adjust_y = ax1_pos(4) - 0.01; % adjust y-axis position of subplot label
[label_x,label_y] = change_plotlabel(ax1_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'a','FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment',horz_align)
[label_x,label_y] = change_plotlabel(ax3_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'b','FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment',horz_align)
[label_x,label_y] = change_plotlabel(ax5_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'd','FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment',horz_align)

ax2_pos = ax2_new.Position;
adjust_x = -0.08;
adjust_y = ax2_pos(4) - 0.01;
[label_x,label_y] = change_plotlabel(ax2_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'c','FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment',horz_align)

% SAVE FIGURE

fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, fullfile(save_dir,filesep,"figure1_6.png"), '-dpng', '-r1200') 