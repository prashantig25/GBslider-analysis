%% INITIALISE VARS

clc
clearvars

linewidth_axes = 0.5; % line width for axes
font_name = 'Arial'; % font
font_size = 6; % font size
horz_align = 'center'; % horizontal alignment for text
vert_align = 'middle'; % vertical alignment for text
colors_manuscript; % colors
face_alpha = 0; % face alpha for boxes
headlength_arrow = 5; % head length for arrow
fontsize_label = 12; % font size for subplot labels
%% INITIALISE TILE LAYOUT

figure("Position",[150,150,380,350])
t = tiledlayout(3,2);
t.TileSpacing = 'compact';
t.Padding = 'compact';
ax1 = nexttile(1);
ax2 = nexttile(2, [2,1]);
ax3 = nexttile(3);
ax5 = nexttile(5, [1,2]);

%% H1: low perceptual uncertainty

% CHANGE AXES POSITION
position_change = [0, -0.05, 0, 0.015]; % adjusted position
new_pos = change_position(ax1,position_change);
ax1_new = axes('Units', 'Normalized', 'Position', new_pos);
box(ax1_new, 'on');
delete(ax1);
axis([0 10 0 10]);
set(ax1_new, 'YTick', [], 'XTick', []);
hold on

% ADD TITLE
h1 = sgtitle('H1: Perceptual uncertainty drives speed of learning');
h1.FontName = font_name;
h1.FontSize = font_size;
h1.FontWeight = 'bold';

% CREATE BOX AND ADD SHADING
x = [0 10 10 0];
y = [0 0 10 10];
c = [0.532,0.534,0.536,0.538]; % colors
cmap = colormap("gray");
patch(x,y,c);
brighten(.9); % alpha

% ADD TEXT BOXES
all_strings = {'Low perceptual uncertainty (PU)','Tasty!!!','Perception','Reward'};
num_strings = 4; % number of strings
text_xpos = [0,3.9,1.3,1.1]; % x-position
text_ypos = [9,2,4.2,0.5]; % y-posiition
box_width = [10, 2, 2, 2]; % box width
box_height = [1.2, 1, 1, 1]; % box height
bg_color = {pu_box,'none','none','none'}; % background color
edge_color = {'k','none','none','none'}; % edge color
face_alpha = [1,0,0,0];
for n = 1:num_strings
    string = all_strings(n);
    position = [text_xpos(n) text_ypos(n) box_width(n) box_height(n)];
    annotate_textbox(ax1_new,position,string,font_name,font_size, ...
        horz_align,vert_align,bg_color{n},face_alpha(n),edge_color{n},linewidth_axes);
    hold on
end

% ADD RECTANGLE
r1 = rectangle('Position',[1.3 5.2 7 3.4]);
r1.LineWidth = 0.01;
a1 = annotation('arrow','HeadLength',headlength_arrow);
a1.Parent = gca;
a1.X = [5.6 6.3];
a1.Y = [3 5];
hold on
%% H1: High perceptual uncertainty

% CHANGE AXES POSITION
position_change = [0, 0, 0, 0.015]; 
new_pos = change_position(ax3,position_change);
ax3_new = axes('Units', 'Normalized', 'Position', new_pos);
box(ax3_new, 'on');
delete(ax3);

% CREATE BOX AND ADD GRAY SHADING
axis([0 10 0 10])
set(ax3_new, 'YTick', [], 'XTick', []);
cmap = colormap("gray");
patch(ax3_new,x,y,c);
brighten(.9);

% ADD TEXT BOXES
all_strings = {'High perceptual uncertainty (PU)','Tasty!!!','Perception','Reward'};
num_strings = 4;
box_height = [1, 1, 1, 1];
face_alpha = [1,0,0,0];
for n = 1:num_strings
    string = all_strings(n);
    position = [text_xpos(n) text_ypos(n) box_width(n) box_height(n)];
    annotate_textbox(ax3_new,position,string,font_name,font_size, ...
        horz_align,vert_align,bg_color{n},face_alpha(n),edge_color{n},linewidth_axes);
end

r2=rectangle('Position',[1.3 5.2 7 3.4]);
r2.LineWidth = 0.01;

% ADD ARROW
a1 = annotation('arrow','HeadLength',headlength_arrow);
a1.Parent = gca;
a1.X = [5.6 6.3];
a1.Y = [3 5];
a1.Color = [0.5 0.5 0.5 0.85];
a1 = annotation('arrow','HeadLength',headlength_arrow);
a1.X = [4.1 3.4];
a1.Y = [3 5];
a1.Color = [0.5 0.5 0.5 1];
%% PLOT LEARNING RATES

% CHANGE AXES POSITION
position_change =  [0.02, 0.05, -0.02, -0.09]; 
new_pos = change_position(ax2,position_change);
ax2_new = axes('Units', 'Normalized', 'Position', new_pos);
box(ax2_new, 'on');
delete(ax2);

% INITIALISE VARS
axis([-1 1 0 1])
pe = linspace(-1,1,5);
up_highPU = pe * 0.1;
up_lowPU = pe * 0.4;
up_midPU = pe * 0.25;
up = [up_highPU;up_lowPU;up_midPU;zeros(5,1).';pe;];
linestyles = {':',':',':','-','-',};
linewidths = [1.5,1.5,1.5,0.5,0.5];
linecolors = {high_PU,low_PU,mid_PU,'k','k',};

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
l = legend({'Low PU (LR = 0.4)','Medium PU (LR = 0.25)','High PU (LR = 0.1)'}, ...
    'FontSize',font_size,'Color','none','EdgeColor', ...
    'none','Location','best','FontName',font_name,'AutoUpdate','off');
l.ItemTokenSize = [7 7];

% ADD ROTATED TEXT BOXES
xpos = [0.8,-0.1];
ypos = 0.9;
rotate = [50, 0];
allstrings = {'LR = 1','LR = 0'};
num_strings = 2;
for n = 1:num_strings
    txt = text(xpos(n),ypos,allstrings{1,n});
    txt.FontSize = font_size + 2;
    txt.FontWeight = 'normal';
    txt.Rotation = rotate(n);
    txt.LineStyle = 'none';
    txt.FontName = font_name;
    txt.HorizontalAlignment = horz_align;
    txt.Parent = ax2_new;
end
%% H2 Box

% ADD TITLE
h2 = title(ax5,'H2: Visual salience drives choices under reward uncertainty');
h2.FontName = font_name;
h2.FontSize = font_size+1;
h2.FontWeight = 'bold';

% CREATE BOX AND ADD GRAY SHADING
set(ax5,'xtick',[])
set(ax5,'ytick',[])
hold on
axis(ax5,[0 10 0 10])
x = [0 10 10 0];
y = [0 0 10 10];
c = [0.532,0.534,0.536,0.538];
cmap = colormap("gray");
patch(ax5,x,y,c);
brighten(0.9);

% ADD ARROWS
num_arrows = 2;
xpos = [1.7 3.5];
ypos = [5 8;4.8 1.8];
for n = 1:num_arrows
    a1 = annotation('arrow','HeadLength',headlength_arrow);
    a1.Parent = ax5;
    a1.Color = gray_arrow;
    a1.X = xpos;
    a1.Y = ypos(1,:);
end
a1 = annotation('arrow','HeadLength',5);
a1.Parent = ax5;
a1.X = [7 8.3];
a1.Y = [5 5];
a1.Color = 'k';
a1.LineWidth = 1.5;

% ADD LINES
linewidths = [1.5,0.5];
xpos = [6 7];
ypos = [7 5;3 5];
num_lines = 2;
for n = 1:num_lines
    a1 = annotation('line');
    a1.Parent = ax5;
    a1.X = xpos;
    a1.Y = ypos(n,:);
    a1.Color = 'k';
    a1.LineWidth = 1.5;
end
%% ADD EXTERNAL PNGs

png_width = 0.1; % width of PNG
png_height = 0.09; % height of PNG
xpos = [0.11,0.82]; % x-position
ypos = 0.16; % y-position
num_pngs = 2; % number of pngs
image_pngs = {'bread_choice.png','bread_bias.png'}; % path
for n = 1:num_pngs
    axes('pos',[xpos(n) ypos png_width png_height])
    [img, ~, tr] = imread(image_pngs{n}); % add image
    im = image('CData',img);
    im.AlphaData = tr;
    set(gca,'YDir','reverse'); % correct for reversed image
    set(gca,'color','none'); % make it transparent
    set(gca,'XColor', 'none','YColor','none')
end

png_width = 0.025;
png_height = 0.04;
xpos = [0.4825,0.515];
ypos = [0.21,0.28];
image_pngs = {'bun_bw.png','sourdough_bw.png'};
for n = 1:num_pngs
    axes('pos',[xpos(n) ypos(n) png_width png_height])
    [img, ~, tr] = imread(image_pngs{n});
    im = image('CData',img);
    im.AlphaData = tr;
    set(gca,'YDir','reverse');
    set(gca,'color','none')
    set(gca,'XColor', 'none','YColor','none')
end

axes('pos',[0.65 0.14 .08 .08])
set(gca,'XColor', 'none','YColor','none')
[img, ~, tr] = imread("integration.png");
im = image('CData',img);
im.AlphaData = tr;
set(gca,'YDir','reverse');
set(gca,'color','none')
colormap(gca,"bone"); 
set(gca,'XColor', 'none','YColor','none')

png_width = 0.04;
png_height = 0.04;
ypos = [0.39,0.67];
xpos = 0.24;
image_pngs = 'man.png';
for n = 1:num_pngs
    axes('pos',[xpos ypos(n) png_width png_height])
    [img, ~, tr] = imread(image_pngs);
    im = image('CData',img);
    im.AlphaData = tr;
    set(gca,'YDir','reverse');
    set(gca,'color','none');
    set(gca,'XColor', 'none','YColor','none')
end

png_width = 0.08;
png_height = 0.1;
xpos = [0.115,0.32,0.115,0.32];
ypos = [0.79,0.79,0.515,0.515];
num_pngs = 4;
image_pngs = {'s0_low.png','s1_low.png','s1_high.png','s0_high.png'};
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
xpos = [0.15,0.24];
ypos = [0.21,0.53];
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
ypos = [0.2,0.07]; % y-position
bar_data = [0.1,0.8;0.65,0.7]; % bar data
ylabel_strings = {'Salience','Probability'}; % strings for y
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
%% SUBPLOT LABELS

ax1_pos = ax1_new.Position;
adjust_x = -0.06;
adjust_y = ax1_pos(4) - 0.01;
[label_x,label_y] = change_plotlabel(ax1_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'a','FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment',horz_align)
[label_x,label_y] = change_plotlabel(ax3_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'b','FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment',horz_align)
[label_x,label_y] = change_plotlabel(ax2_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'c','FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment',horz_align)
[label_x,label_y] = change_plotlabel(ax5,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'd','FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment',horz_align)
%% SAVE FIGURE

fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, 'trial_bread3.png', '-dpng', '-r1200') 