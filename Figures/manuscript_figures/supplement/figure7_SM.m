% figure7_SM creates figure S7 and plots the relationship between
% confirmation bias and signed estimation errors

clc
clearvars

[~,~,~,~,~,~,darkblue_muted,~,~,~,gray_dots,~,~,~,...
    reg_color,dots_edges,~,~,~] = colors_rgb(); % colors

% PATH STUFF

currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)'; % to which directory one must save in
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    disp('Current directory is already the desired path. No need to run createSavePaths.');
    desiredPath = currentDir;
else
    % Call the function to create the desired path
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, filesep, "saved_figures",filesep,"supplement");
mkdir(save_dir)
baseDir = fullfile(desiredPath, 'Data', 'estimation error analysis');
lm_path = fullfile(baseDir, 'lm_signed_esterror_signed_lr.mat');
partial_rsq_path = fullfile(baseDir, 'partialrsq_signed_esterror_signed_lr.mat');

% LOAD THE DATA

lm = importdata(lm_path); % estimated model fit to estimation errors
partial_rsq = importdata(partial_rsq_path); % partial R2 values

linewidth_line = 1.5; % linewidth for plot lines
line_width = 0.5; % linewidth for axes
font_name = 'Arial'; % font name
font_size = 6; % font size
xlim_vals = [0.5 1.5]; % x limits
ylim_vals = [0.5 0.9]; % y limits

% INITIALISE TILES

figure 
set(gcf,'Position',[100 100 400 200])
t = tiledlayout(1,2);
t.Padding = 'compact';
ax2 = nexttile(2,[1,1]);
ax1 = nexttile(1,[1,1]);

% PLOT

% ILLUSTRATION OF OVER-ESTIMATION DRIVEN BY CONFIRMATION BIAS

mu = [0.73 0.77]; % example estimated mu to show over-estimation
l1 = line(ax1,[0 4],[0.8 0.8],'Color','k','LineWidth',0.5,'LineStyle','--'); % actual mu
hold on
b = bar(ax1,1,mu,0.7,"grouped","EdgeColor",darkblue_muted, ...
    FaceColor=darkblue_muted,LineWidth=1); % estimate mu bars

% ADJUST PLOT PROPERTIES

b(1).FaceAlpha = 0.3;
b(2).FaceAlpha = 0.6;
set(ax1,'XTick',[]);
adjust_figprops(ax1,font_name,font_size,line_width,xlim_vals,ylim_vals)
box(ax1,"off")
a = annotation("textbox",[0.48,0.83,0.3,0.03],"String","Actual reward probability", ...
    'BackgroundColor','none','FontName',font_name,'FontSize',font_size,'EdgeColor','none');
a.Parent = ax1;
ylabel('Reported reward probability')
title('Confirmation bias in learning','FontWeight','normal')
xticks([0.86,1.15])
xticklabels({'Absent','Present'});
xlim_vals = [-0.4 0.4]; % x limits
ylim_vals = [-0.4 0.15]; % y limits

% RELATIONSHIP BETWEEN SIGNED EST. ERROR vs. CONFIRMATION-BIAS LR

p = plotAdded(ax2,lm,[5],'Marker','o','MarkerSize', ...
        3,'MarkerFaceColor',gray_dots, ...
            'MarkerEdgeColor',dots_edges);

% ADJUST PLOT PROPERTIES
p(1).Color = darkblue_muted;
p(2).Color = darkblue_muted;
p(3).Color = darkblue_muted;

p(2).LineWidth = linewidth_line;
p(3).LineWidth = 0.2;
p(3).LineStyle = '--';
p(1).LineWidth = 0.7;

legend(ax2,'off')
xlabel(ax2,"Confirmation bias")
ylabel(ax2,"Signed estimation error")
title(ax2,strcat("Partial \itR^2\rm =",{' '},num2str(partial_rsq(5),'%.3f')) + newline + strcat({' '},...
        "\itp\rm = ",{''},num2str(lm.Coefficients.pValue(6),'%.3f')), ...
'FontWeight','normal','Interpreter','tex')
adjust_figprops(ax2,font_name,font_size,line_width)
box(ax2,"off")

% ADD SUBPLOT LABELS

ax1_pos = ax1.Position;
adjust_x = -0.08; % adjust x-axis position of subplot label
adjust_y = ax1_pos(4) + 0.07; % adjust y-axis position of subplot label
[label_x,label_y] = change_plotlabel(ax1,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'a','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(ax2,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'b','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

% SAVE FIGURE

fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, fullfile(save_dir,'figureSM_signed_est_bars.png'), '-dpng', '-r600') 