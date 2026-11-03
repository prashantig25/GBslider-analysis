% figure6_SM creates figure S6 and plots splithalf reliability test results

clc
clearvars

linewidth_line = 1.5; % linewidth for plot lines
line_width = 0.5; % linewidth for axes
font_name = 'Arial'; % font name
font_size = 6; % font size
[~,~,~,~,~,~,darkblue_muted,~,~,~,~,~,~,~,...
    ~,~,~,fits_colors,~] = colors_rgb(); % colors

% PATH STUFF

currentDir = cd;
reqPath = 'GBslider-analysis'; % to which directory one must save in
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
baseDir = fullfile(desiredPath, 'Data', 'LR analyses');
betas_split0_abs_path = fullfile(baseDir, 'betas_abs_split0.mat');
betas_split1_abs_path = fullfile(baseDir, 'betas_abs_split1.mat');
betas_split0_signed_path = fullfile(baseDir, 'betas_signed_split0.mat');
betas_split1_signed_path = fullfile(baseDir, 'betas_signed_split1.mat');

% LOAD THE DATA

betas_split0_abs = importdata(betas_split0_abs_path); % betas from absolute analysis for splithalf group 0
betas_split1_abs = importdata(betas_split1_abs_path); % betas from absolute analysis for splithalf group 1
betas_split0_signed = importdata(betas_split0_signed_path); % betas from signed analysis for splithalf group 0
betas_split1_signed = importdata(betas_split1_signed_path); % betas from signed analysis for splithalf group 1

% INITIALISE TILE LAYOUT

figure
set(gcf,'Position',[100 100 600 400])
t = tiledlayout(2,3);
t.Padding = 'compact';
t.TileSpacing = 'compact';
ax1 = nexttile(1,[1,1]);
ax2 = nexttile(2,[1,1]);
ax3 = nexttile(3,[1,1]);
ax4 = nexttile(4,[1,1]);
ax5 = nexttile(5,[1,1]);
ax6 = nexttile(6,[1,1]);

% PLOT

vars = [1,2,5]; % variables that need to be plotted

% SIGNED LR ANALYSIS -- shows a regressor-name title header (deliberate,
% kept different from the absolute analysis below -- see conversation)
title_variables = ["Fixed LR","Belief-state-adapted LR","Confirmation bias"]; % title strings
axes_new = plot_splithalf_corr(betas_split0_signed,betas_split1_signed,vars, ...
    [ax1,ax2,ax3],fits_colors,title_variables,font_name,font_size,line_width);

% ABSOLUTE LR ANALYSIS -- no regressor-name title header (deliberate)
axes_new_abs = plot_splithalf_corr(betas_split0_abs,betas_split1_abs,vars, ...
    [ax4,ax5,ax6],darkblue_muted,[],font_name,font_size,line_width);

% SUBPLOT LABELS

ax1_pos = axes_new(1).Position;
adjust_x = -0.06; % adjusted x-position for subplot label
adjust_y = ax1_pos(4) + 0.02; % adjusted y-position for subplot label
[label_x,label_y] = change_plotlabel(axes_new(1),adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'a','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(axes_new(2),adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'b','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(axes_new(3),adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'c','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(axes_new_abs(1),adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'd','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(axes_new_abs(2),adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'e','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(axes_new_abs(3),adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'f','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

% SAVE FIGURE

fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, fullfile(save_dir,filesep,'figure8_SM1.png'), '-dpng', '-r600') 