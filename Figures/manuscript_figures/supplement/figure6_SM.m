clc
clearvars

linewidth_line = 1.5; % linewidth for plot lines
line_width = 0.5; % linewidth for axes
font_name = 'Arial'; % font name
font_size = 6; % font size
[~,~,~,~,~,~,darkblue_muted,~,~,~,~,~,~,~,...
    ~,~,~,fits_colors,~] = colors_rgb(); % colors

% INITIALIZE VARS

% directory specification
current_Dir = pwd;
save_dir = fullfile("saved_figures",filesep,"supplement");
mkdir(save_dir)

% Define the base directory
baseDir = fullfile('Data', 'LR analyses');

% Construct the file paths using fullfile
betas_split0_abs_path = fullfile(baseDir, 'betas_abs_split0.mat');
betas_split1_abs_path = fullfile(baseDir, 'betas_abs_split1.mat');
betas_split0_signed_path = fullfile(baseDir, 'betas_signed_split0.mat');
betas_split1_signed_path = fullfile(baseDir, 'betas_signed_split1.mat');

% Load the data using importdata
betas_split0_abs = importdata(betas_split0_abs_path); % betas from absolute analysis for splithalf group 0
betas_split1_abs = importdata(betas_split1_abs_path); % betas from absolute analysis for splithalf group 1
betas_split0_signed = importdata(betas_split0_signed_path); % betas from signed analysis for splithalf group 0
betas_split1_signed = importdata(betas_split1_signed_path); % betas from signed analysis for splithalf group 1
%% INITIALISE TILE LAYOUT

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
%% PLOT

axes_old = [ax1,ax2,ax3]; % initialise old axes
axes_new = axes_old; % initialise axes arrays to save new axes

vars = [1,2,5]; % variables that need to be plotted
pvals_signed = {'','',''}; % initialised arrays to store p-values
for p = 1:length(pvals_signed)
    [rho,pval] = corr(betas_split0_signed(:,vars(p)),betas_split1_signed(:,vars(p)));
    if pval < 0.001
        pvals_signed{p} = '\itp\rm < 0.001';
    elseif pval < 0.01
        pvals_signed{p} = strcat("\itp\rm = ",num2str(round(pval,3)));
    else
        pvals_signed{p} = strcat("\itp\rm = ",num2str(round(pval,3)));
    end
end
title_variables = ["Fixed LR","Belief-state-adapted LR","Confirmation bias"]; % title strings

for i = 1:length(axes_old)

    new_pos = change_position(axes_old(i),[0 0 0 -0.05]); % new position for axis
    axes_new(i) = axes('Units', 'Normalized', 'Position', new_pos); % update the position
    delete(axes_old(i)); % delete old axes

    % PLOT
    hold on
    s2 = scatter(betas_split0_signed(:,vars(i)),betas_split1_signed(:,vars(i)),'filled','o', ...
        'MarkerFaceColor',[220, 220, 220]./255, ...
        'MarkerEdgeColor',[184, 184, 184]./255,'LineWidth',0.7,'MarkerFaceAlpha',1,SizeData=10);
    hold on

    % PLOT PROPERTIES
    xlabel('Odd trials')
    ylabel('Even trials')
    [rho,pval] = corr(betas_split0_signed(:,vars(i)),betas_split1_signed(:,vars(i)));
    title(title_variables(i) + newline + strcat("\itr\rm = ",{''},num2str(round(rho,2))) + newline + ...
        pvals_signed{i},'FontWeight','normal')
    adjust_figprops(axes_new(i),font_name,font_size,line_width)
    box("off")

    % CONFIDENCE INTERVALS
    mdl = fitlm(betas_split0_signed(:,vars(i)),betas_split1_signed(:,vars(i))); % fit a linear model
    xrange = xlim;
    xvalues = linspace(xrange(1), xrange(2), 100); % predictor values
    [yfit, yci] = predict(mdl, xvalues'); % compute and plot confidence interval lines
    hold on
    plot(xvalues, yfit,LineWidth=1.5, Color=fits_colors); % plot the fitted line
    plot(xvalues, yci, Color=fits_colors, LineWidth=0.2,LineStyle='--'); % plot the upper confidence interval
    plot(xvalues, yci(:,1), Color=fits_colors, LineWidth=0.2,LineStyle='--'); % plot the lower confidence interval

end

% INITIALISE
axes_old = [ax4,ax5,ax6];
axes_new_abs = axes_old;
vars = [1,2,5];
pvals_abs = {'','',''};

% P-VALUES
for p = 1:length(pvals_abs)
    [rho,pval] = corr(betas_split0_abs(:,vars(p)),betas_split1_abs(:,vars(p)));
    if pval < 0.001
        pvals_abs{p} = '\itp\rm < 0.001';
    elseif pval < 0.01 && pval > 0.001
        pvals_abs{p} = strcat("\itp\rm = ",num2str(round(pval,3)));
    else
        pvals_abs{p} = strcat("\itp\rm = ",num2str(round(pval,3)));
    end
end

for i = 1:length(axes_old)

    new_pos = change_position(axes_old(i),[0 0 0 -0.05]);
    axes_new_abs(i) = axes('Units', 'Normalized', 'Position', new_pos);
    delete(axes_old(i));
    
    % PLOT
    hold on
    s2 = scatter(betas_split0_abs(:,vars(i)),betas_split1_abs(:,vars(i)),'filled','o','MarkerFaceColor',[220, 220, 220]./255, ...
        'MarkerEdgeColor',[184, 184, 184]./255,'LineWidth',0.7,'MarkerFaceAlpha',1,SizeData=10);
    hold on

    % PLOT PROPERTIES
    xlabel('Odd trials')
    ylabel('Even trials')
    [rho,pval] = corr(betas_split0_abs(:,vars(i)),betas_split1_abs(:,vars(i)));
    title(strcat("\itr\rm = ",{''},num2str(round(rho,2))) + newline + ...
        pvals_abs{i},'FontWeight','normal')
    adjust_figprops(axes_new_abs(i),font_name,font_size,line_width)
    box("off")

    % CONFIDENCE INTERVAL
    mdl = fitlm(betas_split0_abs(:,vars(i)),betas_split1_abs(:,vars(i))); % Fit a linear model
    xrange = xlim;
    xvalues = linspace(xrange(1), xrange(2), 100);
    [yfit, yci] = predict(mdl, xvalues'); % Compute and plot confidence interval lines
    hold on
    plot(xvalues, yfit,LineWidth=1.5, Color=darkblue_muted); % Plot the fitted line
    plot(xvalues, yci, Color=darkblue_muted, LineWidth=0.2,LineStyle='--'); % Plot the upper confidence interval
    plot(xvalues, yci(:,1), Color=darkblue_muted, LineWidth=0.2,LineStyle='--'); % Plot the lower confidence interval
end

%% SUBPLOT LABELS
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
%% SAVE FIGURE

fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, fullfile(save_dir,filesep,'figure8_SM1.png'), '-dpng', '-r600') 