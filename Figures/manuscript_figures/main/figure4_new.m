clc
clearvars

line_width = 0.5; % line width for axes
font_size = 6; % font size
font_name = 'Arial'; % font name
linewidth_plot = 1; % line width for plot lines
save_csv = 0; % if figure caption needs any stats input, then save
currentDir = pwd; 
save_figures = "stats_saved\figures"; % path to save stats for figures
mkdir(save_figures);
[~,high_PU,mid_PU,low_PU,~,~,darkblue_muted,~,~,~,~,light_gray,binned_dots,barface_green,...
    reg_color,~,~,~,~] = colors_rgb(); % colors

% INITIALISE VARS
load("betas_signed_wo_rewunc_obj.mat","betas_all"); betas_subjs = betas_all; % participant betas
load("betas_agent_recoding_wo_rewunc.mat","betas_agent"); % agent betas
load("p_vals_signed_wo_rewunc_obj.mat","p_vals"); % p-vals from ttest
data_subjs = readtable("preprocessed_GB2sensitivity.txt"); % participant lr data
id_subjs = unique(data_subjs.id); % subject IDs
num_subjs = length(id_subjs); % number of subjects
example_participant = 23; % example participant
num_vars = 5; % number of coefficients
weight_y_n = 0; % weighted regression
fontsize_label = 12; % font size for subplot labels
model3 = 'up ~ pe + pe:contrast_diff + pe:congruence + pe:pe_sign + pe:salience';
mdl = model3; % which regression model
pred_vars = {'pe','salience','contrast_diff','congruence','condition','reward_unc','subj_est_unc' ...
        ,'reward','mu','pe_sign','pu'}; % cell array with names of predictor variables
resp_var = 'up'; % name of response variable
cat_vars = {'salience','congruence','condition','reward_unc','pe_sign'}; % cell array with names of categorical variables
%% INITIALISE TILE LAYOUT

figure
set(gcf,'Position',[100 100 400 400])
t = tiledlayout(2,4);
% t.TileSpacing = 'compact';
ax1 = nexttile(1,[1,2]);
ax2 = nexttile(3,[1,1]);
ax3 = nexttile(4,[1,1]);
ax4 = nexttile(5,[1,2]);
ax5 = nexttile(7,[1,2]);
%% PLOT LR FOR CONTRAST DIFFERENCE BINS

% CHANGE AXES POSITION
position_change = [0 -0.05 0 0]; % change in position
new_pos = change_position(ax1,position_change); % new position
ax1_new = axes('Units', 'Normalized', 'Position', new_pos); % update
box(ax1_new, 'off'); % remove box
delete(ax1); % delete old axis

% INITIALISE VARS TO BE PLOTTED
binned_data = abs(data_subjs.con_diff); % absolute contrast difference
nbins = 10; % number of bins
bin_edges = prctile(binned_data, 0:10:100); % calculate percentile edges
bins = discretize(binned_data, bin_edges); % bin contrast differences 
data_subjs.lr = data_subjs.up./data_subjs.pe; % learning rates
data_subjs.abs_lr = abs(data_subjs.lr); % absolute learning rates

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = data_subjs.id(data_subjs.pe ~= 0 & abs(data_subjs.lr)<=2);
y_data = data_subjs.lr(data_subjs.pe ~= 0 & abs(data_subjs.lr)<=2);
bins = bins(data_subjs.pe ~= 0 & abs(data_subjs.lr)<=2);
binned_data = binned_data(data_subjs.pe ~= 0 & abs(data_subjs.lr)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,num_subjs); 
avg_behv_bins = NaN(nbins,num_subjs); 
for b = 1:nbins % run for each bin
    for n = 1:num_subjs % run for each subject
        bins_subj = bins(run_id == id_subjs(n)); 
        y_data_subj = y_data(run_id == id_subjs(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == id_subjs(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(num_subjs);

% PLOT
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
ls.Color = 'k';
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',line_width,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',binned_dots);
xlabel('Contrast-difference bins (1 bin = 0.01)')
ylabel('Mean learning rate (LR)')

% ADJUST FIGURE PROPERTIES
xlim_vals = [0 10.3];
ylim_vals = [0 0.2];
adjust_figprops(ax1_new,font_name,font_size,line_width,xlim_vals,ylim_vals);
[rho,pval] = corr(avg_ydata,avg_binneddata, 'rows', 'pairwise');
title(strcat("\itr\rm = ",{' '},num2str(round(rho,2)),{' '}) + newline + "\itp\rm < 0.01", ...
    'FontWeight','normal','Interpreter','tex')
if save_csv == 1
    save_table = table("subplot_a",round(rho,2),round(pval,3),8,'VariableNames',{'name','rho','pval','df'});
    writetable(save_table,fullfile(save_figures,'figure4.csv'));
end
%% PLOT BETA COEFFICIENTS

% INITIALISE VARS FOR PLOTTING COEFFICIENTS
regressors = [1,2]; % regressors that need to be plotted
axes_old = [ax2,ax3]; % names of old axes
ax2_new = ax2; % initialise new axes
ax3_new = ax3;
ax4_new = ax4;
axes_new = [ax2_new,ax3_new,ax4_new];
position_change = [0, -0.05, 0, 0.02]; % position changes for each axes
adjust_position = 0.015;
ylim_lower = [-0.3,-0.15,-0.2]; % lower y-axis limit for each regressor
ylim_upper = [1,0.4,0.6]; % upper y-axis limit for each regressor
xlabelname = {''}; % x-axis label
ylabelname = {'Fixed LR (\beta_1)','Belief-state-adapted LR (\beta_2)','Confirmation bias'}; % y-axis label name for each regressor
disp_pval = 0; % if p-val stars should be displayed on top of bars
scatter_dots = 1; % if single participant data should be scattered on top of bar
dot_size = 10; % scatter dot size
plot_err = 1; % if error bar should be plotted
disp_legend = [0,1,0]; % if legend should be displayed
xticklabs = {''}; % x-tick labels
y_label = 1; % if p-val stars to be displayed, initialise y-axis location

% PLOT
for r = 1:length(regressors)

    % CHANGE AXES POSITION
    new_pos = change_position(axes_old(r),position_change);
    axes_new(r) = axes('Units', 'Normalized', 'Position', new_pos);
    box(axes_new(r), 'off');
    delete(axes_old(r));
    
    % GET MEAN AND SEM FOR BETAS
    selected_regressors = regressors(r);
    [mean_avg,mean_sd,coeffs_subjs] = prepare_betas(betas_all,selected_regressors,num_subjs);
    
    % GET STARS FOR CORRESPONDING REGRESSOR'S P-VALUES
    bar_labels = {'*'};
    pstars = pvals_stars(p_vals,selected_regressors,bar_labels,0);
    title_name = pstars;
    colors_name = darkblue_muted;
   
    agent_means = mean(betas_agent,1);
    agent_means = agent_means(selected_regressors);
    
    hold on
    h = bar_plots_pval(coeffs_subjs,mean_avg,mean_sd,num_subjs, ...
        length(selected_regressors),1,{'Empirical data','Example participant','Normative agent'}, ...
        xticks,xticklabs,title_name,xlabelname,ylabelname(r),disp_pval,scatter_dots, ...
        dot_size,plot_err,font_size,linewidth_plot,font_name,disp_legend(r),colors_name, ...
        bar_labels,y_label,agent_means,[0.5,1.5],example_participant);
    h.BarWidth = 0.4; 
    ylim_vals = [ylim_lower(r) ylim_upper(r)];
    xlim_vals = [0.5 1.5];
    adjust_figprops(axes_new(r),font_name,font_size,line_width,xlim_vals,ylim_vals);
end

%% ADDED VARIABLE PLOT

position_change = [0, -0.03, 0, 0];
new_pos = change_position(ax4,position_change);
ax5_new = axes('Units', 'Normalized', 'Position', new_pos);
box(ax5_new, 'off');
delete(ax4);

data = data_subjs;

% FIT THE MODEL
for i = example_participant
    tbl = table(data.pe(and(data.id == id_subjs(i),data.pe ~= 0)), ...
        data.up(and(data.id == id_subjs(i),data.pe ~= 0)), ...
        round(data.norm_condiff(and(data.id == id_subjs(i),data.pe ~= 0)),2), ...
        data.contrast(and(data.id == id_subjs(i),data.pe ~= 0)),...
        data.condition(and(data.id == id_subjs(i),data.pe ~= 0)),...
        data.congruence(and(data.id == id_subjs(i),data.pe ~= 0)),...
        data.reward_unc(and(data.id == id_subjs(i),data.pe ~= 0)),...
        data.pe_sign(and(data.id == id_subjs(i),data.pe ~= 0)),...
        'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
        ,'reward_unc','pe_sign'});
    
    [~,~,~,~,lm] = linear_fit(tbl,mdl,pred_vars,resp_var, ...
        cat_vars,num_vars,weight_y_n);

    % ADDED VARIABLE PLOT
    hold on
    p = plotAdded(lm,[2,3],'Marker','o','MarkerSize',3,'MarkerFaceColor',[220, 220, 220]./255, ...
        'MarkerEdgeColor',[184, 184, 184]./255);

    % FIGURE PROPERTIES
    p(1).Color = darkblue_muted;
    p(2).Color = darkblue_muted;
    p(3).Color = darkblue_muted;

    p(2).LineWidth = 1.5;
    p(3).LineWidth = 0.2;
    p(1).LineWidth = 0.7;

    p(3).LineStyle = '--';

    legend('off')
    xline(0,"LineWidth",0.5,LineStyle="--")
    yline(0,"LineWidth",0.5,LineStyle="--")
    xlabel('Belief-state-adapted LR')
    ylabel('Update')
    title('')
    adjust_figprops(ax5_new,font_name,font_size,line_width);
    ylim([-0.3,0.2])
    xlim([-0.75,0.75])
end
box off
%% INTERACTION PLOTS

position_change = [0, -0.03, 0, 0]; 
new_pos = change_position(ax5,position_change);
ax6_new = axes('Units', 'Normalized', 'Position', new_pos);
box(ax6_new, 'off');
delete(ax5);

% INTERACTION PLOT
hold on
h = plotInteraction(lm,'contrast_diff','pe','predictions');

% PLOT PROPERTIES
h(3).Color = low_PU;
h(2).Color = mid_PU;
h(1).Color = high_PU;
xlabel('Prediction error')
ylabel('Update')
title('')
adjust_figprops(ax6_new,font_name,font_size,line_width);
l = legend('Contrast difference','0','0.5','1','Location','best','AutoUpdate','off');
l.EdgeColor = 'none';
l.Color = 'none';
l.ItemTokenSize = [7 7];
box off
ylim([-0.4,0.4])
xline(0,"LineWidth",0.5,LineStyle="--")
yline(0,"LineWidth",0.5,LineStyle="--")

sgtitle('Update = \beta_0 + \beta_1 \cdot \delta + \beta_2 \cdot \delta \cdot Contrast difference + ... + \epsilon', ...
    'Interpreter','Tex','FontSize',7,'FontName','Arial')
%% ADD SUBPLOT LABELS

ax1_pos = ax1_new.Position;
adjust_x = - 0.08; % adjust x-position of subplot label
adjust_y = ax1_pos(4) + 0.01; % adjust y-position of subplot label
[label_x,label_y] = change_plotlabel(ax1_new,adjust_x,adjust_y);

annotation("textbox",[label_x label_y .05 .05],'String', ...
    'a','FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(axes_new(1),adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'b','FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment','center')
[label_x,label_y] = change_plotlabel(axes_new(2),adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'c','FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(ax5_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'd','FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment','center')
[label_x,label_y] = change_plotlabel(ax6_new,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'e','FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment','center')
%% SAVE FIGURE

fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, 'learning2.png', '-dpng', '-r600') 