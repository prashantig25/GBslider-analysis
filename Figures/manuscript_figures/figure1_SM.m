clc
clearvars
colors_manuscript; % colors for the plot
line_width = 0.5; % line width for axes
font_size = 6; % font size
font_name = 'Arial'; % font name
linewidth_plot = 1; % line width for plot lines

% INITIALISE VARS
load("betas_abs.mat","betas_all"); betas_subjs = betas_all; % participant betas
load("betas_abs_agent.mat","betas_all"); betas_agent = betas_all; % agent betas
load("betas_signed.mat","betas_all"); betas_signed = betas_all; % participant betas from signed analysis

load("p_vals_abs.mat","p_vals"); % p-vals from ttest
data_subjs = readtable("lr_data.xlsx"); % participant lr data
id_subjs = unique(data_subjs.run_id); % subject IDs
num_subjs = length(id_subjs); % number of subjects
example_participant = 80; % example participant
num_vars = 7; % number of coefficients
weight_y_n = 0; % weighted regression
fontsize_label = 12; % font size for subplot labels
model3 = 'up ~ pe + pe:contrast_diff + pe:congruence + pe:subj_est_unc + pe:reward_unc + pe:pe_sign + pe:salience';
mdl = model3; % which regression model
pred_vars = {'pe','salience','contrast_diff','congruence','condition','reward_unc','subj_est_unc' ...
        ,'reward','mu','pe_sign','pu'}; % cell array with names of predictor variables
resp_var = 'up'; % name of response variable
cat_vars = {'salience','congruence','condition','reward_unc','pe_sign'}; % cell array with names of categorical variables
%% INITIALISE TILE LAYOUT

figure
set(gcf,'Position',[100 100 700 400])
t = tiledlayout(2,9);
t.TileSpacing = 'compact';
ax3 = nexttile(7,[1 1]);
ax4 = nexttile(8,[1 1]);
ax5 = nexttile(9,[1 1]);

%% PLOT ABSOLUTE UPs for PE BINS

% INITIALISE AXES
ax1 = nexttile(1,[1,3]);
box(ax1,"off");

% INITIALISE VARS TO BE PLOTTED
binned_data = abs(data_subjs.pe_actual_corr); % absolute contrast difference
nbins = 10; % number of bins
bin_edges = prctile(binned_data, 0:10:100); % calculate percentile edges
bins = discretize(binned_data, bin_edges); % bin contrast differences 
data_subjs.lr = data_subjs.up_actual_corr./data_subjs.pe_actual_corr; % learning rates
data_subjs.abs_lr = abs(data_subjs.lr); % absolute learning rates

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = data_subjs.run_id(data_subjs.pe_actual_corr ~= 0 & abs(data_subjs.lr)<=2);
y_data = abs(data_subjs.up_actual_corr(data_subjs.pe_actual_corr ~= 0 & abs(data_subjs.lr)<=2));
bins = bins(data_subjs.pe_actual_corr ~= 0 & abs(data_subjs.lr)<=2);
binned_data = binned_data(data_subjs.pe_actual_corr ~= 0 & abs(data_subjs.lr)<=2);

% MEAN ABSOLUTE UP for PE BINS
avg_ydata_bins = NaN(nbins,num_subjs); 
avg_behv_bins = NaN(nbins,num_subjs); 
for b = 1:nbins
    for n = 1:num_subjs
        bins_subj = bins(run_id == id_subjs(n));
        y_data_subj = y_data(run_id == id_subjs(n));
        binned_data_subj = binned_data(run_id == id_subjs(n));
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b));
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b));
    end
end
avg_ydata = nanmean(avg_ydata_bins,2);
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(num_subjs);

% PLOT
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',line_width,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"k",'MarkerFaceColor',binned_dots);
ls = lsline;
ls.Color = 'k';
xlabel('Absolute PE bins (1 bin = 0.01)')
ylabel('Mean absolute UP')

% ADJUST FIGURE PROPERTIES
xlim_vals = [2.5 10.3];
ylim_vals = [0 0.25];
adjust_figprops(ax1,font_name,font_size,line_width,xlim_vals,ylim_vals);
corr_coeff = corr(avg_ydata,avg_binneddata, 'rows', 'pairwise');
title(strcat({'\rho = '},{' '},num2str(round(corr_coeff,2)),{' '},{'P < 0.001'}), ...
    'FontWeight','normal')

%% PLOT ABSOLUTE UPs for CONTRAST DIFFERENCE BINS

% INITIALISE AXES
ax2 = nexttile(4,[1,3]);
box(ax2,"off");

% INITIALISE VARS TO BE PLOTTED
binned_data = abs(data_subjs.con_diff_choice); % absolute contrast difference
nbins = 10; % number of bins
bin_edges = prctile(binned_data, 0:10:100); % calculate percentile edges
bins = discretize(binned_data, bin_edges); % bin contrast differences 

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = data_subjs.run_id(data_subjs.pe_actual_corr ~= 0 & abs(data_subjs.lr)<=2);
y_data = abs(data_subjs.up_actual_corr(data_subjs.pe_actual_corr ~= 0 & abs(data_subjs.lr)<=2));
bins = bins(data_subjs.pe_actual_corr ~= 0 & abs(data_subjs.lr)<=2);
binned_data = binned_data(data_subjs.pe_actual_corr ~= 0 & abs(data_subjs.lr)<=2);

% MEAN ABSOLUTE UP for CONTRAST DIFFERENCE BINS
avg_ydata_bins = NaN(nbins,num_subjs); 
avg_behv_bins = NaN(nbins,num_subjs); 
for b = 1:nbins
    for n = 1:num_subjs
        bins_subj = bins(run_id == id_subjs(n));
        y_data_subj = y_data(run_id == id_subjs(n));
        binned_data_subj = binned_data(run_id == id_subjs(n));
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b));
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b));
    end
end
avg_ydata = nanmean(avg_ydata_bins,2);
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(num_subjs);

% PLOT
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',line_width,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"k",'MarkerFaceColor',binned_dots);
ls = lsline;
ls.Color = 'k';
xlabel('Absolute contrast difference bins (1 bin = 0.01)')
ylabel('Mean absolute UP')

% ADJUST FIGURE PROPERTIES
xlim_vals = [2.5 10.3];
ylim_vals = [0.08 0.14];
adjust_figprops(ax2,font_name,font_size,line_width,xlim_vals,ylim_vals);
corr_coeff = corr(avg_ydata,avg_binneddata, 'rows', 'pairwise');
title(strcat({'\rho = '},{' '},num2str(round(corr_coeff,2)),{' '},{'n.s.'}), ...
    'FontWeight','normal')
%% PLOT BETA COEFFICIENTS

% INITIALISE VARS FOR PLOTTING COEFFICIENTS
regressors = [1,2,7]; % regressors that need to be plotted
axes_old = [ax3,ax4,ax5]; % names of old axes
ylim_lower = [-0.4,-0.15,-0.4]; % lower y-axis limit for each regressor
ylim_upper = [1.2,0.4,1]; % upper y-axis limit for each regressor
xlabelname = {''}; % x-axis label
ylabelname = {'Fixed LR','Belief states adapted LR','Confirmation bias adapted LR'}; % y-axis label name for each regressor
disp_pval = 0; % if p-val stars should be displayed on top of bars
scatter_dots = 1; % if single participant data should be scattered on top of bar
dot_size = 5; % scatter dot size
plot_err = 1; % if error bar should be plotted
disp_legend = [0,1,0]; % if legend should be displayed
xticklabs = {''}; % x-tick labels
y_label = 1; % if p-val stars to be displayed, initialise y-axis location

% PLOT
for r = 1:length(regressors)

    % INITIALISE AXES
    axes_old(r) = nexttile(r+6,[1,1]);
    
    % GET MEAN AND SEM FOR BETAS
    selected_regressors = regressors(r);
    [mean_avg,mean_sd,coeffs_subjs] = prepare_betas(betas_all,selected_regressors,num_subjs);
    
    % GET STARS FOR CORRESPONDING REGRESSOR'S P-VALUES
    bar_labels = {'*'};
    pstars = pvals_stars(p_vals,selected_regressors,bar_labels);
    title_name = pstars;
    colors_name = barface_green;
   
    agent_means = mean(betas_agent,1);
    agent_means = agent_means(selected_regressors);
    
    hold on
    h = bar_plots_pval(coeffs_subjs,mean_avg,mean_sd,num_subjs, ...
        length(selected_regressors),1,{'Empirical data','Example participant'}, ...
        xticks,xticklabs,title_name,xlabelname,ylabelname(r),disp_pval,scatter_dots, ...
        dot_size,plot_err,font_size,linewidth_plot,font_name,disp_legend(r),colors_name, ...
        bar_labels,y_label,agent_means,[0.5,1.5],example_participant);
    h.BarWidth = 0.4; 
    ylim_vals = [ylim_lower(r) ylim_upper(r)];
    xlim_vals = [0.5 1.5];
    adjust_figprops(axes_old(r),font_name,font_size,line_width,xlim_vals,ylim_vals);
end

%% COMPARE ABSOLUTE vs. SIGNED LRs

% INITIALISE TILE
ax6 = nexttile(13,[1,3]);
box(ax6,"off");

% GET VARS
pe_condiff_abs = betas_subjs(:,2);
pe_condiff_signed = betas_signed(:,2);

% PLOT
hold on
scatter(pe_condiff_abs,pe_condiff_signed,'MarkerEdgeColor',[184, 184, 184]./255,'MarkerFaceColor',[220, 220, 220]./255 ...
    ,'SizeData',20,'XJitterWidth',0.5,'YJitterWidth',0.5)
ls = lsline;
ls.Color = 'k';
xlabel('BS adapted absolute LR')
ylabel('BS adapted signed LR')
title(strcat('\rho = ',{' '},num2str(round(corr(pe_condiff_abs,pe_condiff_signed),2)),{' '},"P < 0.001"),'FontWeight','normal')
set(gca,'Color','none','FontName','Arial','FontSize',font_size)
set(gca,'LineWidth',line_width)
%% INTERACTION PLOTS

% EXAMPLE PARTICIPANT
i = 80; % example participant
data = data_subjs;
tbl = table(abs(data.pe_actual_corr(and(data.run_id == id_subjs(i),data.pe_actual_corr ~= 0))), ...
    abs(data.up_actual_corr(and(data.run_id == id_subjs(i),data.pe_actual_corr ~= 0))),...
    round(data.norm_condiff(and(data.run_id == id_subjs(i),data.pe_actual_corr ~= 0)),2), ...
    data.contrast(and(data.run_id == id_subjs(i),data.pe_actual_corr ~= 0)),...
    data.condition(and(data.run_id == id_subjs(i),data.pe_actual_corr ~= 0)),...
    data.congruence(and(data.run_id == id_subjs(i),data.pe_actual_corr ~= 0)),...
    data.reward_unc(and(data.run_id == id_subjs(i),data.pe_actual_corr ~= 0)),...
    data.norm_subjest(and(data.run_id == id_subjs(i),data.pe_actual_corr ~= 0)),...
    data.pe_sign(and(data.run_id == id_subjs(i),data.pe_actual_corr ~= 0)),...
    'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
    ,'reward_unc','subj_est_unc','pe_sign'});

[~,~,~,~,lm] = linear_fit(tbl,mdl,pred_vars,resp_var, ...
    cat_vars,num_vars,weight_y_n);

% INITIALISE TILE
ax7 = nexttile(10,[1,3]);
box(ax7,"off");

% PLOT
hold on
h = plotInteraction(lm,'contrast_diff','pe','predictions');
h(3).Color = low_PU;
h(2).Color = mid_PU;
h(1).Color = high_PU;
xlabel('Prediction error')
ylabel('Update')
title('')
adjust_figprops(ax7,font_name,font_size,line_width);
l = legend('Contrast difference','0','0.5','1','Location','best');
l.EdgeColor = 'none';
l.Color = 'none';
l.ItemTokenSize = [7 7];
box off

% INITIALISE TILE
ax8 = nexttile(16,[1,3]);
box(ax8,"off");

% PLOT
h = plotInteraction(lm,'pe_sign','pe','predictions');
h(2).Color = 'k';
h(1).Color = light_gray;
xlabel('Prediction error')
ylabel('')
title('')
adjust_figprops(ax8,font_name,font_size,line_width);
l = legend('Sign of PE','Negative','Positive','Location','best');
l.EdgeColor = 'none';
l.Color = 'none';
l.ItemTokenSize = [7 7];
box off
%% ADD SUBPLOT LABELS

ax1_pos = ax1.Position;
adjust_x = - 0.06;
adjust_y = ax1_pos(4);

all_axes = [ax1,ax2,ax3,ax6,ax7,ax8];
subplot_labels = {'a','b','c','d','e','f'};
for i = 1:6
    [label_x,label_y] = change_plotlabel(all_axes(i),adjust_x,adjust_y);
    annotation("textbox",[label_x label_y .05 .05],'String', ...
        subplot_labels{i},'FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment','center')
end
%% SAVE FIGURE

fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, 'figure4_absolute6.png', '-dpng', '-r600') 