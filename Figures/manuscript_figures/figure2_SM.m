clc
clearvars
colors_manuscript; % colors for the plot
line_width = 0.5; % line width for axes
font_size = 6; % font size
font_name = 'Arial'; % font name
linewidth_plot = 1; % line width for plot lines

% INITIALISE VARS
load("betas_signed.mat","betas_all"); betas_signed = betas_all; % participant betas from signed analysis
load("betas_agent.mat","betas_all"); betas_agent = betas_all; % participant betas from signed analysis
load("p_vals.mat","p_vals"); % p-vals from ttest
betas_agent(:,7) = betas_agent(:,4);
betas_agent(:,4) = NaN(length(id_subjs),1);
num_subjs = length(id_subjs);
selected_regressors = [1,2,7,6,5,3,4];
dot_size = 10;
%% INITIALISE TILE LAYOUT

figure
set(gcf,'Position',[100 100 700 300])
t = tiledlayout(1,4);
t.TileSpacing = 'compact';
t.Padding = 'compact';

%% PLOT ALL BETA COEFFICIENTS

% TILE
ax2 = nexttile(1,[1,2]);

% GET MEAN AND SEM FOR BETAS
[mean_avg,mean_sd,coeffs_subjs] = prepare_betas(betas_signed,selected_regressors,num_subjs);

% GET STARS FOR CORRESPONDING REGRESSOR'S P-VALUES
bar_labels = {'*','*','*','*','*','*','*'};
p_vals = p_vals(:,selected_regressors);
pstars = pvals_stars(p_vals,selected_regressors,bar_labels);

% OTHER FIGURE PROPERTIES
xticks = [1:length(selected_regressors)];
row1 = {'Fixed LR' 'BS adapted LR' 'EU adapted LR' 'Confirmation' 'Risk adapted LR' 'Salience' 'Congruence'};
row2 = {'' '' '' 'bias adapted LR' '' 'adapted LR' 'adapted LR'};
labelArray = [row1; row2;]; 
tickLabels = strtrim(sprintf('%s\\newline%s\n', labelArray{:}));
xticklabs = tickLabels;
title_name = {''};
xlabelname = {'Regressor'};
ylabelname = {'Mean beta coefficients'};
colors_name = barface_green;
y_label = repelem(1.3,1,length(selected_regressors));
disp_pval = 1;

% PLOT
hold on
bar_plots_pval(coeffs_subjs,mean_avg,mean_sd,num_subjs, ...
    length(selected_regressors),1,{'Empirical data','Normative agent'}, ...
    xticks,xticklabs,title_name,xlabelname, ...
    ylabelname,disp_pval,1,10,1,font_size,line_width,font_name,1,colors_name,bar_labels, ...
    y_label,mean(betas_agent,1),[0.5,7.5]) 
set(gca,'Color','none')

%% PLOT R-SQUARED VALUES

% CHANGE TILE POSITION
ax4 = nexttile(4,[1,1]);
position_change = [0.1, 0, -0.1, 0];
new_pos = change_position(ax4,position_change);
ax4_new = axes('Units', 'Normalized', 'Position', new_pos);
box(ax4_new, 'off');
delete(ax4);

% PLOT
title_name = {''};
bar_plots_pval(rsquared,mean(rsquared),std(rsquared)./sqrt(num_subjs),num_subjs, ...
    1,1,{''},1,{''},title_name,{''},{''},0,1,dot_size,1,font_size,line_width,font_name,0,barface_green) 
hold on
xlabel('R-squared')
set(gca,'Color','none')

%%
data_subjs = readtable("data_subjs.xlsx");

mu = data_subjs.mu;
data_subjs = data_subjs;
flipped_mu = mu;
incorr_mu = mu;
correct_choice = data_subjs.choice_corr;
mu_pe = mu;
agent = 0;
[flipped_mu,incorr_mu,mu_pe] = get_rew_mu(mu,data_subjs,flipped_mu,incorr_mu,correct_choice,mu_pe,agent);

obtained = data_subjs.correct;
expected = mu_pe; %data_subjs.actual_mu_corr;
flipped_mu = flipped_mu; %data_subjs.flipped_mu;
incorr_mu = incorr_mu; %data_subjs.incorr_mu;
choice_corr = data_subjs.choice_corr;
[pe,up] = get_pe_up(obtained,expected,flipped_mu,incorr_mu,choice_corr);
data_subjs.pe_actual_corr = pe;
data_subjs.up_actual_corr = up;

data_subjs.pe_actual_corr(data_subjs.trials == 1) = 0;
data_subjs.up_actual_corr(data_subjs.trials == 1) = 0;

subj_est = compute_subjest(flipped_mu);
data_subjs.subj_est_unc = subj_est;

all_cond = 0;
if all_cond == 0
    data_subjs = data_subjs(data_subjs.choice_cond ~= 3,:);
end

y = data_subjs.up_actual_corr(data_subjs.pe_actual_corr ~= 0);
y_hat = posterior_up_subjs;

pos = ax1.Position;
new_pos = pos + [0, 0, 0.1, 0]; % [left, bottom, width, height]

% Create a new set of axes with the desired position
ax1_new = axes('Units', 'Normalized', 'Position', new_pos);
box(ax1_new, 'off');
delete(ax1);

hold on
h = histfit(data_subjs.up_actual_corr(data_subjs.pe_actual_corr ~= 0));
hold on
h1 = histfit(y_hat);

h1(1).FaceAlpha = 0.3;
h(1).FaceAlpha = 0.7;

h1(2).Color = [185, 170, 203]./255;%corr_green;
h(2).Color = [37, 50, 55]/255;

h1(1).EdgeColor = [185, 170, 203]./255; %corr_green;
h(1).EdgeColor = [37, 50, 55]/255;

h1(1).FaceColor = [185, 170, 203]./255; %corr_green;
h(1).FaceColor = [37, 50, 55]/255;
set(gca,'Color','none','FontName','Arial','FontSize',6)
set(gca,'LineWidth',0.5)
l = legend('Empirical updates','','Regression fits','','EdgeColor','none','Color','none');
l.ItemTokenSize = [7 7];
% title('Updates','FontWeight','normal')
xlabel('Updates')
ylim([-20,4000])
%%
annotation("textbox",[0.01 .93 .05 .07],'LineWidth',1.5,'String', ...
    'a','FontSize',12,'LineStyle','none','HorizontalAlignment','center')
annotation("textbox",[0.46 .93 .05 .07],'LineWidth',1.5,'String', ...
    'b','FontSize',12,'LineStyle','none','HorizontalAlignment','center')
annotation("textbox",[0.82 .93 .05 .07],'LineWidth',1.5,'String', ...
    'c','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

%%
fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, 'figure2_SM.png', '-dpng', '-r600') 