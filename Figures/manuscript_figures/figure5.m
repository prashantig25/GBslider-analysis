%% INITIALISE GENERAL PLOT VARS
colors_manuscript; % colors for plot
linewidth_line = 1.5; % linewidth for plot lines
line_width = 0.5; % linewidth for axes
font_name = 'Arial'; % font name
font_size = 6; % font size
xlim_vals = [0 1]; % x limits
ylim_vals = [0 1.2]; % y limits
%% FIT LEARNING RATE ANALYSIS COEFFICIENTS TO MEAN PERFORMANCE

% INITIALISE VARS
num_subjs = 99; % number of subjects
num_vars = 7; % number of variables in model 
ecoperf = NaN(num_subjs,1); % initialised array for economic performance

% INITIALISE MODELS
mdl = 'perf ~ pe + pe__condiff + pe__pesign';
mdl_pe = 'perf ~ pe__condiff + pe__pesign';
mdl_pe_condiff = 'perf ~ pe + pe__pesign';
mdl_pe_pesign = 'perf ~ pe + pe__condiff';

% LOAD DATA
load('betas_signed.mat','betas_all'); % betas from lr analysis model
data = readtable("data_subjs.xlsx"); % get choice performance
id_subjs = unique(data.run_id); % subject IDs

% GET ECONOMIC PERFORMANCE
for i = 1:num_subjs
    ecoperf(i,:) = nanmean(data.choice_corr(data.run_id == id_subjs(i)));
end

% NORMALISE ALL COEFFICIENTS
norm_betas_all = betas_all;
for i = 1:num_vars
    norm_betas_all(:,i) = normalise_zero_one(betas_all(:,i),NaN(height(betas_all),1));
end
norm_ecoperf = normalise_zero_one(ecoperf,NaN(height(betas_all),1));
data = [norm_betas_all,norm_ecoperf];

% INITIALISE VARIABLES TO FIT MODEL
var_names = {'pe','pe__condiff','pe__salience','pe__congruence','pe__rewunc','pe__subjest','pe__pesign','perf'};
data_tbl = array2table(data, 'VariableNames', var_names);
num_vars = 3; 
num_vars_partial = 2;
pred_vars = {'pe','pe__condiff','pe__salience','pe__congruence','pe__rewunc','pe__subjest','pe__pesign'};
resp_var = 'perf';
cat_vars = '';
res_subjs = [];

% FIT FULL AND PARTIAL MODELS
[~,~,~,~,lm] = linear_fit(data_tbl,mdl,pred_vars,resp_var, ...
    cat_vars,num_vars,0);
[~,~,~,~,lm_pe] = linear_fit(data_tbl,mdl_pe,pred_vars,resp_var, ...
    cat_vars,num_vars_partial,0);
[~,~,~,~,lm_pe_condiff] = linear_fit(data_tbl,mdl_pe_condiff,pred_vars,resp_var, ...
    cat_vars,num_vars_partial,0);
[betas,rsquared,residuals,coeffs_name,lm_pe_pesign] = linear_fit(data_tbl,mdl_pe_pesign,pred_vars,resp_var, ...
    cat_vars,num_vars_partial,0);

% COMPUTE PARTIAL R-SQAURE
rsq_full = lm.Rsquared.Adjusted; % r-square for full model
rsq_partial = [lm_pe.Rsquared.Adjusted; lm_pe_condiff.Rsquared.Adjusted; 
    lm_pe_pesign.Rsquared.Adjusted]; % r-square for partial models
partial_rsq = NaN(3,1);
for i = 1:3
    partial_rsq(i,1) = compute_partialrsq(rsq_partial(i),rsq_full);
end
%% INITIALISE TILE LAYOUT

figure
set(gcf,'Position',[100 100 600 200])
t = tiledlayout(1,3);
t.Padding = 'compact';
ax1 = nexttile(1,[1,1]);
ax2 = nexttile(2,[1,1]);
ax3 = nexttile(3,[1,1]);

%% ADDED VARIABLE PLOTS

variables = [2,3,4];
axes_variables = [ax1,ax2,ax3];
xlabels_variables = ["Adjusted fixed LR","Adjusted BS adapted LR","Adjusted confirmation bias LR"];

for v = 1:length(variables)
    hold on
    p = plotAdded(axes_variables(v),lm,[variables(v)],'Marker','o','MarkerSize', ...
        3,'MarkerFaceColor',gray_dots, ...
            'MarkerEdgeColor',dots_edges);
    p(1).Color = reg_color;
    p(2).Color = reg_color;
    p(3).Color = reg_color;
    p(2).LineWidth = linewidth_line;
    legend(axes_variables(v),'off')
    xlabel(axes_variables(v),xlabels_variables(v))
    ylabel(axes_variables(v),'Adjusted economic performance')
    title(axes_variables(v),strcat('Partial R^2 = ',{' '},num2str(sprintf('%.2f', partial_rsq(v)))), ...
    'FontWeight','normal','Interpreter','tex')
    adjust_figprops(axes_variables(v),font_name,font_size,line_width,xlim_vals,ylim_vals)
    box(axes_variables(v),"off")
end

%% ADD SUBPLOT LABELS

ax1_pos = ax1.Position;
adjust_x = -0.055;
adjust_y = ax1_pos(4) + 0.052;
[label_x,label_y] = change_plotlabel(ax1,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'a','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(ax2,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'b','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(ax3,adjust_x,adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'c','FontSize',12,'LineStyle','none','HorizontalAlignment','center')
%% SAVE FIGURE

fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, 'figure4_int1.png', '-dpng', '-r600') 