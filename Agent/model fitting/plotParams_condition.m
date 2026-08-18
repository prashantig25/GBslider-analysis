clc; clearvars;
% Load parameter files (edit these names if needed)
% basicRL = importdata('params_basicRL_NOintegral.mat'); % fields: alpha, kappa
% RLSigma = importdata('params_RLsigma_NOintegral.mat'); % fields: alpha, kappa, sigma
% RLSigma_confirmBias = importdata("params_RLsigma_confirmBias_NOintegral.mat"); % fields: alpha, kappa, sigma
% bayesianAgent = importdata('params_bayesianAgent_NOintegral_evalana.mat'); % fields: kappa, sigma
% bayesianAgent_confirmBias = importdata('params_bayesianAgent_confirmBias_NOintegral_evalana.mat'); % fields: kappa, sigma
% import parameters
basicRL = importdata("params_basicRL_Agent.mat");
RLSigma = importdata("params_RLsigma_modelAgent.mat");
RLSigma_confirmBias = importdata("params_RLsigma_confirmBias_modelAgent.mat");
bayesianAgent = importdata("params_bayesianAgent_Agent.mat");
bayesianAgent_confirmBias = importdata("params_bayesianAgent_confirmBias_Agentfixed.mat");
basicRL_confirmBias = importdata("params_basicRL_confirmBias_modelAgent.mat");

% Collect parameter arrays (NaN pad where not present)
n = length(basicRL.alpha); % number of subjects

% Parameters to plot: alpha, kappa, sigma (use NaN if not present)
alpha_all = [basicRL.alpha, RLSigma.alpha, NaN(n,1), NaN(n,1), NaN(n,1), NaN(n,1)];
kappa_all = [basicRL.kappa, RLSigma.kappa, RLSigma_confirmBias.kappa, bayesianAgent.kappa, bayesianAgent_confirmBias.kappa, basicRL_confirmBias.kappa];
sigma_all = [basicRL.sigma, RLSigma.sigma, RLSigma_confirmBias.sigma, bayesianAgent.sigma, bayesianAgent_confirmBias.sigma, basicRL_confirmBias.sigma];
confirmBias_all = [NaN(n,1), NaN(n,1),RLSigma_confirmBias.confirmBias,NaN(n,1), bayesianAgent_confirmBias.confirmBias, basicRL_confirmBias.confirmBias];

% Stack all parameters: columns = [basicRL, RLsigma, PWRL, bayesianAgent]
param_names = {'alpha','kappa','sigma','confirmBias'};
model_names = {'basicRL','RL+EstSens','RL+EstSense+ConfirmBias','BayesianAgent','BayesianAgent + confirmBias','basicRL + CB'};

% Define colors ONCE for each model, using lines(5)
all_facecolors = lines(6); % 5 models

% Create a single figure with subplots
figure('Position', [100, 100, 1200, 800]); % Adjust figure size as needed

% Prepare data for each parameter
for p = 1:length(param_names)
    switch param_names{p}
        case 'alpha'
            y = alpha_all;
        case 'kappa'
            y = kappa_all;
        case 'sigma'
            y = sigma_all;
        case 'confirmBias'
            y = confirmBias_all;
        case 'beta'
            y = beta_all;
    end
    
    % Remove columns that are all NaN (e.g., alpha for BayesianAgent)
    valid_cols = ~all(isnan(y),1);
    y = y(:,valid_cols);
    this_model_names = model_names(valid_cols);
    
    % Select colors for the valid models, keeping the order
    facecolors = all_facecolors(valid_cols, :);
    mean_all = nanmean(y);
    SEM_all = nanstd(y)./sqrt(sum(~isnan(y)));
    
    x_groups = 1; % just one group, all models
    bars = size(y,2);
    legend_names = this_model_names;
    xticks = 1;
    xticklabs = {param_names{p}};
    title_name = [param_names{p} ' parameter across models'];
    xlabelname = '';
    ylabelname = param_names{p};
    disp_pval = 0; % don't display significance stars
    scatter_dots = 1;
    dot_size = 20;
    plot_err = 1;
    fontsize = 12;
    linewidth = 1;
    fontname = 'Arial';
    disp_legend = 1; % Show legend for each subplot with its relevant models
    
    % Create subplot
    subplot(2, 2, p); % 2x2 grid for 4 parameters
    
    bar_plots_pval(y,mean_all,SEM_all,n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
        xlabelname,ylabelname,disp_pval,scatter_dots,dot_size,plot_err,fontsize,linewidth,fontname,disp_legend,facecolors);
end

% Add an overall title to the figure
sgtitle('Both condition', 'FontSize', 16, 'FontWeight', 'normal');

%%
 
AICBIC_table = importdata("AICBIC_Agentfixed.mat");
% Color scheme
all_facecolors = lines(6); % 5 models
% Model names for legend and x-axis labels
model_names = {'basicRL','RLsigma','RLSigma + Confirm bias','BayesianAgent','BayesianAgent + Confirm bias','basicRL + CB'};

% Extract AIC, BIC, and delta_BIC from AICBIC_table
AIC = [AICBIC_table.AIC_basicRL, AICBIC_table.AIC_RLsigma, AICBIC_table.AIC_RLSigma_confirmBias, AICBIC_table.AIC_BayesianAgent, AICBIC_table.AIC_BayesianAgent_confirmBias, AICBIC_table.AIC_basicRL_confirmBias];
BIC = [AICBIC_table.BIC_basicRL, AICBIC_table.BIC_RLsigma, AICBIC_table.BIC_RLSigma_confirmBias, AICBIC_table.BIC_BayesianAgent, AICBIC_table.BIC_BayesianAgent_confirmBias, AICBIC_table.BIC_basicRL_confirmBias];
delta_BIC = [AICBIC_table.delta_BIC_basicRL, AICBIC_table.delta_BIC_RLsigma, AICBIC_table.delta_BIC_RLsigma_confirmBias, AICBIC_table.delta_BIC_BayesianAgent, AICBIC_table.delta_BIC_BayesianAgent_confirmBias, AICBIC_table.delta_BIC_basicRL_confirmBias];

% Get number of subjects
numSubjs = size(AIC, 1);

% Calculate sum AIC and BIC for each model
sum_AIC = nansum(AIC, 1);
sum_BIC = nansum(BIC, 1);

% Create a single figure with subplots
figure('Position', [100, 100, 1400, 500]); % Adjust figure size for 3 subplots in a row

% Subplot 1: Sum AIC
subplot(1, 3, 1);
h_aic = bar(1:6, sum_AIC, 'FaceColor', 'flat');
% set(gca, 'XTickLabel', model_names);
title('Sum of AIC across all subjects');
ylabel('Sum AIC');
xlabel('Model');
box off;
set(gca, 'FontSize', 12, 'FontName', 'Arial', 'LineWidth', 1);
% Add individual colors and set face alpha
for i = 1:6
    h_aic.CData(i,:) = all_facecolors(i,:);
end
h_aic.FaceAlpha = 0.5;
% Rotate x-axis labels for better readability
xtickangle(45);

% Subplot 2: Sum BIC
subplot(1, 3, 2);
h_bic = bar(1:6, sum_BIC, 'FaceColor', 'flat');
% set(gca, 'XTickLabel', model_names);
title('Sum of BIC across all subjects');
ylabel('Sum BIC');
xlabel('Model');
box off;
set(gca, 'FontSize', 12, 'FontName', 'Arial', 'LineWidth', 1);
% Add individual colors and set face alpha
for i = 1:5
    h_bic.CData(i,:) = all_facecolors(i,:);
end
h_bic.FaceAlpha = 0.5;
% Rotate x-axis labels for better readability
xtickangle(45);

% Subplot 3: Delta BIC using bar_plots_pval function
subplot(1, 3, 3);
% Calculate mean and SEM for delta BIC
mean_delta_BIC = nanmean(delta_BIC, 1);
SEM_delta_BIC = nanstd(delta_BIC, 1) ./ sqrt(sum(~isnan(delta_BIC), 1));

% Prepare parameters for bar_plots_pval function
y_delta = delta_BIC;
mean_all_delta = mean_delta_BIC;
SEM_all_delta = SEM_delta_BIC;
n_subjects = size(delta_BIC, 1);
x_groups = 1;
bars = 6;
legend_names = model_names;
xticks = 1;
xticklabs = {'Delta BIC'};
title_name = 'Mean Delta BIC across models';
xlabelname = '';
ylabelname = 'Delta BIC';
disp_pval = 0;
scatter_dots = 1;
dot_size = 20;
plot_err = 1;
fontsize = 12;
linewidth = 1;
fontname = 'Arial';
disp_legend = 1;

% Create the delta BIC plot using your custom function
bar_plots_pval(y_delta, mean_all_delta, SEM_all_delta, n_subjects, x_groups, bars, ...
    legend_names, xticks, xticklabs, title_name, xlabelname, ylabelname, ...
    disp_pval, scatter_dots, dot_size, plot_err, fontsize, linewidth, fontname, ...
    disp_legend, all_facecolors);

% Add an overall title to the figure
sgtitle('Model Comparison: Simulated with RL Sigma', 'FontSize', 16, 'FontWeight', 'normal');

% Adjust subplot spacing for better appearance
set(gcf, 'Units', 'normalized');
set(gcf, 'Position', [0.1, 0.1, 0.8, 0.4]);