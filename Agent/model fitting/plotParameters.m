clc; clearvars; 

% Load parameter files (edit these names if needed)
% basicRL = importdata('params_basicRL_NOintegral.mat');      % fields: alpha, kappa
% RLSigma = importdata('params_RLsigma_NOintegral.mat');      % fields: alpha, kappa, sigma
% RLSigma_confirmBias = importdata("params_RLsigma_confirmBias_NOintegral.mat");            % fields: alpha, kappa, sigma
% bayesianAgent = importdata('params_bayesianAgent_NOintegral_evalana.mat'); % fields: kappa, sigma
% bayesianAgent_confirmBias = importdata('params_bayesianAgent_confirmBias_NOintegral_evalana.mat'); % fields: kappa, sigma

% import parameters
basicRL = importdata("params_basicRL_RLSigmafixed.mat");
basicRL_CB = importdata("params_basicRL_confirmBias_RLSigmafixed.mat");
RLSigma = importdata("params_RLsigma_RLSigmafixed.mat");
RLSigma_confirmBias = importdata("params_RLsigma_confirmBias_RLSigmafixed.mat");
bayesianAgent = importdata("params_bayesianAgent_RLSigmafixed.mat");
bayesianAgent_confirmBias = importdata("params_bayesianAgent_confirmBias_RLSigmafixed.mat");

% Collect parameter arrays (NaN pad where not present)
n = length(basicRL.alpha); % number of subjects

% Parameters to plot: alpha, kappa, sigma (use NaN if not present)
% alpha_all = [basicRL.alpha, RLSigma.alpha, PWRL.alpha, NaN(n,1)];
% kappa_all = [basicRL.kappa, RLSigma.kappa, PWRL.kappa, bayesianAgent.kappa];
% sigma_all = [NaN(n,1), RLSigma.sigma, PWRL.sigma, bayesianAgent.sigma];

% Parameters to plot: alpha, kappa, sigma (use NaN if not present)
alpha_all = [basicRL.alpha, RLSigma.alpha,  RLSigma_confirmBias.alpha, NaN(n,1), NaN(n,1), NaN(n,1)];
kappa_all = [basicRL.kappa, RLSigma.kappa, RLSigma_confirmBias.kappa, bayesianAgent.kappa, bayesianAgent_confirmBias.kappa, basicRL_CB.kappa];
sigma_all = [basicRL.sigma, RLSigma.sigma, RLSigma_confirmBias.sigma, bayesianAgent.sigma, bayesianAgent_confirmBias.sigma, basicRL_CB.sigma];
confirmBias_all = [NaN(n,1), NaN(n,1),  RLSigma_confirmBias.confirmBias, RLSigma_confirmBias.noconfirmBias, bayesianAgent_confirmBias.confirmBias, basicRL_CB.confirmBias];

% Stack all parameters: columns = [basicRL, RLsigma, PWRL, bayesianAgent]
param_names = {'alpha','kappa','sigma','confirmBias'};
model_names = {'basicRL','RL+EstSens','RL+EstSense+ConfirmBias','BayesianAgent','BayesianAgent + confirmBias','basicRL + CB'};

% Define colors ONCE for each model, using lines(4)
all_facecolors = lines(6); % 4 models

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
    disp_legend = 1;

    figure
    bar_plots_pval(y,mean_all,SEM_all,n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
        xlabelname,ylabelname,disp_pval,scatter_dots,dot_size,plot_err,fontsize,linewidth,fontname,disp_legend,facecolors);
end

%%

AICBIC_table = importdata("AICBIC_modelRLSigmafixed.mat");
% Color scheme
all_facecolors = lines(5); % 4 models

% Model names for legend and x-axis labels
% model_names = {'basicRL','RLsigma','PWRL','BayesianAgent'};
model_names = {'basicRL','RLsigma','RLSigma + Confirm bias','BayesianAgent','BayesianAgent + Confirm bias'};

% Extract AIC, BIC, and delta_BIC from AICBIC_table
AIC = [AICBIC_table.AIC_basicRL, AICBIC_table.AIC_RLsigma, AICBIC_table.AIC_RLSigma_confirmBias, AICBIC_table.AIC_BayesianAgent, AICBIC_table.AIC_BayesianAgent_confirmBias];
BIC = [AICBIC_table.BIC_basicRL, AICBIC_table.BIC_RLsigma, AICBIC_table.BIC_RLSigma_confirmBias, AICBIC_table.BIC_BayesianAgent, AICBIC_table.AIC_BayesianAgent_confirmBias];
delta_BIC = [AICBIC_table.delta_BIC_basicRL, AICBIC_table.delta_BIC_RLsigma, AICBIC_table.delta_BIC_RLsigma_confirmBias, AICBIC_table.delta_BIC_BayesianAgent, AICBIC_table.delta_BIC_BayesianAgent_confirmBias];

% Get number of subjects
numSubjs = size(AIC, 1);

% Calculate sum AIC and BIC for each model
sum_AIC = nansum(AIC, 1);
sum_BIC = nansum(BIC, 1);

% Figure 1: Sum AIC with face alpha adjustment
figure
h_aic = bar(1:5, sum_AIC, 'FaceColor', 'flat');
set(gca, 'XTickLabel', model_names);
title('Sum of AIC across all subjects');
ylabel('Sum AIC');
xlabel('Model');
box off;
set(gca, 'FontSize', 12, 'FontName', 'Arial', 'LineWidth', 1);

% Add individual colors and set face alpha
for i = 1:5
    h_aic.CData(i,:) = all_facecolors(i,:);
end
h_aic.FaceAlpha = 0.5;

% Figure 2: Sum BIC with face alpha adjustment
figure
h_bic = bar(1:5, sum_BIC, 'FaceColor', 'flat');
set(gca, 'XTickLabel', model_names);
title('Sum of BIC across all subjects');
ylabel('Sum BIC');
xlabel('Model')
box off;
set(gca, 'FontSize', 12, 'FontName', 'Arial', 'LineWidth', 1);

% Add individual colors and set face alpha
for i = 1:5
    h_bic.CData(i,:) = all_facecolors(i,:);
end
h_bic.FaceAlpha = 0.5;


%% 

% Calculate mean and SEM for delta BIC
mean_delta_BIC = nanmean(delta_BIC, 1);
SEM_delta_BIC = nanstd(delta_BIC, 1) ./ sqrt(sum(~isnan(delta_BIC), 1));

% Prepare parameters for bar_plots_pval function
y_delta = delta_BIC;
mean_all_delta = mean_delta_BIC;
SEM_all_delta = SEM_delta_BIC;
n_subjects = size(delta_BIC, 1);
x_groups = 1;
bars = 5;
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
figure
bar_plots_pval(y_delta, mean_all_delta, SEM_all_delta, n_subjects, x_groups, bars, ...
    legend_names, xticks, xticklabs, title_name, xlabelname, ylabelname, ...
    disp_pval, scatter_dots, dot_size, plot_err, fontsize, linewidth, fontname, ...
    disp_legend, all_facecolors);
% ylim([0,400])


