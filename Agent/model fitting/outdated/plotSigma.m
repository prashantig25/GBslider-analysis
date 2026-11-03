clc; clearvars;

perceptualSigma = importdata("sigma_perceptualChoice.mat");

% Collect parameter arrays (NaN pad where not present)
n = length(perceptualSigma.sigma); % number of subjects

% Parameters to plot: alpha, kappa, sigma (use NaN if not present)
sigma_all = [perceptualSigma.sigma];

% Stack all parameters: columns = [basicRL, RLsigma, PWRL, bayesianAgent]
param_names = {'sigma'};
model_names = {'Perceptual choice model'};

% Define colors ONCE for each model, using lines(5)
all_facecolors = lines(1); % 5 models

% Create a single figure with subplots
figure('Position', [100, 100, 200, 200]); % Adjust figure size as needed

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
    title_name = [' '];
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
    hold on
    
    bar_plots_pval(y,mean_all,SEM_all,n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
        xlabelname,ylabelname,disp_pval,scatter_dots,dot_size,plot_err,fontsize,linewidth,fontname,disp_legend,facecolors);
end

% Add an overall title to the figure
% sgtitle('Perceptual condition', 'FontSize', 16, 'FontWeight', 'normal');
