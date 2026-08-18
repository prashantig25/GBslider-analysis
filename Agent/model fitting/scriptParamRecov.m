%% Parameter Recovery Analysis Script
% This script runs parameter recovery analysis for all RL models
% and creates comprehensive visualization and summary statistics

clc; clear; close all;

%% Initialize Parameter Recovery Analysis
% Create parameter recovery object with desired settings
pr = ParameterRecovery(50, 20, 500); % 50 parameter sets, 20 starting points, 500 trials

%% Run Parameter Recovery for All Models
fprintf('Starting comprehensive parameter recovery analysis...\n');
fprintf('This may take several minutes to complete.\n\n');

% Run recovery for each model
tic;
pr.run_basicRL_recovery();
fprintf('Basic RL completed in %.2f seconds\n', toc);

tic;
pr.run_RLsigma_recovery();
fprintf('RL+Sigma completed in %.2f seconds\n', toc);

tic;
pr.run_PWRL_recovery();
fprintf('PWRL completed in %.2f seconds\n', toc);

tic;
pr.run_bayesianAgent_recovery();
fprintf('Bayesian Agent completed in %.2f seconds\n', toc);

%% Generate Visualizations
fprintf('\nGenerating visualizations...\n');

% Plot results for all models
pr.plot_all_models();

%% Compute and Display Summary Statistics
summary_stats = pr.compute_summary_statistics();

fprintf('\n=== PARAMETER RECOVERY SUMMARY ===\n');
model_names = fieldnames(summary_stats);

for i = 1:length(model_names)
    model = model_names{i};
    fprintf('\n%s:\n', pr.results.(model).model_name);
    fprintf('%-10s %10s %10s %10s %10s\n', 'Parameter', 'Corr', 'RMSE', 'Bias', 'N');
    fprintf('%-10s %10s %10s %10s %10s\n', '---------', '----', '----', '----', '-');
    
    param_names = fieldnames(summary_stats.(model));
    for j = 1:length(param_names)
        param = param_names{j};
        stats = summary_stats.(model).(param);
        fprintf('%-10s %10.3f %10.4f %10.4f %10d\n', ...
            param, stats.correlation, stats.rmse, stats.bias, stats.n_valid);
    end
end

%% Create Comprehensive Comparison Plot
figure('Position', [50, 50, 1200, 800]);

model_names = fieldnames(pr.results);
n_models = length(model_names);
max_params = max(cellfun(@(x) length(pr.results.(x).param_names), model_names));

plot_idx = 1;
for i = 1:n_models
    result = pr.results.(model_names{i});
    n_params = length(result.param_names);
    
    for j = 1:n_params
        subplot(n_models, max_params, plot_idx);
        
        % Create scatter plot with correlation
        scatter(result.true_params(:, j), result.best_recovered(:, j), ...
            30, 'filled', 'MarkerFaceAlpha', 0.7);
        
        hold on;
        
        % Add identity line
        min_val = min([result.true_params(:, j); result.best_recovered(:, j)]);
        max_val = max([result.true_params(:, j); result.best_recovered(:, j)]);
        plot([min_val, max_val], [min_val, max_val], 'r--', 'LineWidth', 1.5);
        
        % Calculate and display correlation
        r = corr(result.true_params(:, j), result.best_recovered(:, j), 'rows', 'complete');
        
        xlabel(sprintf('True %s', result.param_names{j}));
        ylabel(sprintf('Recovered %s', result.param_names{j}));
        title(sprintf('%s: %s (r=%.3f)', result.model_name, result.param_names{j}, r));
        grid on;
        
        plot_idx = plot_idx + 1;
    end
    
    % Fill empty subplots if needed
    while mod(plot_idx - 1, max_params) ~= 0
        subplot(n_models, max_params, plot_idx);
        axis off;
        plot_idx = plot_idx + 1;
    end
end

sgtitle('Parameter Recovery Comparison Across All Models', 'FontSize', 16, 'FontWeight', 'bold');

%% Save Results
fprintf('\nSaving results...\n');
save('parameter_recovery_results.mat', 'pr', 'summary_stats');

fprintf('Parameter recovery analysis completed!\n');
fprintf('Results saved to parameter_recovery_results.mat\n');
