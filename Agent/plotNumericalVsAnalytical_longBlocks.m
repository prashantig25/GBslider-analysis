%% ========================================================================
%  Script: Numerical vs Analytical Belief Update, With Integration, o_t=c_t
%  ------------------------------------------------------------------------
%  Loads Agent/script_sims.m's two long-block, with-integration
%  (task_agent_analysis=1, so o_t=c_t exactly -- see conversation)
%  simulations -- numerical (eval_ana=0, num_trials_long=1000) and
%  analytical (eval_ana=1, num_trials_long=40, capped there because its
%  polynomial belief update becomes numerically unstable past ~50-75
%  trials -- see script_sims.m's eval_ana=1 section header) -- and
%  overlays their mean +/- SEM simulated-mu curves on a shared x-axis
%  (trials 1-1000), one subplot per (condition, sigma). No new simulation
%  is run here.
%  ========================================================================
clc; clearvars; close all;

% PATH STUFF
currentDir = cd;
reqPath = 'GBslider-analysis';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
data_dir = fullfile(desiredPath, 'Data', 'agent simulations');
save_fig_dir = fullfile(desiredPath, 'Agent', 'simfigures');
if ~exist(save_fig_dir, 'dir')
    mkdir(save_fig_dir);
end

% STYLING
font_size = 9;
font_name = 'Arial';
numerical_color = [0, 0.5, 0.5]; % teal
analytical_color = [0.75, 0, 0.3]; % crimson

% LOAD DATA
data_numerical = readtable(fullfile(data_dir, 'data_agent_longBlocks_withInt_evalAna0_increasedRange.txt')); % eval_ana=0, 1000 trials
data_analytical = readtable(fullfile(data_dir, 'data_agent_longBlocks_withInt_evalAna1.txt')); % eval_ana=1, 40 trials

sigma_values = unique(data_numerical.sigma)';
conditions = unique(data_numerical.choice_cond)';
true_mu_by_cond = [0.7, 1]; % Agent/Task.m's taskMu: condition 1 -> 0.7, condition 2 -> 1 (rangeMu = 0)
num_trials_numerical = max(data_numerical.trials); % 1000
num_trials_analytical = max(data_analytical.trials); % 40

%% PLOT
fig = figure('Position', [50, 50, 1500, 500]);
plot_idx = 1;
for c = 1:length(conditions)
    cond = conditions(c);
    for s = 1:length(sigma_values)
        sigma_val = sigma_values(s);
        ax = subplot(length(conditions), length(sigma_values), plot_idx);
        hold on;

        values_numerical = local_trial_values(data_numerical, cond, sigma_val, 'mu', num_trials_numerical);
        mean_numerical = mean(values_numerical, 2);
        sem_numerical = std(values_numerical, 0, 2) ./ sqrt(size(values_numerical, 2));

        values_analytical = local_trial_values(data_analytical, cond, sigma_val, 'mu', num_trials_analytical);
        mean_analytical = mean(values_analytical, 2);
        sem_analytical = std(values_analytical, 0, 2) ./ sqrt(size(values_analytical, 2));

        h_numerical = shadedErrorBar(1:num_trials_numerical, mean_numerical, sem_numerical, {'Color', numerical_color, 'LineWidth', 1.5}, 0);
        h_analytical = shadedErrorBar(1:num_trials_analytical, mean_analytical, sem_analytical, {'Color', analytical_color, 'LineWidth', 1.5}, 0);
        h_trueMu = yline(true_mu_by_cond(c), '--k', 'LineWidth', 1);

        title(sprintf('Cond %d (mu=%.1f), sigma=%.2f', cond, true_mu_by_cond(c), sigma_val), 'FontWeight', 'normal', 'FontSize', font_size);
        if c == length(conditions)
            xlabel('Trial');
        end
        if s == 1
            ylabel('Simulated mu');
        end
        set(ax, 'FontName', font_name, 'FontSize', font_size, 'LineWidth', 1, 'Color', 'none');
        box off;
        ylim([0, 1]);
        xlim([0, num_trials_numerical]); % shared x-axis -- analytical's curve only spans its own 40 trials within this range
        plot_idx = plot_idx + 1;
    end
end
legend([h_numerical.mainLine, h_analytical.mainLine, h_trueMu], ...
    {'Numerical (eval\_ana=0, 1000 trials)', 'Analytical (eval\_ana=1, 40 trials)', 'True mu'}, ...
    'Position', [0.92, 0.45, 0.07, 0.1], 'Box', 'off', 'Color', 'none');
sgtitle('Numerical vs Analytical Belief Update, With Integration (o\_t = c\_t)', 'FontWeight', 'bold');

exportgraphics(fig, fullfile(save_fig_dir, 'simulatedMu_numericalVsAnalytical_withInt.pdf'), 'ContentType', 'vector');

%% ========================================================================
%  Local functions
%  ========================================================================
function values_matrix = local_trial_values(data, cond, sigma_val, varname, num_trials_long)
    subset = data(data.choice_cond == cond & abs(data.sigma - sigma_val) < 1e-9, :);
    num_sims = height(subset) / num_trials_long;
    values_matrix = reshape(subset.(varname), num_trials_long, num_sims); % trials x sims
end
