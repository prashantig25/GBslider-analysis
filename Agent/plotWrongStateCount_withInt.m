%% ========================================================================
%  Script: Count Trials Where the Agent's Believed State Is Wrong, Across
%  Sigma (With Integration, eval_ana = 0)
%  ------------------------------------------------------------------------
%  Loads Agent/script_sims.m's with-integration, eval_ana=0 long-block
%  simulations (data_agent_longBlocks_withInt_evalAna0_increasedRange.txt
%  -- o_t = c_t exactly, with the fixed state_0/state_1 marginalization --
%  see conversation) and, for each sigma value, counts -- per individual
%  simulation -- how many of its num_trials_long trials the agent's
%  believed state (whichever of state_0/state_1 is larger) disagrees with
%  the task's true state. Plots one bar chart: mean count per sigma (SEM
%  error bars) with individual-simulation data points overlaid (jittered
%  on x).
%  ------------------------------------------------------------------------
%  Design notes:
%  - Pooled across both mu conditions (choice_cond) -- the state-wrongness
%    count is a purely perceptual quantity (believed vs true state) that
%    doesn't depend on the reward-contingency condition, since both
%    conditions share the identical state/contrast generative process
%    (only the reward mapping differs). Pooling doubles the number of
%    individual-simulation data points per sigma (num_sims_long x 2).
%  - Each sigma's subset is reshaped into a [num_trials_long x num_sims]
%    matrix (same approach as plotLongBlocks_integrationComparison.m's
%    local_trial_values) to get a per-simulation wrong-trial count,
%    robust to any row-order quirks in how run_simulations accumulates
%    blocks.
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

data_withInt = readtable(fullfile(data_dir, 'data_agent_longBlocks_withInt_evalAna0_increasedRange.txt'));
num_trials_long = max(data_withInt.trials);
sigma_values = unique(data_withInt.sigma)';

% STYLING
font_size = 9;
font_name = 'Arial';
bar_color = [0.8500, 0.3250, 0.0980]; % matches integration_color elsewhere
dot_color = [0.3, 0.3, 0.3];
jitter_width = 0.15;

wrong_counts_by_sigma = cell(length(sigma_values), 1);
for s = 1:length(sigma_values)
    sigma_val = sigma_values(s);
    subset = data_withInt(abs(data_withInt.sigma - sigma_val) < 1e-9, :); % pooled across both choice_cond values
    num_sims = height(subset) / num_trials_long;
    believed_state = double(subset.state_1 > subset.state_0); % 1 if the agent believed state=1 was more likely, else 0
    wrong = believed_state ~= subset.state;
    wrong_matrix = reshape(wrong, num_trials_long, num_sims); % trials x sims
    wrong_counts_by_sigma{s} = sum(wrong_matrix, 1)'; % sims x 1 -- wrong-trial count per simulation
end

mean_wrong = cellfun(@mean, wrong_counts_by_sigma);
sem_wrong = cellfun(@(x) std(x) / sqrt(length(x)), wrong_counts_by_sigma);

figure('Position', [100, 100, 600, 450]);
hold on;
bar(1:length(sigma_values), mean_wrong, 'FaceColor', bar_color, 'FaceAlpha', 0.6, 'EdgeColor', 'none');
errorbar(1:length(sigma_values), mean_wrong, sem_wrong, 'k', 'LineStyle', 'none', 'LineWidth', 1.2, 'CapSize', 8);
for s = 1:length(sigma_values)
    n = length(wrong_counts_by_sigma{s});
    x_jitter = s + (rand(n, 1) - 0.5) * 2 * jitter_width;
    scatter(x_jitter, wrong_counts_by_sigma{s}, 12, dot_color, 'filled', 'MarkerFaceAlpha', 0.4);
end

set(gca, 'XTick', 1:length(sigma_values), 'XTickLabel', arrayfun(@(x) sprintf('%.2f', x), sigma_values, 'UniformOutput', false));
xlabel('Sigma');
ylabel(sprintf('Trials with wrong believed state (out of %d)', num_trials_long));
title('Perceptual State Errors Across Sigma (With Integration, eval\_ana = 0, o_t = c_t)', 'FontWeight', 'bold');
set(gca, 'FontName', font_name, 'FontSize', font_size, 'LineWidth', 1, 'Color', 'none');
box off;

exportgraphics(gcf, fullfile(save_fig_dir, 'wrongStateCount_withInt_evalAna0.pdf'), 'ContentType', 'vector');
