%% ========================================================================
%  Script: Simulated Mu, With vs Without Integration, eval_ana = 1
%  (Analytical Solution), Across Sigma and True Mu
%  ------------------------------------------------------------------------
%  Loads the long-block, sigma-swept eval_ana=1 (analytical/polynomial
%  belief update) simulations from Agent/script_sims.m's "SIMULATE LONG
%  BLOCKS ACROSS SIGMA VALUES..." (eval_ana = 1) section
%  (data_agent_longBlocks_{noInt,withInt}_evalAna1.txt -- no new
%  simulation is run here) and produces ONE figure with both no-integration
%  and with-integration plotted together in each subplot (different
%  colors), rather than separate figures: individual-simulation thin lines
%  + mean +/- SEM shaded band per series, with the condition's true mu as
%  a dotted reference, in a condition x sigma subplot grid. Saved as a PDF
%  under Agent/simfigures/.
%  ------------------------------------------------------------------------
%  Design notes:
%  - eval_ana=1 uses num_trials_long=50 (not 1000 like the eval_ana=0
%    long-block sections) -- its polynomial belief update becomes
%    numerically unstable past ~50-75 trials (see script_sims.m's eval_ana
%    = 1 section header), so this is the longest block length it reliably
%    survives.
%  - The eval_ana=0 version of this comparison (and the economic-
%    performance figure) is commented out below, not deleted -- restore by
%    uncommenting if needed again.
%  - Each (condition, sigma) combination's num_sims_long replicates are
%    identified by filtering on the sigma/choice_cond columns
%    Agent/script_sims.m added after each run_simulations call, then
%    reshaped into a [num_trials_long x num_sims_long] matrix --
%    robust regardless of any row-order quirks in how run_simulations
%    accumulates blocks (see that function's prepend-based accumulation).
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
line_width_mean = 1.5; % mean-curve line width
line_width_thin = 0.05; % single-simulation trace line width
font_name = 'Arial';
noIntegration_color = [0, 0.4470, 0.7410]; % MATLAB default blue
noIntegration_color_light = noIntegration_color + (1 - noIntegration_color) * 0.75; % lightened (towards white), individual-simulation lines
integration_color = [0.8500, 0.3250, 0.0980]; % MATLAB default orange
integration_color_light = integration_color + (1 - integration_color) * 0.75; % lightened (towards white), individual-simulation lines

eval_ana_val = 1; % analytical solution -- see design notes above

% LOAD DATA
data_noInt = readtable(fullfile(data_dir, 'data_agent_longBlocks_noInt_evalAna1.txt'));
data_withInt = readtable(fullfile(data_dir, 'data_agent_longBlocks_withInt_evalAna1.txt'));

sigma_values = unique(data_noInt.sigma)';
conditions = unique(data_noInt.choice_cond)';
true_mu_by_cond = [0.7, 1]; % Agent/Task.m's taskMu: condition 1 -> 0.7, condition 2 -> 1 (rangeMu = 0)
num_trials_long = max(data_noInt.trials);

%% =================== FIGURE: MU, WITH VS WITHOUT INTEGRATION ===========
fig_mu = figure('Position', [50, 50, 1500, 500]);
plot_idx = 1;
for c = 1:length(conditions)
    cond = conditions(c);
    for s = 1:length(sigma_values)
        sigma_val = sigma_values(s);
        ax = subplot(length(conditions), length(sigma_values), plot_idx);
        hold on;

        values_noInt = local_trial_values(data_noInt, cond, sigma_val, 'mu', num_trials_long);
        mean_noInt = mean(values_noInt, 2);
        sem_noInt = std(values_noInt, 0, 2) ./ sqrt(size(values_noInt, 2));

        values_withInt = local_trial_values(data_withInt, cond, sigma_val, 'mu', num_trials_long);
        mean_withInt = mean(values_withInt, 2);
        sem_withInt = std(values_withInt, 0, 2) ./ sqrt(size(values_withInt, 2));

        % Individual-simulation lines drawn first so the mean+SEM overlays sit on top.
        h_noInt_lines = plot(1:num_trials_long, values_noInt, 'Color', noIntegration_color_light, 'LineWidth', line_width_thin);
        h_withInt_lines = plot(1:num_trials_long, values_withInt, 'Color', integration_color_light, 'LineWidth', line_width_thin);
        h_noInt_mu = shadedErrorBar(1:num_trials_long, mean_noInt, sem_noInt, {'Color', noIntegration_color, 'LineWidth', line_width_mean}, 0);
        h_withInt_mu = shadedErrorBar(1:num_trials_long, mean_withInt, sem_withInt, {'Color', integration_color, 'LineWidth', line_width_mean}, 0);
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
        plot_idx = plot_idx + 1;
    end
end
% Single shared legend using explicit proxy handles (from the last
% subplot) -- see design notes above.
legend([h_noInt_mu.mainLine, h_withInt_mu.mainLine, h_trueMu], {'No Integration', 'Integration', 'True mu'}, 'Position', [0.92, 0.45, 0.07, 0.1], 'Box', 'off', 'Color', 'none');
sgtitle(sprintf('Simulated mu: With vs Without Integration (eval\\_ana = %d)', eval_ana_val), 'FontWeight', 'bold');
exportgraphics(fig_mu, fullfile(save_fig_dir, sprintf('simulatedMu_withVsNoInt_evalAna%d.pdf', eval_ana_val)), 'ContentType', 'vector');

%% =================== FIGURE: MU, NO INTEGRATION (eval_ana = 0) ==========
% eval_ana_val = 0; % only eval_ana = 0 kept active -- see design notes above
% data_noInt = readtable(fullfile(data_dir, 'data_agent_longBlocks_noInt_evalAna0_increasedRange.txt'));
% sigma_values = unique(data_noInt.sigma)';
% conditions = unique(data_noInt.choice_cond)';
% num_trials_long = max(data_noInt.trials);
%
% fig_mu = figure('Position', [50, 50, 1500, 500]);
% plot_idx = 1;
% for c = 1:length(conditions)
%     cond = conditions(c);
%     for s = 1:length(sigma_values)
%         sigma_val = sigma_values(s);
%         ax = subplot(length(conditions), length(sigma_values), plot_idx);
%         hold on;
%
%         values_noInt = local_trial_values(data_noInt, cond, sigma_val, 'mu', num_trials_long);
%         mean_noInt = mean(values_noInt, 2);
%         sem_noInt = std(values_noInt, 0, 2) ./ sqrt(size(values_noInt, 2));
%
%         h_noInt_lines = plot(1:num_trials_long, values_noInt, 'Color', noIntegration_color_light, 'LineWidth', line_width_thin);
%         h_noInt_mu = shadedErrorBar(1:num_trials_long, mean_noInt, sem_noInt, {'Color', noIntegration_color, 'LineWidth', line_width_mean}, 0);
%         h_trueMu = yline(true_mu_by_cond(c), '--k', 'LineWidth', 1);
%
%         title(sprintf('Cond %d (mu=%.1f), sigma=%.2f', cond, true_mu_by_cond(c), sigma_val), 'FontWeight', 'normal', 'FontSize', font_size);
%         if c == length(conditions)
%             xlabel('Trial');
%         end
%         if s == 1
%             ylabel('Simulated mu');
%         end
%         set(ax, 'FontName', font_name, 'FontSize', font_size, 'LineWidth', 1, 'Color', 'none');
%         box off;
%         ylim([0, 1]);
%         plot_idx = plot_idx + 1;
%     end
% end
% legend([h_noInt_lines(1), h_noInt_mu.mainLine, h_trueMu], {'Individual simulations', 'Mean \pm SEM', 'True mu'}, 'Position', [0.92, 0.45, 0.07, 0.1], 'Box', 'off', 'Color', 'none');
% sgtitle(sprintf('Simulated mu: No Integration (eval\\_ana = %d)', eval_ana_val), 'FontWeight', 'bold');
% exportgraphics(fig_mu, fullfile(save_fig_dir, sprintf('simulatedMu_noInt_evalAna%d.pdf', eval_ana_val)), 'ContentType', 'vector');

%% =================== FIGURE: CHOICE BEHAVIOR (CORRECT), BINNED =========
% nbins = 20;
% fig_perf = figure('Position', [50, 50, 1500, 500]);
% plot_idx = 1;
% for c = 1:length(conditions)
%     cond = conditions(c);
%     for s = 1:length(sigma_values)
%         sigma_val = sigma_values(s);
%         ax = subplot(length(conditions), length(sigma_values), plot_idx);
%         hold on;
%
%         [bin_centers, binned_noInt] = local_binned_values(data_noInt, cond, sigma_val, 'correct', num_trials_long, nbins);
%         [~, binned_withInt] = local_binned_values(data_withInt, cond, sigma_val, 'correct', num_trials_long, nbins);
%
%         h_noInt_perf = plot(bin_centers, binned_noInt, 'Color', noIntegration_color, 'LineWidth', line_width_thin);
%         h_withInt_perf = plot(bin_centers, binned_withInt, 'Color', integration_color, 'LineWidth', line_width_thin);
%         ylim([0, 1]);
%
%         title(sprintf('Cond %d (mu=%.1f), sigma=%.2f', cond, true_mu_by_cond(c), sigma_val), 'FontWeight', 'normal', 'FontSize', font_size);
%         if c == length(conditions)
%             xlabel('Trial');
%         end
%         if s == 1
%             ylabel('Proportion correct');
%         end
%         set(ax, 'FontName', font_name, 'FontSize', font_size, 'LineWidth', 1, 'Color', 'none');
%         box off;
%         plot_idx = plot_idx + 1;
%     end
% end
% legend([h_noInt_perf(1), h_withInt_perf(1)], {'No Integration', 'Integration'}, 'Position', [0.92, 0.45, 0.07, 0.1], 'Box', 'off', 'Color', 'none');
% sgtitle(sprintf('Economic performance: With vs Without Integration (eval\\_ana = %d)', eval_ana_val), 'FontWeight', 'bold');
% exportgraphics(fig_perf, fullfile(save_fig_dir, sprintf('economicPerformance_evalAna%d.pdf', eval_ana_val)), 'ContentType', 'vector');

%% ========================================================================
%  Local functions
%  ========================================================================
function values_matrix = local_trial_values(data, cond, sigma_val, varname, num_trials_long)
    subset = data(data.choice_cond == cond & abs(data.sigma - sigma_val) < 1e-9, :);
    num_sims = height(subset) / num_trials_long;
    values_matrix = reshape(subset.(varname), num_trials_long, num_sims); % trials x sims
end

function [bin_centers, binned_matrix] = local_binned_values(data, cond, sigma_val, varname, num_trials_long, nbins)
    subset = data(data.choice_cond == cond & abs(data.sigma - sigma_val) < 1e-9, :);
    num_sims = height(subset) / num_trials_long;
    values_matrix = reshape(subset.(varname), num_trials_long, num_sims); % trials x sims
    bin_width = num_trials_long / nbins;
    bin_centers = NaN(nbins, 1);
    binned_matrix = NaN(nbins, num_sims); % bins x sims
    for b = 1:nbins
        trial_start = round((b-1)*bin_width) + 1;
        trial_end = round(b*bin_width);
        bin_centers(b) = (trial_start + trial_end) / 2;
        bin_vals = values_matrix(trial_start:trial_end, :); % trials-in-bin x sims
        binned_matrix(b, :) = mean(bin_vals, 1); % mean within bin, per sim
    end
end
