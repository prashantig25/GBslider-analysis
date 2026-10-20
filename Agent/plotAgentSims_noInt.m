%% ========================================================================
%  Script: Plot Mu Curves and Economic Performance from Agent Simulations,
%  Task-Agent-Analysis = 0 (No Int) and = 1 (Int)
%  ------------------------------------------------------------------------
%  Loads two existing simulated datasets, both produced by
%  Agent/script_sims.m (no new simulation is run here):
%    - Data/agent simulations/data_agentpupil0.06.txt -- task_agent_
%      analysis = 1 ("Int"): observation is the exact presented contrast
%      difference, decisions read off deterministically (script_sims.m's
%      original "RUN SIMULATIONS FOR NORMATIVE AGENT" section, conditions
%      1 and 2, concatenated in its "SAVE ALL CONDITION SIMULATIONS"
%      section).
%    - Data/agent simulations/data_agent_noInt.txt -- task_agent_analysis
%      = 0 ("No Int"): observation sampled with perceptual noise,
%      decisions drawn probabilistically (script_sims.m's "SIMULATE
%      CONDITIONS 1 AND 2 WITH TASK_AGENT_ANALYSIS = 0" section).
%  For each dataset, produces the same two plots, one curve/bar per
%  condition (1 and 2):
%    1) Mean simulated mu (agent belief) across trial-within-block
%       position, with a shaded SEM band (lg_curves) and the task's
%       actual/true mu for that condition as a dotted reference line
%       (condition 1 -> 0.7, condition 2 -> 1, per Agent/Task.m's taskMu
%       with rangeMu = 0).
%    2) Mean economic performance (the `correct` column) as a bar per
%       condition, with an SEM error bar and individual per-block
%       simulation dots overlaid (bar_plots_pval).
%  ------------------------------------------------------------------------
%  Design notes:
%  - Both files share the exact same column layout (state, action,
%    reward, state_0, state_1, mu, trials, correct, contrast_diff,
%    subjective_obs, condition, contrast, value_a1, congruence,
%    choice_cond), so the same plot_mu_and_econperf() local function
%    below handles both.
%  - Neither file has an explicit simulation/block index column, so each
%    contiguous chunk of n_trials_per_block rows (detected via
%    max(data.trials), the trial-within-block counter, which resets to 1
%    at the start of every block) is treated as one independent block --
%    this matches Agent/task_agent_int.m's own behavior (the agent's
%    belief resets to 0.5 at the start of every block, so blocks don't
%    carry information across each other), and matches how each
%    condition's rows are known to be contiguous, uninterleaved chunks
%    (Agent/script_sims.m concatenates each condition's complete blocks,
%    not interleaved trial-by-trial).
%  - Mu-curve mean/SEM is computed by pooling every block (all
%    simulations x all blocks) at each trial-within-block position
%    directly, rather than the two-level block-then-subject averaging
%    Agent/model fitting/plotMuCongruence_group.m uses for real
%    multi-block subjects -- appropriate here since each block is already
%    an independent replication, so there's no "subject" level above
%    "block" to average over first.
%  - "Agent simulation dots" in the economic-performance plot are
%    per-block means (one dot per independent task_agent_int call), not
%    per the 300-simulation outer loop specifically -- there's no column
%    distinguishing which of the 4 blocks within a simulation a row came
%    from, so block is the finest independent unit available.
%  ========================================================================
clc; clearvars;

% PATH STUFF -- anchor to the repo root, regardless of MATLAB's current
% working directory.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
data_dir = fullfile(desiredPath, 'Data', 'agent simulations');

% STYLING
font_size = 12;
line_width = 2;
font_name = 'Arial';
[~,~,~,~,~,~,~,mix_color,perc_color,~,~,~,~,~,~,~,~,~,~,study2_blue] = colors_rgb();

conditions = [1, 2];
true_mu = [0.7, 1]; % Agent/Task.m's taskMu: condition 1 -> 0.7, condition 2 -> 1 (rangeMu = 0)
condition_labels = {'Both', 'Perceptual'};
condition_colors = [mix_color; perc_color];

%% =================== NO INT (task_agent_analysis = 0) =====================
plot_mu_and_econperf(fullfile(data_dir, 'data_agent_noInt.txt'), 'No Int', ...
    conditions, true_mu, condition_labels, condition_colors, study2_blue, font_size, line_width, font_name);

%% =================== INT (task_agent_analysis = 1) =========================
plot_mu_and_econperf(fullfile(data_dir, 'data_agentpupil0.06.txt'), 'Int', ...
    conditions, true_mu, condition_labels, condition_colors, study2_blue, font_size, line_width, font_name);

%% ========================================================================
%  Local function: mu-curve (lg_curves + true-mu yline) and economic-
%  performance (bar_plots_pval) plots for one dataset -- see header for
%  the full rationale and conventions.
%  ========================================================================
function plot_mu_and_econperf(data_file, title_suffix, conditions, true_mu, condition_labels, ...
    condition_colors, bar_color, font_size, line_width, font_name)

    data = readtable(data_file);
    n_trials_per_block = max(data.trials); % trial-within-block counter resets every block

    % MU CURVES
    mean_curves = NaN(length(conditions), n_trials_per_block);
    sem_curves = NaN(length(conditions), n_trials_per_block);
    for c = 1:length(conditions)
        cond_data = data(data.choice_cond == conditions(c), :);
        for t = 1:n_trials_per_block
            mu_t = cond_data.mu(cond_data.trials == t);
            mean_curves(c,t) = nanmean(mu_t);
            sem_curves(c,t) = nanstd(mu_t) / sqrt(sum(~isnan(mu_t)));
        end
    end

    figure('Position', [100, 100, 500, 400]);
    lg_curves(1:n_trials_per_block, mean_curves, sem_curves, condition_colors, condition_labels, ...
        sprintf('Simulated mu (%s)', title_suffix), 'Trial within block', 'Mean simulated mu', ...
        font_size, line_width, font_name);
    hold on;
    for c = 1:length(conditions)
        yline(true_mu(c), '--', 'Color', condition_colors(c,:), 'LineWidth', line_width);
    end

    % MEAN ECONOMIC PERFORMANCE
    block_perf = cell(1, length(conditions));
    for c = 1:length(conditions)
        cond_data = data(data.choice_cond == conditions(c), :);
        correct_matrix = reshape(cond_data.correct, n_trials_per_block, []).'; % n_blocks x n_trials_per_block
        block_perf{c} = nanmean(correct_matrix, 2); % one economic-performance value per block
    end

    n_blocks = length(block_perf{1});
    y = [block_perf{1}; block_perf{2}];
    mean_all = [nanmean(block_perf{1}); nanmean(block_perf{2})];
    SEM_all = [nanstd(block_perf{1})/sqrt(n_blocks); nanstd(block_perf{2})/sqrt(n_blocks)];

    figure('Position', [100, 100, 350, 400]);
    bar_plots_pval(y, mean_all, SEM_all, n_blocks, length(conditions), 1, {}, 1:length(conditions), ...
        condition_labels, sprintf('Mean economic performance (%s)', title_suffix), '', 'Proportion correct', ...
        0, 1, 15, 1, font_size, line_width, font_name, 0, bar_color);
end
