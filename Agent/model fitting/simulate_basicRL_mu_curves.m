%% ========================================================================
%  Script: Simulate basicRL and RLsigma Mu Curves, With vs Without
%  Integration, Across Sigma -- Both Condition
%  ------------------------------------------------------------------------
%  Forward-simulates both the basicRL and RLsigma models, for a given
%  kappa (Beta-distribution concentration governing simulated slider-
%  report noise) swept across the same five sigma values used for the
%  Agent simulations ([0.02, 0.04, 0.06, 0.08, 0.1]), at the task's real
%  Both-condition structure: 4 blocks of 25 trials each, true mu = 0.7
%  (Agent/taskvars.m's condition == 1 mapping). Runs num_sims independent
%  simulations per sigma for each of:
%    - basicRL, WITH integration: fitSlider_ALLmodels.
%      simulate_basicRL_integrated_choice -- belief marginalized over a
%      20-point set_o grid each trial (Agent.p_s_giv_o-based), same
%      mechanism as Agent.m's integrate_voi. Plain (winner-take-all)
%      alpha update.
%    - basicRL, WITHOUT integration: local_simulate_basicRL_pointObs_
%      choice (this file) -- belief from a single point estimate of the
%      actual observed condiff, no grid. Same plain alpha update.
%    - RLsigma, WITH integration: fitSlider_ALLmodels.
%      simulate_RLsigma_integrated_choice -- same grid marginalization as
%      basicRL's with-integration version, but a belief-weighted
%      (pi_0*alpha / pi_1*alpha) update instead of plain alpha.
%    - RLsigma, WITHOUT integration: local_simulate_RLsigma_pointObs_
%      choice (this file) -- same point-estimate belief as basicRL's
%      no-integration version, but the same belief-weighted update as
%      RLsigma's with-integration version.
%  Produces one figure: a 2 (model) x 5 (sigma) subplot grid, each panel
%  overlaying the no-integration and with-integration mean +/- SEM curves
%  for that model and sigma.
%  ------------------------------------------------------------------------
%  Design notes:
%  - fitSlider_ALLmodels.m has integrated (objective-PE) choice-generating
%    functions for both basicRL and RLsigma, and point-observation (no
%    integration) choice-generating variants for both -- but those
%    point-observation variants only exist with SUBJECTIVE PE (anchoring
%    each trial's update to the model's own previously SAMPLED mu_hat(n-1)
%    rather than a cleanly-tracked q_0_0). There is no existing point-
%    observation, OBJECTIVE-PE, choice-generating basicRL or RLsigma
%    function in that file, so the two local_simulate_*_pointObs_choice
%    functions below fill that gap: basicRL's mirrors nll_basicRL_
%    pointObs's point-estimate belief formula, RLsigma's mirrors
%    nll_RLsigma's belief-weighted point-estimate formula, both combined
%    with the existing *_integrated_choice functions' choice-generating
%    structure (objective PE throughout, q_0_0 tracked cleanly). Kept
%    local to this script rather than added to the shared fitSlider_
%    ALLmodels.m library.
%  - alpha (learning rate), beta (fixed softmax inverse temperature for
%    economic choice), and kappa aren't swept by this request -- defaulted
%    to 0.1, 100, and 20 respectively (alpha/beta matching Agent/
%    agentvars.m's own defaults). Adjust alpha_val/beta_val/kappa_val
%    below if different values are wanted.
%  - Task()/taskvars() (Agent/Task.m, Agent/taskvars.m) generate the
%    state/contrast sequence each trial -- condition=1 maps to mu=0.7 (the
%    "Both" condition), kappa_max=0.1 (20 discrete contrast levels),
%    Theta=0.5 state probability -- matching the real task's actual Both-
%    condition generative process. This is a DIFFERENT "kappa" from the RL
%    models' own kappa parameter (Beta concentration, in params) -- the
%    names collide but the quantities are unrelated. All four model/
%    integration variants share the SAME generated task sequence per
%    (sigma, simulation) iteration, for a fairer comparison.
%  - mu_hat (plotted here) = each model's own belief PLUS simulated report
%    noise, drawn as betarnd(q*kappa, (1-q)*kappa) -- same quantity each
%    model is fit against in real subject data.
%  - Each simulation's 4 blocks are independent replicate learning curves
%    (belief resets to 0.5 at the start of every block) -- reshaped and
%    pooled with the 50 simulations into 200 total replicates per (model,
%    integration, sigma) combination for the mean/SEM.
%  ========================================================================
clc; clearvars; close all;

% PATH STUFF
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_fig_dir = fullfile(desiredPath, 'Agent', 'simfigures');
if ~exist(save_fig_dir, 'dir')
    mkdir(save_fig_dir);
end

%% PARAMETERS
sigma_values = [0.02, 0.04, 0.06, 0.08, 0.1]; % same sweep used for the Agent simulations
kappa_val = 20;   % Beta-distribution concentration for simulated slider reports (higher = less report noise)
alpha_val = 0.1;  % learning rate -- matches agentvars.m's own default
beta_val = 100;   % fixed softmax inverse temperature for economic choice -- matches agentvars.m's own default
p_reward_correct = 0.7; % true mu, Both condition (Agent/taskvars.m: condition==1 -> mu=0.7)

num_sims = 50;
num_blocks = 4;
num_trials_per_block = 25;
n_trials_total = num_blocks * num_trials_per_block;

%% SIMULATE
% Indexed [trial, block, sim, sigma] for each of the four model/integration combinations.
basicRL_int_all = NaN(num_trials_per_block, num_blocks, num_sims, length(sigma_values));
basicRL_noInt_all = NaN(num_trials_per_block, num_blocks, num_sims, length(sigma_values));
RLsigma_int_all = NaN(num_trials_per_block, num_blocks, num_sims, length(sigma_values));
RLsigma_noInt_all = NaN(num_trials_per_block, num_blocks, num_sims, length(sigma_values));

for sg = 1:length(sigma_values)
    sigma_val = sigma_values(sg);
    params = [alpha_val, kappa_val, sigma_val];

    for sim = 1:num_sims
        % Generate this simulation's Both-condition task sequence (state,
        % contrast) fresh -- shared across all four model/integration
        % variants this iteration, for a fairer comparison.
        task = Task();
        task.condition = 1; % Both condition
        task.taskMu(0); % rangeMu = 0 -> fixed mu = 0.7 for condition 1

        state = NaN(n_trials_total, 1);
        condiff = NaN(n_trials_total, 1);
        blocks = NaN(n_trials_total, 1);
        for b = 1:num_blocks
            for t = 1:num_trials_per_block
                idx = (b-1)*num_trials_per_block + t;
                task.state_sample();
                task.contrast_sample();
                state(idx) = task.s_t;
                condiff(idx) = task.c_t;
                blocks(idx) = b;
            end
        end

        mu_hat_basicRL_int = fitSlider_ALLmodels.simulate_basicRL_integrated_choice(params, blocks, state, condiff, beta_val, p_reward_correct);
        mu_hat_basicRL_noInt = local_simulate_basicRL_pointObs_choice(params, blocks, state, condiff, beta_val, p_reward_correct);
        mu_hat_RLsigma_int = fitSlider_ALLmodels.simulate_RLsigma_integrated_choice(params, blocks, state, condiff, beta_val, p_reward_correct);
        mu_hat_RLsigma_noInt = local_simulate_RLsigma_pointObs_choice(params, blocks, state, condiff, beta_val, p_reward_correct);

        basicRL_int_all(:, :, sim, sg) = reshape(mu_hat_basicRL_int, num_trials_per_block, num_blocks);
        basicRL_noInt_all(:, :, sim, sg) = reshape(mu_hat_basicRL_noInt, num_trials_per_block, num_blocks);
        RLsigma_int_all(:, :, sim, sg) = reshape(mu_hat_RLsigma_int, num_trials_per_block, num_blocks);
        RLsigma_noInt_all(:, :, sim, sg) = reshape(mu_hat_RLsigma_noInt, num_trials_per_block, num_blocks);
    end
end

%% PLOT
% Grouped by integration type (rows), not by model -- each panel overlays
% basicRL vs RLsigma for that integration setting and sigma. Colors now
% distinguish MODEL (basicRL vs RLsigma) instead of integration setting,
% since the row already encodes with/without integration -- using
% different colors from the blue/orange pair so the two groupings aren't
% visually confused with each other.
font_size = 12;
font_name = 'Arial';
basicRL_color = [0, 100, 0]./255; % MATLAB default green
RLsigma_color = [0.4940, 0.1840, 0.5560]; % MATLAB default purple

integration_labels = {'No Integration', 'Integration'};
data_basicRL_by_type = {basicRL_noInt_all, basicRL_int_all};
data_RLsigma_by_type = {RLsigma_noInt_all, RLsigma_int_all};

figure('Position', [50, 50, 1500, 500]);
plot_idx = 1;
for it = 1:length(integration_labels)
    for sg = 1:length(sigma_values)
        ax = subplot(length(integration_labels), length(sigma_values), plot_idx);
        hold on;

        % Flatten block x sim into one replicate dimension -- each block
        % is an independent replicate learning curve (belief resets to
        % 0.5 at the start of every block).
        replicates_basicRL = reshape(data_basicRL_by_type{it}(:, :, :, sg), num_trials_per_block, num_blocks * num_sims);
        replicates_RLsigma = reshape(data_RLsigma_by_type{it}(:, :, :, sg), num_trials_per_block, num_blocks * num_sims);

        mean_basicRL = mean(replicates_basicRL, 2);
        sem_basicRL = std(replicates_basicRL, 0, 2) ./ sqrt(size(replicates_basicRL, 2));
        mean_RLsigma = mean(replicates_RLsigma, 2);
        sem_RLsigma = std(replicates_RLsigma, 0, 2) ./ sqrt(size(replicates_RLsigma, 2));

        h_basicRL = shadedErrorBar(1:num_trials_per_block, mean_basicRL, sem_basicRL, {'Color', basicRL_color, 'LineWidth', 1.5}, 0);
        h_RLsigma = shadedErrorBar(1:num_trials_per_block, mean_RLsigma, sem_RLsigma, {'Color', RLsigma_color, 'LineWidth', 1.5}, 0);
        h_trueMu = yline(p_reward_correct, '--k', 'LineWidth', 1);

        title(sprintf('%s, sigma=%.2f', integration_labels{it}, sigma_values(sg)), 'FontWeight', 'normal', 'FontSize', font_size);
        if it == length(integration_labels)
            xlabel('Trial (within block)');
        end
        if sg == 1
            ylabel('Simulated mu\_hat');
        end
        set(ax, 'FontName', font_name, 'FontSize', font_size, 'LineWidth', 1, 'Color', 'none');
        box off;
        ylim([0.5, 1]);
        plot_idx = plot_idx + 1;
    end
end
legend([h_basicRL.mainLine, h_RLsigma.mainLine, h_trueMu], {'basicRL', 'RLsigma', 'True mu'}, 'Position', [0.92, 0.45, 0.07, 0.1], 'Box', 'off', 'Color', 'none');
sgtitle(sprintf('basicRL vs RLsigma, Grouped by Integration (Both Condition, kappa=%.0f, alpha=%.2f)', kappa_val, alpha_val), 'FontWeight', 'bold');

exportgraphics(gcf, fullfile(save_fig_dir, 'basicRLvsRLsigma_simulatedMu_both_groupedByIntegration_acrossSigma.pdf'), 'ContentType', 'vector');

%% ========================================================================
%  Local functions
%  ========================================================================
function mu_hat = local_simulate_basicRL_pointObs_choice(params, blocks, state, condiff, beta, p_reward_correct)
    % No-integration, objective-PE counterpart to fitSlider_ALLmodels.
    % simulate_basicRL_integrated_choice -- see this script's header for
    % why this doesn't already exist in fitSlider_ALLmodels.m. Belief
    % comes from a single point estimate of condiff(n) (matching
    % nll_basicRL_pointObs's formula) instead of a marginalized grid;
    % q_0_0 is tracked objectively (not anchored to the model's own
    % previous sampled report, unlike the _subjectivePE variants).
    alpha = params(1);
    kappa = params(2);
    sigma = params(3);

    n_trials = length(condiff);
    mu_hat = NaN(n_trials, 1);
    q_0_0 = 0.5;

    for n = 1:n_trials
        if n == 1 || blocks(n) ~= blocks(n-1)
            q_0_0 = 0.5;
        end

        belief_state = normcdf(condiff(n), 0, sigma);
        if condiff(n) < 0
            pi_0 = 1 - belief_state;
            pi_1 = belief_state;
        else
            pi_1 = belief_state;
            pi_0 = 1 - belief_state;
        end

        v_a0 = (pi_0 - pi_1) * q_0_0 + pi_1; % uses q_0_0 from BEFORE this trial's update
        v_a1 = 1 - v_a0;
        p_choice0 = 1 / (1 + exp(-beta * (v_a0 - v_a1)));
        choice_n = double(rand >= p_choice0); % 0 w.p. p_choice0, else 1

        if choice_n == state(n)
            reward_n = double(rand < p_reward_correct);
        else
            reward_n = double(rand < (1 - p_reward_correct));
        end
        recoded_reward = fitSlider_ALLmodels.recode_rewards_choice(reward_n, choice_n);

        % Plain (winner-take-all) alpha update -- matches nll_basicRL_pointObs.
        if pi_0 >= pi_1
            q_0_0 = q_0_0 + alpha * (recoded_reward - q_0_0);
        else
            q_0_0 = q_0_0 + alpha * ((1 - recoded_reward) - q_0_0);
        end

        a = max(q_0_0 * kappa, 1e-9);
        b = max((1 - q_0_0) * kappa, 1e-9);
        mu_hat(n) = betarnd(a, b);
    end
end

function mu_hat = local_simulate_RLsigma_pointObs_choice(params, blocks, state, condiff, beta, p_reward_correct)
    % No-integration, objective-PE counterpart to fitSlider_ALLmodels.
    % simulate_RLsigma_integrated_choice -- see this script's header for
    % why this doesn't already exist in fitSlider_ALLmodels.m. Belief
    % comes from a single point estimate of condiff(n) (matching
    % nll_RLsigma's formula) instead of a marginalized grid, with the
    % same belief-weighted (pi_0*alpha / pi_1*alpha) update as
    % simulate_RLsigma_integrated_choice -- the only difference from
    % local_simulate_basicRL_pointObs_choice above is this weighting.
    alpha = params(1);
    kappa = params(2);
    sigma = params(3);

    n_trials = length(condiff);
    mu_hat = NaN(n_trials, 1);
    q_0_0 = 0.5;

    for n = 1:n_trials
        if n == 1 || blocks(n) ~= blocks(n-1)
            q_0_0 = 0.5;
        end

        belief_state = normcdf(condiff(n), 0, sigma);
        if condiff(n) < 0
            pi_0 = 1 - belief_state;
            pi_1 = belief_state;
        else
            pi_1 = belief_state;
            pi_0 = 1 - belief_state;
        end

        v_a0 = (pi_0 - pi_1) * q_0_0 + pi_1; % uses q_0_0 from BEFORE this trial's update
        v_a1 = 1 - v_a0;
        p_choice0 = 1 / (1 + exp(-beta * (v_a0 - v_a1)));
        choice_n = double(rand >= p_choice0); % 0 w.p. p_choice0, else 1

        if choice_n == state(n)
            reward_n = double(rand < p_reward_correct);
        else
            reward_n = double(rand < (1 - p_reward_correct));
        end
        recoded_reward = fitSlider_ALLmodels.recode_rewards_choice(reward_n, choice_n);

        % Belief-weighted update -- matches nll_RLsigma.
        if pi_0 >= pi_1
            q_0_0 = q_0_0 + pi_0 * alpha * (recoded_reward - q_0_0);
        else
            q_0_0 = q_0_0 + pi_1 * alpha * ((1 - recoded_reward) - q_0_0);
        end

        a = max(q_0_0 * kappa, 1e-9);
        b = max((1 - q_0_0) * kappa, 1e-9);
        mu_hat(n) = betarnd(a, b);
    end
end
