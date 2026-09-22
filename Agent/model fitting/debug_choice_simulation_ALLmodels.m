%% ========================================================================
%  Script: Simulate + save choice data for recovery_ReducedModelSpace.m
%  (basicRL, RLsigma, bayesianAgent)
%  ------------------------------------------------------------------------
%  Each section below simulates n_parameters random true parameter sets for that model over
%  40 blocks of 25 trials (1000 trials), sanity-checks the mechanism with
%  the same two pooled plots, and saves the result for
%  recovery_ReducedModelSpace.m to load and refit.
%
%  Each section re-seeds with rng(123) before drawing anything, so its
%  output is identical to what the old standalone script for that model
%  produced -- sections run sequentially in ONE script, so without
%  re-seeding, RLsigma/bayesianAgent would draw from further along the
%  random stream (after basicRL's draws) and silently produce different
%  data than before.
%  ========================================================================
clc; clearvars;

n_parameters = 20; % number of parameters for recovery
nBlocks = 40; % number of blocks to be simulated
block_size = 25; % number of trials per block
n_trials = nBlocks * block_size; % total trials
blocks = repelem(1:nBlocks, block_size)'; % create blocks

p_reward_correct = 0.7;   % state-action reward contingency (all three models)
beta = 3;                 % fixed softmax inverse-temperature for basicRL/RLsigma only

% PATH STUFF -- anchor the save location to Agent/model fitting/ so
% recovery_ReducedModelSpace.m (same folder) can load it reliably.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting'); % where the simulated data gets saved

%% =================== BASIC RL ============================================
rng(123); % reset seed so this section's draws are reproducible on their own

% true hidden state per trial, and perceptual evidence (condiff)
% consistent with that state (negative for state 0, positive for state 1)
state = randi([0 1], n_trials, 1);
condiff = NaN(n_trials, 1);
condiff(state == 0) = unifrnd(-0.08, 0, sum(state == 0), 1);
condiff(state == 1) = unifrnd(0, 0.08, sum(state == 1), 1);

% fitting bounds for [alpha, kappa, sigma], and n_parameters true values
% drawn uniformly within them
lb_basicRL = [0, 1, 0];
ub_basicRL = [1, 100, 0.1];
alphaTrue = unifrnd(lb_basicRL(1), ub_basicRL(1), [n_parameters, 1]);
kappaTrue = unifrnd(lb_basicRL(2), ub_basicRL(2), [n_parameters, 1]);
sigmaTrue = unifrnd(lb_basicRL(3), ub_basicRL(3), [n_parameters, 1]);

% simulate mu_hat/choice/reward for each true parameter set
mu_hat_all = NaN(n_trials, n_parameters);
choice_all = NaN(n_trials, n_parameters);
reward_all = NaN(n_trials, n_parameters);
for p = 1:n_parameters
    [mu_hat_all(:,p), choice_all(:,p), reward_all(:,p), ~] = ...
        fitSlider_ALLmodels.simulate_basicRL_integrated_choice( ...
        [alphaTrue(p), kappaTrue(p), sigmaTrue(p)], blocks, state, condiff, beta, p_reward_correct);
end

% print a quick summary before looking at the plots - remove this
% eventually
fprintf('[basicRL] beta=%.1f, p_reward_correct=%.2f, %d parameter sets x %d trials\n', ...
    beta, p_reward_correct, n_parameters, n_trials);
fprintf('[basicRL] Overall choice accuracy, pooled across parameter sets: %.3f\n', ...
    mean(choice_all(:) == repmat(state, n_parameters, 1)));

% sanity-check plots: does accuracy/mu_hat look sensible within a block? -
% remove this eventually
plot_choice_sanity_checks(choice_all, mu_hat_all, state, block_size, nBlocks, n_parameters, 'basicRL');

% package everything needed to refit this model and save it
save_data = struct();
save_data.alphaTrue = alphaTrue;
save_data.kappaTrue = kappaTrue;
save_data.sigmaTrue = sigmaTrue;
save_data.beta = beta;
save_data.p_reward_correct = p_reward_correct;
save_data.blocks = blocks;
save_data.state = state;
save_data.condiff = condiff;
save_data.mu_hat = mu_hat_all;
save_data.choice = choice_all;
save_data.reward = reward_all;
safe_saveall(fullfile(save_dir, 'simdata_basicRL_choice_recovery.mat'), save_data);
fprintf('[basicRL] Saved %d simulated parameter sets to simdata_basicRL_choice_recovery.mat\n', n_parameters);

%% =================== RL + EST SENSITIVITY (RLsigma) ======================
rng(123); % reset seed so this section's draws are reproducible on their own

% true hidden state per trial, and perceptual evidence (condiff)
% consistent with that state
state = randi([0 1], n_trials, 1);
condiff = NaN(n_trials, 1);
condiff(state == 0) = unifrnd(-0.08, 0, sum(state == 0), 1);
condiff(state == 1) = unifrnd(0, 0.08, sum(state == 1), 1);

% fitting bounds for [alpha, kappa, sigma], and n_parameters true values
% drawn uniformly within them
lb_RLsigma = [0, 1, 0];
ub_RLsigma = [1, 100, 0.1];
alphaTrue = unifrnd(lb_RLsigma(1), ub_RLsigma(1), [n_parameters, 1]);
kappaTrue = unifrnd(lb_RLsigma(2), ub_RLsigma(2), [n_parameters, 1]);
sigmaTrue = unifrnd(lb_RLsigma(3), ub_RLsigma(3), [n_parameters, 1]);

% simulate mu_hat/choice/reward for each true parameter set
mu_hat_all = NaN(n_trials, n_parameters);
choice_all = NaN(n_trials, n_parameters);
reward_all = NaN(n_trials, n_parameters);
for p = 1:n_parameters
    [mu_hat_all(:,p), choice_all(:,p), reward_all(:,p), ~] = ...
        fitSlider_ALLmodels.simulate_RLsigma_integrated_choice( ...
        [alphaTrue(p), kappaTrue(p), sigmaTrue(p)], blocks, state, condiff, beta, p_reward_correct);
end

% print a quick summary before looking at the plots - remove eventually
fprintf('[RLsigma] beta=%.1f, p_reward_correct=%.2f, %d parameter sets x %d trials\n', ...
    beta, p_reward_correct, n_parameters, n_trials);
fprintf('[RLsigma] Overall choice accuracy (choice == state), pooled across parameter sets: %.3f\n', ...
    mean(choice_all(:) == repmat(state, n_parameters, 1)));

% sanity-check plots: does accuracy/mu_hat look sensible within a block? -
% remove eventually
plot_choice_sanity_checks(choice_all, mu_hat_all, state, block_size, nBlocks, n_parameters, 'RLsigma');

% package everything needed to refit this model and save it
save_data = struct();
save_data.alphaTrue = alphaTrue;
save_data.kappaTrue = kappaTrue;
save_data.sigmaTrue = sigmaTrue;
save_data.beta = beta;
save_data.p_reward_correct = p_reward_correct;
save_data.blocks = blocks;
save_data.state = state;
save_data.condiff = condiff;
save_data.mu_hat = mu_hat_all;
save_data.choice = choice_all;
save_data.reward = reward_all;
safe_saveall(fullfile(save_dir, 'simdata_RLsigma_choice_recovery.mat'), save_data);
fprintf('[RLsigma] Saved %d simulated parameter sets to simdata_RLsigma_choice_recovery.mat\n', n_parameters);

%% =================== BAYESIAN AGENT ======================================
rng(123); % reset seed so this section's draws are reproducible on their own

% true hidden state per trial, and perceptual evidence (condiff)
% consistent with that state
state = randi([0 1], n_trials, 1);
condiff = NaN(n_trials, 1);
condiff(state == 0) = unifrnd(-0.08, 0, sum(state == 0), 1);
condiff(state == 1) = unifrnd(0, 0.08, sum(state == 1), 1);

% fitting bounds for [kappa, sigma], and n_parameters true values drawn
% uniformly within them (no alpha/beta -- see the Agent class instead)
lb_bayesianAgent = [1, 0];
ub_bayesianAgent = [100, 0.1];
kappaTrue = unifrnd(lb_bayesianAgent(1), ub_bayesianAgent(1), [n_parameters, 1]);
sigmaTrue = unifrnd(lb_bayesianAgent(2), ub_bayesianAgent(2), [n_parameters, 1]);

% simulate mu_hat/choice/reward for each true parameter set
mu_hat_all = NaN(n_trials, n_parameters);
choice_all = NaN(n_trials, n_parameters);
reward_all = NaN(n_trials, n_parameters);
for p = 1:n_parameters
    [mu_hat_all(:,p), choice_all(:,p), reward_all(:,p), ~] = ...
        fitSlider_ALLmodels.simulate_bayesianAgent_integrated_choice( ...
        [kappaTrue(p), sigmaTrue(p)], blocks, state, condiff, p_reward_correct);
end

% print a quick summary before looking at the plots - remove eventually
fprintf('[bayesianAgent] p_reward_correct=%.2f, %d parameter sets x %d trials\n', ...
    p_reward_correct, n_parameters, n_trials);
fprintf('[bayesianAgent] Overall choice accuracy (choice == state), pooled across parameter sets: %.3f\n', ...
    mean(choice_all(:) == repmat(state, n_parameters, 1)));

% sanity-check plots: does accuracy/mu_hat look sensible within a block? -
% remove eventually
plot_choice_sanity_checks(choice_all, mu_hat_all, state, block_size, nBlocks, n_parameters, 'bayesianAgent');

% package everything needed to refit this model and save it to disk
save_data = struct();
save_data.kappaTrue = kappaTrue;
save_data.sigmaTrue = sigmaTrue;
save_data.p_reward_correct = p_reward_correct;
save_data.blocks = blocks;
save_data.state = state;
save_data.condiff = condiff;
save_data.mu_hat = mu_hat_all;
save_data.choice = choice_all;
save_data.reward = reward_all;
safe_saveall(fullfile(save_dir, 'simdata_bayesianAgent_choice_recovery.mat'), save_data);
fprintf('[bayesianAgent] Saved %d simulated parameter sets to simdata_bayesianAgent_choice_recovery.mat\n', n_parameters);

%% ========================================================================
%  Local function: choice/mu_hat sanity-check plots - REMOVE EVENTUALLY !!!
%  ------------------------------------------------------------------------
%  Shared by all three sections above -- two plots pooled across all
%  parameter sets x blocks:
%    1) mean choice == state (the more rewarding choice), stacked
%       block-over-block (+/- SEM across all blocks x parameter sets) by
%       within-block trial position 1-25 -- checks accuracy rises over
%       the course of a block as the agent learns, rather than sitting
%       flat at chance.
%    2) mean mu_hat stacked the same way -- checks the average
%       within-block learning curve looks sensible (starts near 0.5,
%       settles as the model's parameters dictate).
%  ========================================================================
function plot_choice_sanity_checks(choice_all, mu_hat_all, state, block_size, nBlocks, n_parameters, model_name)
    % --- Plot 1: accuracy (choice == state) by within-block trial ---
    correct_all = double(choice_all == repmat(state, 1, n_parameters)); % [n_trials x n_parameters]
    correct_byBlock = reshape(correct_all, block_size, nBlocks * n_parameters); % [trial-in-block x (block x param set)]
    acc_mean = mean(correct_byBlock, 2); % average across all blocks x parameter sets
    acc_sem = std(correct_byBlock, 0, 2) ./ sqrt(nBlocks * n_parameters); % SEM of that average

    figure('Position', [100 100 700 350]);
    errorbar(1:block_size, acc_mean, acc_sem, 'o-', 'LineWidth', 1.2, ...
        'Color', [0.2 0.4 0.8], 'MarkerFaceColor', [0.2 0.4 0.8]);
    yline(0.5, 'k--', 'chance');
    xlabel('Trial within block');
    ylabel('P(choice == state)');
    title(sprintf('%s: Mean choice == state by within-block trial, stacked over %d blocks x %d parameter sets (+/- SEM)', model_name, nBlocks, n_parameters));
    ylim([0 1]);
    grid on;

    % --- Plot 2: mu_hat by within-block trial ---
    mu_byBlock = reshape(mu_hat_all, block_size, nBlocks * n_parameters); % [trial-in-block x (block x param set)]
    mu_mean = mean(mu_byBlock, 2); % average across all blocks x parameter sets
    mu_sem = std(mu_byBlock, 0, 2) ./ sqrt(nBlocks * n_parameters); % SEM of that average

    figure('Position', [100 100 700 350]);
    errorbar(1:block_size, mu_mean, mu_sem, 'o-', 'LineWidth', 1.2, ...
        'Color', [0.8 0.3 0.3], 'MarkerFaceColor', [0.8 0.3 0.3]);
    yline(0.5, 'k--', 'no info');
    xlabel('Trial within block');
    ylabel('mu\_hat');
    title(sprintf('%s: Mean mu\\_hat by within-block trial, stacked over %d blocks x %d parameter sets (+/- SEM)', model_name, nBlocks, n_parameters));
    ylim([0 1]);
    grid on;
end
