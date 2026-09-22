%% ========================================================================
%  Script: Parameter Recovery for fitReducedModelSpace.m
%  ========================================================================
clc; clearvars;
rng(123); % for reproducibility

%% =================== SETTINGS SHARED ACROSS ALL RUNS ====================
n_parameters = 20;      % number of ground-truth parameter sets to test, per model
n_startingPoints = 10;  % fmincon starting points per parameter set (see note 5 above)

block_size = 25;   % trials per block (task design; matches all three loaded datasets)

% fix_sigma toggles between the two recovery modes for all three models below:
%   false (default) - sigma is a free parameter, recovered alongside
%                      alpha/kappa (or kappa alone for bayesianAgent), same
%                      as fitReducedModelSpace.m.
%   true             - sigma is fixed to its simulated ground-truth value
%                      (sigmaTrue) and spliced into the NLL call as a
%                      constant, mirroring fitReducedModelSpace_fixedSigma.m
%                      (which fixes sigma to each subject's own
%                      perceptual-choice-fit value instead). Only
%                      alpha/kappa (or kappa alone) are then actually
%                      optimized; recovered_sigma is just set equal to
%                      sigmaTrue and its recovery plot is skipped, since
%                      sigma was never fit.
fix_sigma = false;

% Shared fmincon options for every multi-start fit in this script
optimOptions = optimoptions('fmincon', ...
    'Display', 'off', ...
    'Algorithm', 'interior-point', ...
    'FiniteDifferenceType', 'central', ...
    'MaxIterations', 1000, ...
    'FunctionTolerance', 1e-6);

% PATH STUFF -- anchor to the repo root so relative loads/saves below work
% regardless of MATLAB's current working directory.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end

%% =================== BASIC RL MODEL ======================================
% Bounds match fitReducedModelSpace.m's "BASIC RL MODEL" section exactly:
% params = [alpha, kappa, sigma]
lb_basicRL = [0, 1, 0];
ub_basicRL = [1, 100, 0.1];

% Ground-truth data is pre-simulated by debug_choice_simulation_ALLmodels.m
% rather than regenerated here, so it's loaded once
% and reused as-is -- no per-condition regeneration, since basicRL's
% fitting function has no condition-dependent behavior.
% safe_saveall stores the struct under the variable name 'newData' (its
% own parameter name), not the caller's variable name -- so load() wraps
% it one level deeper than a plain save() would.
loaded_basicRL = load(fullfile(desiredPath, 'Agent', 'model fitting', 'simdata_basicRL_choice_recovery.mat'));
simdata_basicRL = loaded_basicRL.newData;
alphaTrue_basicRL = simdata_basicRL.alphaTrue;
kappaTrue_basicRL = simdata_basicRL.kappaTrue;
sigmaTrue_basicRL = simdata_basicRL.sigmaTrue;
blocks_basicRL = simdata_basicRL.blocks;
condiff_basicRL = simdata_basicRL.condiff;
mu_hat_basicRL = simdata_basicRL.mu_hat;   % [n_trials x n_parameters]
choice_basicRL = simdata_basicRL.choice;   % [n_trials x n_parameters]
reward_basicRL = simdata_basicRL.reward;   % [n_trials x n_parameters], RAW (not action-0-recoded)

% Random starting points for fmincon: one row per parameter set, one
% column per starting point, for each of the 3 free parameters. Sigma's
% init points are drawn regardless of fix_sigma (cheap, and parfor needs
% the variable to exist even when the fix_sigma branch below doesn't use it).
initAlpha_basicRL = unifrnd(lb_basicRL(1), ub_basicRL(1), [n_parameters, n_startingPoints]);
initKappa_basicRL = unifrnd(lb_basicRL(2), ub_basicRL(2), [n_parameters, n_startingPoints]);
initSigma_basicRL = unifrnd(lb_basicRL(3), ub_basicRL(3), [n_parameters, n_startingPoints]);

% fmincon bounds for the parameters actually being optimized -- drops
% sigma's bounds when it's fixed rather than recovered.
if fix_sigma
    lb_fit_basicRL = lb_basicRL(1:2);
    ub_fit_basicRL = ub_basicRL(1:2);
else
    lb_fit_basicRL = lb_basicRL;
    ub_fit_basicRL = ub_basicRL;
end

recovered_alpha = NaN(n_parameters, 1);
recovered_kappa = NaN(n_parameters, 1);
recovered_sigma = NaN(n_parameters, 1);

parfor p = 1:n_parameters
    mu_hat = mu_hat_basicRL(:, p);

    % nll_basicRL_integrated's belief update assumes reward already in a
    % fixed, action-0-referenced frame. simulate_basicRL_integrated_choice
    % returns the RAW reward (was the simulated choice reinforced?), so
    % recode it here via the same helper that function uses internally,
    % before handing it to the likelihood.
    recoded_reward = fitSlider_ALLmodels.recode_rewards_choice(reward_basicRL(:, p), choice_basicRL(:, p));

    % --- Refit from n_startingPoints starts, keep the lowest-NLL fit ---
    best_nll = Inf;
    if fix_sigma
        best_params = NaN(1, 2);
    else
        best_params = NaN(1, 3);
    end
    for sp = 1:n_startingPoints
        if fix_sigma
            init_params = [initAlpha_basicRL(p, sp), initKappa_basicRL(p, sp)];
            nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated( ...
                [params(1), params(2), sigmaTrue_basicRL(p)], mu_hat, blocks_basicRL, recoded_reward, condiff_basicRL);
        else
            init_params = [initAlpha_basicRL(p, sp), initKappa_basicRL(p, sp), initSigma_basicRL(p, sp)];
            nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated( ...
                params, mu_hat, blocks_basicRL, recoded_reward, condiff_basicRL);
        end
        [params_fit, nll] = fmincon(nll_fun, init_params, [], [], [], [], ...
            lb_fit_basicRL, ub_fit_basicRL, [], optimOptions);
        if nll < best_nll
            best_nll = nll;
            best_params = params_fit;
        end
    end

    recovered_alpha(p) = best_params(1);
    recovered_kappa(p) = best_params(2);
    if fix_sigma
        recovered_sigma(p) = sigmaTrue_basicRL(p); % fixed, not fitted
    else
        recovered_sigma(p) = best_params(3);
    end

    fprintf('[basicRL] parameter set %d/%d recovered\n', p, n_parameters);
end

% Recovered-vs-true scatter plot for each free parameter
plot_recovery(alphaTrue_basicRL, recovered_alpha, 'basicRL: Alpha Recovery');
plot_recovery(kappaTrue_basicRL, recovered_kappa, 'basicRL: Kappa Recovery');
if ~fix_sigma
    plot_recovery(sigmaTrue_basicRL, recovered_sigma, 'basicRL: Sigma Recovery');
end

%% =================== RL + EST SENSITIVITY MODEL (RLsigma) ===============
% Bounds match fitReducedModelSpace.m's "RL + EST SENSITIVITY MODEL"
% section exactly: params = [alpha, kappa, sigma]
lb_RLsigma = [0, 1, 0];
ub_RLsigma = [1, 100, 0.1];

% Ground-truth data is pre-simulated by debug_choice_simulation_ALLmodels.m
% (see design note 2) rather than regenerated here, so it's loaded once
% and reused as-is -- no per-condition regeneration, since RLsigma's
% fitting function has no condition-dependent behavior (see design note 4).
loaded_RLsigma = load(fullfile(desiredPath, 'Agent', 'model fitting', 'simdata_RLsigma_choice_recovery.mat'));
simdata_RLsigma = loaded_RLsigma.newData;
alphaTrue_RLsigma = simdata_RLsigma.alphaTrue;
kappaTrue_RLsigma = simdata_RLsigma.kappaTrue;
sigmaTrue_RLsigma = simdata_RLsigma.sigmaTrue;
blocks_RLsigma = simdata_RLsigma.blocks;
condiff_RLsigma = simdata_RLsigma.condiff;
mu_hat_RLsigma = simdata_RLsigma.mu_hat;   % [n_trials x n_parameters]
choice_RLsigma = simdata_RLsigma.choice;   % [n_trials x n_parameters]
reward_RLsigma = simdata_RLsigma.reward;   % [n_trials x n_parameters], RAW (not action-0-recoded)

% Random starting points for fmincon: one row per parameter set, one
% column per starting point, for each of the 3 free parameters. Sigma's
% init points are drawn regardless of fix_sigma (cheap, and parfor needs
% the variable to exist even when the fix_sigma branch below doesn't use it).
initAlpha_RLsigma = unifrnd(lb_RLsigma(1), ub_RLsigma(1), [n_parameters, n_startingPoints]);
initKappa_RLsigma = unifrnd(lb_RLsigma(2), ub_RLsigma(2), [n_parameters, n_startingPoints]);
initSigma_RLsigma = unifrnd(lb_RLsigma(3), ub_RLsigma(3), [n_parameters, n_startingPoints]);

% fmincon bounds for the parameters actually being optimized -- drops
% sigma's bounds when it's fixed rather than recovered.
if fix_sigma
    lb_fit_RLsigma = lb_RLsigma(1:2);
    ub_fit_RLsigma = ub_RLsigma(1:2);
else
    lb_fit_RLsigma = lb_RLsigma;
    ub_fit_RLsigma = ub_RLsigma;
end

recovered_alpha = NaN(n_parameters, 1);
recovered_kappa = NaN(n_parameters, 1);
recovered_sigma = NaN(n_parameters, 1);

parfor p = 1:n_parameters
    mu_hat = mu_hat_RLsigma(:, p);

    % nll_RLsigma_VOI's belief update assumes reward already in a fixed,
    % action-0-referenced frame. simulate_RLsigma_integrated_choice
    % returns the RAW reward, so recode it here via the same helper that
    % function uses internally (same convention as the basicRL section
    % above).
    recoded_reward = fitSlider_ALLmodels.recode_rewards_choice(reward_RLsigma(:, p), choice_RLsigma(:, p));

    % --- Refit from n_startingPoints starts, keep the lowest-NLL fit ---
    best_nll = Inf;
    if fix_sigma
        best_params = NaN(1, 2);
    else
        best_params = NaN(1, 3);
    end
    for sp = 1:n_startingPoints
        if fix_sigma
            init_params = [initAlpha_RLsigma(p, sp), initKappa_RLsigma(p, sp)];
            nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI( ...
                [params(1), params(2), sigmaTrue_RLsigma(p)], mu_hat, blocks_RLsigma, recoded_reward, condiff_RLsigma);
        else
            init_params = [initAlpha_RLsigma(p, sp), initKappa_RLsigma(p, sp), initSigma_RLsigma(p, sp)];
            nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI( ...
                params, mu_hat, blocks_RLsigma, recoded_reward, condiff_RLsigma);
        end
        [params_fit, nll] = fmincon(nll_fun, init_params, [], [], [], [], ...
            lb_fit_RLsigma, ub_fit_RLsigma, [], optimOptions);
        if nll < best_nll
            best_nll = nll;
            best_params = params_fit;
        end
    end

    recovered_alpha(p) = best_params(1);
    recovered_kappa(p) = best_params(2);
    if fix_sigma
        recovered_sigma(p) = sigmaTrue_RLsigma(p); % fixed, not fitted
    else
        recovered_sigma(p) = best_params(3);
    end

    fprintf('[RLsigma] parameter set %d/%d recovered\n', p, n_parameters);
end

plot_recovery(alphaTrue_RLsigma, recovered_alpha, 'RLsigma: Alpha Recovery');
plot_recovery(kappaTrue_RLsigma, recovered_kappa, 'RLsigma: Kappa Recovery');
if ~fix_sigma
    plot_recovery(sigmaTrue_RLsigma, recovered_sigma, 'RLsigma: Sigma Recovery');
end

%% =================== BAYESIAN AGENT MODEL ================================
% Bounds match fitReducedModelSpace.m's "BAYESIAN AGENT MODEL" section
% exactly: params = [kappa, sigma]
lb_bayesianAgent = [1, 0];
ub_bayesianAgent = [100, 0.1];

% Ground-truth data is pre-simulated by
% debug_choice_simulation_ALLmodels.m (see design note 2) rather than
% regenerated here, so it's loaded once and reused as-is -- no
% per-condition regeneration, since nll_bayesianAgent never sets
% agent.condition itself (see design note 3: it always runs at the Agent
% class's default, agentvars.m's condition = 2), and neither does
% simulate_bayesianAgent_integrated_choice.
loaded_bayesianAgent = load(fullfile(desiredPath, 'Agent', 'model fitting', 'simdata_bayesianAgent_choice_recovery.mat'));
simdata_bayesianAgent = loaded_bayesianAgent.newData;
kappaTrue_bayesianAgent = simdata_bayesianAgent.kappaTrue;
sigmaTrue_bayesianAgent = simdata_bayesianAgent.sigmaTrue;
blocks_bayesianAgent = simdata_bayesianAgent.blocks;
condiff_bayesianAgent = simdata_bayesianAgent.condiff;
mu_hat_bayesianAgent = simdata_bayesianAgent.mu_hat;   % [n_trials x n_parameters]
choice_bayesianAgent = simdata_bayesianAgent.choice;   % [n_trials x n_parameters]
reward_bayesianAgent = simdata_bayesianAgent.reward;   % [n_trials x n_parameters]
uniqueBlocks_bayesianAgent = unique(blocks_bayesianAgent);

% Random starting points for fmincon: one row per parameter set, one
% column per starting point, for each of the 2 free parameters. Sigma's
% init points are drawn regardless of fix_sigma (cheap, and parfor needs
% the variable to exist even when the fix_sigma branch below doesn't use it).
initKappa_bayesianAgent = unifrnd(lb_bayesianAgent(1), ub_bayesianAgent(1), [n_parameters, n_startingPoints]);
initSigma_bayesianAgent = unifrnd(lb_bayesianAgent(2), ub_bayesianAgent(2), [n_parameters, n_startingPoints]);

% fmincon bounds for the parameters actually being optimized -- drops
% sigma's bound when it's fixed rather than recovered.
if fix_sigma
    lb_fit_bayesianAgent = lb_bayesianAgent(1);
    ub_fit_bayesianAgent = ub_bayesianAgent(1);
else
    lb_fit_bayesianAgent = lb_bayesianAgent;
    ub_fit_bayesianAgent = ub_bayesianAgent;
end

recovered_kappa = NaN(n_parameters, 1);
recovered_sigma = NaN(n_parameters, 1);

parfor p = 1:n_parameters
    mu_hat = mu_hat_bayesianAgent(:, p);

    % nll_bayesianAgent recodes reward internally (via
    % Agent.compute_action_dep_rew, using data.choice/agent.a_t), so the
    % RAW reward is passed through as-is here -- unlike the basicRL/
    % RLsigma sections above, no external recoding is needed.
    dataTable = table();
    dataTable.blocks = blocks_bayesianAgent;
    dataTable.condiff_relative = condiff_bayesianAgent;
    dataTable.choice = choice_bayesianAgent(:, p);

    % --- Refit from n_startingPoints starts, keep the lowest-NLL fit ---
    best_nll = Inf;
    if fix_sigma
        best_params = NaN(1, 1);
    else
        best_params = NaN(1, 2);
    end
    for sp = 1:n_startingPoints
        if fix_sigma
            init_params = initKappa_bayesianAgent(p, sp);
            nll_fun = @(params) fitSlider_ALLmodels.nll_bayesianAgent( ...
                [params(1), sigmaTrue_bayesianAgent(p)], mu_hat, dataTable, length(uniqueBlocks_bayesianAgent), block_size, uniqueBlocks_bayesianAgent, reward_bayesianAgent(:, p));
        else
            init_params = [initKappa_bayesianAgent(p, sp), initSigma_bayesianAgent(p, sp)];
            nll_fun = @(params) fitSlider_ALLmodels.nll_bayesianAgent( ...
                params, mu_hat, dataTable, length(uniqueBlocks_bayesianAgent), block_size, uniqueBlocks_bayesianAgent, reward_bayesianAgent(:, p));
        end
        [params_fit, nll] = fmincon(nll_fun, init_params, [], [], [], [], ...
            lb_fit_bayesianAgent, ub_fit_bayesianAgent, [], optimOptions);
        if nll < best_nll
            best_nll = nll;
            best_params = params_fit;
        end
    end

    recovered_kappa(p) = best_params(1);
    if fix_sigma
        recovered_sigma(p) = sigmaTrue_bayesianAgent(p); % fixed, not fitted
    else
        recovered_sigma(p) = best_params(2);
    end

    fprintf('[bayesianAgent] parameter set %d/%d recovered\n', p, n_parameters);
end

plot_recovery(kappaTrue_bayesianAgent, recovered_kappa, 'BayesianAgent: Kappa Recovery');
if ~fix_sigma
    plot_recovery(sigmaTrue_bayesianAgent, recovered_sigma, 'BayesianAgent: Sigma Recovery');
end

%% ========================================================================
%  Local function: recovered-vs-true scatter plot
%  ------------------------------------------------------------------------
%  Shared plotting helper used by all three models above -- one scatter of
%  true parameter value (x) vs. best-recovered value (y), with a fitted
%  regression line. Points that fall along the diagonal indicate good
%  recovery; the printed correlation gives a quick numeric summary
%  (values near 1 = well-identified, values near 0 = poorly identified).
%  ========================================================================
function plot_recovery(true_vals, recovered_vals, title_str)
    color = copper(5);
    figure;
    scatter(true_vals, recovered_vals, 'filled', 'o', 'MarkerFaceColor', ...
        color(3,:), 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.3, 'SizeData', 50);
    lsline;
    xlabel('True Parameter');
    ylabel('Recovered Parameter');
    title(title_str);
    axis equal;
    grid on;

    r = corr(true_vals, recovered_vals);
    fprintf('%s: recovered-true correlation r = %.3f\n', title_str, r);
end
