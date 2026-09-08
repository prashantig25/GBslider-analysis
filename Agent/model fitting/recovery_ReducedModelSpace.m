%% ========================================================================
%  Script: Parameter Recovery for fitReducedModelSpace.m
%  ------------------------------------------------------------------------
%  Validates that the three models fit by fitReducedModelSpace.m (basicRL,
%  RLsigma, bayesianAgent) can recover their own known ground-truth
%  parameters. For each model this simulates synthetic slider data from a
%  set of "true" parameters, refits those synthetic data with the exact
%  same NLL function and parameter bounds fitReducedModelSpace.m uses, and
%  plots recovered vs. true values. This is done separately for both
%  conditions fitReducedModelSpace.m fits (Both, Perceptual) -> 2
%  conditions x 3 models = 6 recovery runs in this one script.
%
%  Design notes (why this differs from the older per-model recovery
%  scripts, e.g. paramRecov_basicRL_conditin.m / paramRecov_RLSigma_condition.m
%  / paramRecov_Agent_Condition.m):
%
%  1. TRIAL/BLOCK COUNT matches what a real subject actually contributes
%     per condition: 4 blocks x 25 trials = 100 trials (task design, see
%     "Raw data preprocessing/preprocess_mainstudy.m": t = 25 trials per
%     block, num_blocks = 4 blocks per condition). The older recovery
%     scripts simulated 1000 trials across 40 blocks -- ~10x more data
%     than any subject provides, which makes recovery look better than it
%     would for the real per-subject fits.
%
%  2. RLsigma's ground truth is simulated with predict_allModels.predict_RLsigma
%     (single Q-value, branching update), NOT predict_RLsigma_updated (a
%     dual-Q-value model). nll_RLsigma_VOI -- the function actually used by
%     fitReducedModelSpace.m and refit here -- assumes the single-Q
%     branching dynamics, so simulating with predict_RLsigma_updated would
%     silently test recovery of the wrong model (this was the case in
%     paramRecov_RLSigma_condition.m).
%
%  3. For the Bayesian Agent model, predict_allModels.predict_bayesianAgent
%     takes a "condition" argument that sets agent.condition, which
%     changes the agent's internal sensitivity/contingency parameters
%     (see Agent/agentvars.m). However, fitSlider_ALLmodels.nll_bayesianAgent
%     -- the function this script refits with -- never sets agent.condition
%     itself, so it always runs with the Agent class's default,
%     agentvars.m's `condition = 2`. To keep the simulated data consistent
%     with what the fitting function actually assumes, this script always
%     simulates with condition = 2 as well, for BOTH the "Both" and
%     "Perceptual" recovery runs -- varying it per condition would
%     reintroduce the same kind of generator/NLL mismatch fixed in (2).
%
%  4. Bounds/initial values match fitReducedModelSpace.m's fmincon calls
%     exactly (that script currently uses the SAME bounds for a given
%     model in both conditions, so the only thing that differs between
%     the "Both" and "Perceptual" runs below is the random draw of
%     ground-truth parameters/starting points/simulated data -- this is
%     intentional and mirrors fitReducedModelSpace.m's actual setup).
%
%  5. Multi-start (10 random starting points, keep the lowest-NLL fit) is
%     used here even though fitReducedModelSpace.m's production fit
%     currently uses a single starting point. This tests whether each
%     model is identifiable at all under generous optimization -- it is a
%     best-case check, not a simulation of the current single-start
%     production fit's actual behavior.
%  ========================================================================
clc; clearvars;
rng(123); % for reproducibility

%% =================== SETTINGS SHARED ACROSS ALL RUNS ====================
n_parameters = 20;      % number of ground-truth parameter sets to test, per model/condition
n_startingPoints = 10;  % fmincon starting points per parameter set (see note 5 above)

nBlocks = 4;                       % blocks per condition (task design)
block_size = 25;                   % trials per block (task design)
n_trials = nBlocks * block_size;   % 100 trials -- matches real per-subject, per-condition volume
blocks = repelem(1:nBlocks, block_size)';  % block index for every simulated trial

% Condition labels used purely to organize/label the two recovery runs
% per model below (see design note 4: the underlying settings are
% currently identical between them, matching fitReducedModelSpace.m).
conditionLabels = {'Both', 'Perceptual'};

% Shared fmincon options for every multi-start fit in this script
optimOptions = optimoptions('fmincon', ...
    'Display', 'off', ...
    'Algorithm', 'interior-point', ...
    'FiniteDifferenceType', 'central', ...
    'MaxIterations', 1000, ...
    'FunctionTolerance', 1e-6);

%% =================== BASIC RL MODEL ======================================
% Bounds/init match fitReducedModelSpace.m's "BASIC RL MODEL" section
% exactly: params = [alpha, kappa, sigma]
lb_basicRL = [0, 1, 0];
ub_basicRL = [1, 100, 0.1];

% Draw n_parameters "true" parameter sets to test recovery of, uniformly
% across each parameter's fitting bounds
alphaTrue_basicRL = unifrnd(lb_basicRL(1), ub_basicRL(1), [n_parameters, 1]);
kappaTrue_basicRL = unifrnd(lb_basicRL(2), ub_basicRL(2), [n_parameters, 1]);
sigmaTrue_basicRL = unifrnd(lb_basicRL(3), ub_basicRL(3), [n_parameters, 1]);

% Random starting points for fmincon: one row per parameter set, one
% column per starting point, for each of the 3 free parameters
initAlpha_basicRL = unifrnd(lb_basicRL(1), ub_basicRL(1), [n_parameters, n_startingPoints]);
initKappa_basicRL = unifrnd(lb_basicRL(2), ub_basicRL(2), [n_parameters, n_startingPoints]);
initSigma_basicRL = unifrnd(lb_basicRL(3), ub_basicRL(3), [n_parameters, n_startingPoints]);

for c = 1:length(conditionLabels)
    % Fresh synthetic task structure for this condition's recovery run:
    % true hidden state per trial, and the perceptual evidence (condiff)
    % consistent with that state (negative for state 0, positive for
    % state 1, as in the real task's condiff_relative)
    state = randi([0 1], n_trials, 1);
    condiff = NaN(n_trials, 1);
    condiff(state == 0) = unifrnd(-0.08, 0, sum(state == 0), 1);
    condiff(state == 1) = unifrnd(0, 0.08, sum(state == 1), 1);

    % Preallocate recovered-parameter output for this condition's run
    recovered_alpha = NaN(n_parameters, 1);
    recovered_kappa = NaN(n_parameters, 1);
    recovered_sigma = NaN(n_parameters, 1);

    parfor p = 1:n_parameters
        % --- Simulate synthetic slider (mu_hat) data from KNOWN true params ---
        % predict_basicRL implements the same branching Q-update assumed by
        % fitSlider_ALLmodels.nll_basicRL_integrated, so refitting the
        % simulated data with that NLL is a fair identifiability test.
        [~, mu_hat, ~, ~, rewards] = predict_allModels.predict_basicRL( ...
            [alphaTrue_basicRL(p), kappaTrue_basicRL(p), sigmaTrue_basicRL(p)], ...
            blocks, state, condiff, 'sample');

        % --- Refit from n_startingPoints starts, keep the lowest-NLL fit ---
        best_nll = Inf;
        best_params = NaN(1, 3);
        for sp = 1:n_startingPoints
            init_params = [initAlpha_basicRL(p, sp), initKappa_basicRL(p, sp), initSigma_basicRL(p, sp)];
            nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated( ...
                params, mu_hat, blocks, rewards, condiff);
            [params_fit, nll] = fmincon(nll_fun, init_params, [], [], [], [], ...
                lb_basicRL, ub_basicRL, [], optimOptions);
            if nll < best_nll
                best_nll = nll;
                best_params = params_fit;
            end
        end

        recovered_alpha(p) = best_params(1);
        recovered_kappa(p) = best_params(2);
        recovered_sigma(p) = best_params(3);

        fprintf('[basicRL - %s] parameter set %d/%d recovered\n', conditionLabels{c}, p, n_parameters);
    end

    % Recovered-vs-true scatter plot for each free parameter
    plot_recovery(alphaTrue_basicRL, recovered_alpha, sprintf('basicRL: Alpha Recovery (%s condition)', conditionLabels{c}));
    plot_recovery(kappaTrue_basicRL, recovered_kappa, sprintf('basicRL: Kappa Recovery (%s condition)', conditionLabels{c}));
    plot_recovery(sigmaTrue_basicRL, recovered_sigma, sprintf('basicRL: Sigma Recovery (%s condition)', conditionLabels{c}));
end

%% =================== RL + EST SENSITIVITY MODEL (RLsigma) ===============
% Bounds/init match fitReducedModelSpace.m's "RL + EST SENSITIVITY MODEL"
% section exactly: params = [alpha, kappa, sigma]
lb_RLsigma = [0, 1, 0];
ub_RLsigma = [1, 100, 0.1];

alphaTrue_RLsigma = unifrnd(lb_RLsigma(1), ub_RLsigma(1), [n_parameters, 1]);
kappaTrue_RLsigma = unifrnd(lb_RLsigma(2), ub_RLsigma(2), [n_parameters, 1]);
sigmaTrue_RLsigma = unifrnd(lb_RLsigma(3), ub_RLsigma(3), [n_parameters, 1]);

initAlpha_RLsigma = unifrnd(lb_RLsigma(1), ub_RLsigma(1), [n_parameters, n_startingPoints]);
initKappa_RLsigma = unifrnd(lb_RLsigma(2), ub_RLsigma(2), [n_parameters, n_startingPoints]);
initSigma_RLsigma = unifrnd(lb_RLsigma(3), ub_RLsigma(3), [n_parameters, n_startingPoints]);

for c = 1:length(conditionLabels)
    state = randi([0 1], n_trials, 1);
    condiff = NaN(n_trials, 1);
    condiff(state == 0) = unifrnd(-0.08, 0, sum(state == 0), 1);
    condiff(state == 1) = unifrnd(0, 0.08, sum(state == 1), 1);

    recovered_alpha = NaN(n_parameters, 1);
    recovered_kappa = NaN(n_parameters, 1);
    recovered_sigma = NaN(n_parameters, 1);

    parfor p = 1:n_parameters
        % predict_RLsigma (NOT predict_RLsigma_updated) matches the
        % single-Q, belief-weighted branching update that
        % fitSlider_ALLmodels.nll_RLsigma_VOI assumes -- see design note 2.
        [~, mu_hat, rewards, ~] = predict_allModels.predict_RLsigma( ...
            [alphaTrue_RLsigma(p), kappaTrue_RLsigma(p), sigmaTrue_RLsigma(p)], ...
            blocks, state, condiff, 'sample');

        best_nll = Inf;
        best_params = NaN(1, 3);
        for sp = 1:n_startingPoints
            init_params = [initAlpha_RLsigma(p, sp), initKappa_RLsigma(p, sp), initSigma_RLsigma(p, sp)];
            nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI( ...
                params, mu_hat, blocks, rewards, condiff);
            [params_fit, nll] = fmincon(nll_fun, init_params, [], [], [], [], ...
                lb_RLsigma, ub_RLsigma, [], optimOptions);
            if nll < best_nll
                best_nll = nll;
                best_params = params_fit;
            end
        end

        recovered_alpha(p) = best_params(1);
        recovered_kappa(p) = best_params(2);
        recovered_sigma(p) = best_params(3);

        fprintf('[RLsigma - %s] parameter set %d/%d recovered\n', conditionLabels{c}, p, n_parameters);
    end

    plot_recovery(alphaTrue_RLsigma, recovered_alpha, sprintf('RLsigma: Alpha Recovery (%s condition)', conditionLabels{c}));
    plot_recovery(kappaTrue_RLsigma, recovered_kappa, sprintf('RLsigma: Kappa Recovery (%s condition)', conditionLabels{c}));
    plot_recovery(sigmaTrue_RLsigma, recovered_sigma, sprintf('RLsigma: Sigma Recovery (%s condition)', conditionLabels{c}));
end

%% =================== BAYESIAN AGENT MODEL ================================
% Bounds/init match fitReducedModelSpace.m's "BAYESIAN AGENT MODEL"
% section exactly: params = [kappa, sigma]
lb_bayesianAgent = [1, 0];
ub_bayesianAgent = [100, 0.1];

kappaTrue_bayesianAgent = unifrnd(lb_bayesianAgent(1), ub_bayesianAgent(1), [n_parameters, 1]);
sigmaTrue_bayesianAgent = unifrnd(lb_bayesianAgent(2), ub_bayesianAgent(2), [n_parameters, 1]);

initKappa_bayesianAgent = unifrnd(lb_bayesianAgent(1), ub_bayesianAgent(1), [n_parameters, n_startingPoints]);
initSigma_bayesianAgent = unifrnd(lb_bayesianAgent(2), ub_bayesianAgent(2), [n_parameters, n_startingPoints]);

% agent.condition is always 2 here regardless of the "Both"/"Perceptual"
% label -- see design note 3: nll_bayesianAgent never sets agent.condition,
% so it always runs at the Agent class's default (agentvars.m: condition = 2).
agentCondition = 2;

for c = 1:length(conditionLabels)
    state = randi([0 1], n_trials, 1);
    condiff = NaN(n_trials, 1);
    condiff(state == 0) = unifrnd(-0.08, 0, sum(state == 0), 1);
    condiff(state == 1) = unifrnd(0, 0.08, sum(state == 1), 1);

    recovered_kappa = NaN(n_parameters, 1);
    recovered_sigma = NaN(n_parameters, 1);

    parfor p = 1:n_parameters
        % Simulate ground-truth choices/rewards/mu_hat from the Bayesian
        % agent, then package them into the trial table nll_bayesianAgent
        % expects (it indexes into data.blocks/data.choice/data.condiff_relative).
        [~, mu_hat, rewards, choices] = predict_allModels.predict_bayesianAgent( ...
            [kappaTrue_bayesianAgent(p), sigmaTrue_bayesianAgent(p)], ...
            blocks, condiff, agentCondition, 'sample', state);

        dataTable = table();
        dataTable.blocks = blocks;
        dataTable.condiff_relative = condiff;
        dataTable.choice = choices;

        best_nll = Inf;
        best_params = NaN(1, 2);
        for sp = 1:n_startingPoints
            init_params = [initKappa_bayesianAgent(p, sp), initSigma_bayesianAgent(p, sp)];
            nll_fun = @(params) fitSlider_ALLmodels.nll_bayesianAgent( ...
                params, mu_hat, dataTable, length(unique(blocks)), block_size, unique(blocks), rewards);
            [params_fit, nll] = fmincon(nll_fun, init_params, [], [], [], [], ...
                lb_bayesianAgent, ub_bayesianAgent, [], optimOptions);
            if nll < best_nll
                best_nll = nll;
                best_params = params_fit;
            end
        end

        recovered_kappa(p) = best_params(1);
        recovered_sigma(p) = best_params(2);

        fprintf('[bayesianAgent - %s] parameter set %d/%d recovered\n', conditionLabels{c}, p, n_parameters);
    end

    plot_recovery(kappaTrue_bayesianAgent, recovered_kappa, sprintf('BayesianAgent: Kappa Recovery (%s condition)', conditionLabels{c}));
    plot_recovery(sigmaTrue_bayesianAgent, recovered_sigma, sprintf('BayesianAgent: Sigma Recovery (%s condition)', conditionLabels{c}));
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
