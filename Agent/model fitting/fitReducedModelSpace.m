%% ========================================================================
%  Script: Fit RL Models to Participant Data and Save Parameters (+AIC/BIC)
%  ------------------------------------------------------------------------
%  Sigma (perceptual sensitivity) is a free parameter here, estimated
%  jointly with each model's other parameters -- see
%  fitReducedModelSpace_fixedSigma.m for the counterpart that instead
%  fixes sigma to each subject's own perceptual-choice-fit value.
%  ------------------------------------------------------------------------
%  Structured as four self-contained sections, one per combination of:
%    - PE type: objective (model's own internally-tracked belief) vs
%      subjective (belief anchored to the subject's actual previous slider
%      report, mu_hat(n-1), instead)
%    - Observation handling: integrated (belief computed by marginalizing
%      over a discretized grid of hypothetical observations) vs no
%      integration (belief computed from a single point estimate of the
%      actual observed condiff)
%  Section order: 1) Objective PE, Integrated; 2) Subjective PE, Integrated;
%  3) Objective PE, No Integration; 4) Subjective PE, No Integration. Each
%  section fits the Both condition, then the Perceptual condition, then
%  runs its own AIC/BIC/model-selection comparison. All four sections
%  always run -- there are no PE-type or integration selector toggles; the
%  only toggle left is run_fitting, which controls whether each section's
%  fitting blocks (re)run or are skipped in favor of reloading that
%  section's previously-saved *_multiSP.mat files.
%  bayesianAgent only appears in Section 1 -- it has no subjective-PE or
%  no-integration variant (its belief update goes through the Agent
%  class's own internal mechanism, not a simple alpha-driven delta rule or
%  the set_o grid-integration Sections 3/4 use instead).
%  Uses the same shared fitSlider_ALLmodels comparison/plotting helpers
%  (compute_model_comparison, plot_model_comparison, plot_fitted_params,
%  compute_family_model_comparison, plot_family_model_comparison,
%  plot_scatter_spearman) as fitReducedModelSpace_fixedSigma.m.
%  ========================================================================
clc; clearvars;
rng(123); % for reproducability

% PATH STUFF -- anchor all saves/reads to Agent/model fitting/ so output
% always lands next to this script, regardless of MATLAB's current folder.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)'; % to which directory one must save in
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');

%% =================== LOAD AND PREPARE DATA =============================
[dataBoth, dataPerceptual, uniqueID, numSubjs] = fitSlider_ALLmodels.load_fitting_data();

% Multi-start: fmincon is a local optimizer, so draw multiple random
% starting points per subject within each model's [lb, ub] and keep the
% lowest-NLL fit, to reduce the risk of converging to a local minimum.
n_startingPoints = 10;

% run_fitting toggles whether the fitting blocks below (re)run at all, in
% all four sections of this script:
%   true (default)  - run every parfor fit and save fresh params_*/nll_*
%                      .mat files, as before.
%   false            - skip all fitting blocks entirely. The comparison
%                      sections already reload nll_*/params_* from disk
%                      via importdata() before using them, so with this
%                      set to false the whole rest of the script (AIC/BIC,
%                      model-selection, and all plots) runs straight off
%                      each section's previously-saved *_multiSP.mat
%                      files, without waiting through the fitting loops
%                      again.
run_fitting = true;

% Progress bar setup: parfor workers can't open/update a figure directly,
% so each completed (subject, starting point) combination is sent via
% DataQueue to the client, which updates a waitbar window
% (fitSlider_ALLmodels.progress_bar). progress_bar('reset', ...) is called
% before each parfor block to relabel the bar and restart it at 0%. A
% fresh DataQueue is created before each block's afterEach() call, since
% afterEach() adds a new listener rather than replacing the previous one
% -- reusing one queue across blocks would leave earlier blocks' listeners
% (with their stale captured label) still firing and overwriting the bar.

%% ========================================================================
%  SECTION 1 OF 4: OBJECTIVE PE, INTEGRATED
%  ------------------------------------------------------------------------
%  Belief each trial comes from integrating over a discretized grid of
%  hypothetical observations (nll_basicRL_integrated, nll_RLsigma_VOI,
%  nll_bayesianAgent), anchored to each model's own internally-tracked
%  running belief.
%  ========================================================================

%% ----------------- Both condition: fitting ------------------------------
%% --- Basic RL model ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_basicRL = NaN(numSubjs, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Basic RL (Both)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Basic RL (Both)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataBoth, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,3);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp)];
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]);
    end

    alphaParameter(n) = best_params(1);
    kappaParameter(n) = best_params(2);
    sigmaParameter(n) = best_params(3);
    nll_basicRL(n) = best_nll;
end
params_basicRL.alpha = alphaParameter;
params_basicRL.kappa = kappaParameter;
params_basicRL.sigma = sigmaParameter;
safe_saveall(fullfile(save_dir, 'params_basicRL_RBVoi_Both_multiSP.mat'), params_basicRL);
safe_saveall(fullfile(save_dir, 'nll_basicRL_sigma_RBVoi_Both_multiSP.mat'), nll_basicRL);
end

%% --- Bayesian Agent model ---
if run_fitting
sigmaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
nll_bayesianAgent = NaN(numSubjs, 1);
lb = [1, 0];
ub = [100, 0.1];
% Multi-start (see BASIC RL MODEL section above for rationale).
initKappa = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Bayesian Agent (Both)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Bayesian Agent (Both)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataBoth, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_bayesianAgent(params, subj.mu_hat, subj.dataTable, ...
        length(unique(subj.blocks)), 25, unique(subj.blocks), rewards);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,2);
    for sp = 1:n_startingPoints
        start_params = [initKappa(n,sp), initSigma(n,sp)];
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]);
    end

    sigmaParameter(n) = best_params(2);
    kappaParameter(n) = best_params(1);
    nll_bayesianAgent(n) = best_nll;
end
params_bayesianAgent.sigma = sigmaParameter;
params_bayesianAgent.kappa = kappaParameter;
safe_saveall(fullfile(save_dir, 'params_bayesianAgent_integratedBoth_multiSP.mat'), params_bayesianAgent);
safe_saveall(fullfile(save_dir, 'nll_bayesianAgent_integratedBoth_multiSP.mat'), nll_bayesianAgent);
end

%% --- RL + Est. Sensitivity model ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
nll_RLsigma = NaN(numSubjs, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
% Multi-start (see BASIC RL MODEL section above for rationale).
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'RL + Est. Sensitivity (Both)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'RL + Est. Sensitivity (Both)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataBoth, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,3);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp)];
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]);
    end

    alphaParameter(n) = best_params(1);
    sigmaParameter(n) = best_params(3);
    kappaParameter(n) = best_params(2);
    nll_RLsigma(n) = best_nll;
end
params_RLsigma.alpha = alphaParameter;
params_RLsigma.sigma = sigmaParameter;
params_RLsigma.kappa = kappaParameter;
safe_saveall(fullfile(save_dir, 'params_RLSigma_RBVoi_Both_multiSP.mat'), params_RLsigma);
safe_saveall(fullfile(save_dir, 'nll_RLSigma_RBVoi_Both_multiSP.mat'), nll_RLsigma);
end

%% ----------------- Perceptual condition: fitting -------------------------
%% --- Basic RL model (Perceptual) ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_basicRL = NaN(numSubjs, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
% Multi-start (see BASIC RL MODEL section above for rationale).
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Basic RL (Perceptual)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Basic RL (Perceptual)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,3);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp)];
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]);
    end

    alphaParameter(n) = best_params(1);
    kappaParameter(n) = best_params(2);
    sigmaParameter(n) = best_params(3);
    nll_basicRL(n) = best_nll;
end
params_basicRL.alpha = alphaParameter;
params_basicRL.kappa = kappaParameter;
params_basicRL.sigma = sigmaParameter;
safe_saveall(fullfile(save_dir, 'params_basicRL_RBVoi_Perceptual_multiSP.mat'), params_basicRL);
safe_saveall(fullfile(save_dir, 'nll_basicRL_sigma_RBVoi_Perceptual_multiSP.mat'), nll_basicRL);
end

%% --- Bayesian Agent model (Perceptual) ---
if run_fitting
sigmaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
nll_bayesianAgent = NaN(numSubjs, 1);
lb = [1, 0];
ub = [100, 0.1];
% Multi-start (see BASIC RL MODEL section above for rationale).
initKappa = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Bayesian Agent (Perceptual)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Bayesian Agent (Perceptual)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_bayesianAgent(params, subj.mu_hat, subj.dataTable, ...
        length(unique(subj.blocks)), 25, unique(subj.blocks), rewards);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,2);
    for sp = 1:n_startingPoints
        start_params = [initKappa(n,sp), initSigma(n,sp)];
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]);
    end

    sigmaParameter(n) = best_params(2);
    kappaParameter(n) = best_params(1);
    nll_bayesianAgent(n) = best_nll;
end
params_bayesianAgent.sigma = sigmaParameter;
params_bayesianAgent.kappa = kappaParameter;
safe_saveall(fullfile(save_dir, 'params_bayesianAgent_integratedPerceptual_multiSP.mat'), params_bayesianAgent);
safe_saveall(fullfile(save_dir, 'nll_bayesianAgent_integratedPerceptual_multiSP.mat'), nll_bayesianAgent);
end

%% --- RL + Est. Sensitivity model (Perceptual) ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
nll_RLsigma = NaN(numSubjs, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
% Multi-start (see BASIC RL MODEL section above for rationale).
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'RL + Est. Sensitivity (Perceptual)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'RL + Est. Sensitivity (Perceptual)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,3);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp)];
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]);
    end

    alphaParameter(n) = best_params(1);
    sigmaParameter(n) = best_params(3);
    kappaParameter(n) = best_params(2);
    nll_RLsigma(n) = best_nll;
end
params_RLsigma.alpha = alphaParameter;
params_RLsigma.sigma = sigmaParameter;
params_RLsigma.kappa = kappaParameter;
safe_saveall(fullfile(save_dir, 'params_RLSigma_RBVoi_Perceptual_multiSP.mat'), params_RLsigma);
safe_saveall(fullfile(save_dir, 'nll_RLSigma_RBVoi_Perceptual_multiSP.mat'), nll_RLsigma);
end

%% ----------------- Model comparison ---------------------------------------
%% --- Both condition ---
nll_basicRL = importdata(fullfile(save_dir, "nll_basicRL_sigma_RBVoi_Both_multiSP.mat"));
nll_RLsigma = importdata(fullfile(save_dir, "nll_RLSigma_RBVoi_Both_multiSP.mat"));
nll_bayesianAgent = importdata(fullfile(save_dir, "nll_bayesianAgent_integratedBoth_multiSP.mat"));
% Reloaded (rather than reused in-memory) for the same reason as
% fitReducedModelSpace_fixedSigma.m: these variable names get overwritten
% by the Perceptual-condition fitting blocks above.
params_basicRL = importdata(fullfile(save_dir, "params_basicRL_RBVoi_Both_multiSP.mat"));
params_RLsigma = importdata(fullfile(save_dir, "params_RLSigma_RBVoi_Both_multiSP.mat"));
params_bayesianAgent = importdata(fullfile(save_dir, "params_bayesianAgent_integratedBoth_multiSP.mat"));

disp('Computing AIC and BIC for all models...');
% Number of trials per subject -- NOT a fixed 200: each condition (Both/
% Perceptual) only gets 4 of the task's 12 blocks (100 trials), and
% requireMuHat further drops any trials with a missing slider response,
% so this is computed per subject rather than assumed. Named per-condition
% (num_trials_both, not a shared num_trials) so the Perceptual block below
% can't silently overwrite the value this Both-condition comparison needs.
num_trials_both = NaN(numSubjs, 1);
for n = 1:numSubjs
    num_trials_both(n) = length(preprocess_fitSlider(dataBoth, uniqueID(n)).mu_hat);
end
% Model parameter counts -- sigma is a free parameter here (unlike
% fitReducedModelSpace_fixedSigma.m), so each model has one more parameter
% than its fixed-sigma counterpart. [basicRL, RLsigma, BayesianAgent]
num_params = [3, 3, 2];
model_names = {'basicRL','RLsigma','BayesianAgent'};
% One call computes AIC/BIC/delta-BIC per subject and model, finds each
% subject's best-fitting model, and runs Bayesian Model Selection (BMS)
% across subjects -- see fitSlider_ALLmodels.compute_model_comparison.
[AICBIC_table, bestModel_table, BMS_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL, nll_RLsigma, nll_bayesianAgent], model_names, num_params, num_trials_both, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_integratedBoth_reducedMS_multiSP.mat'), AICBIC_table);
safe_saveall(fullfile(save_dir, 'bestModel_Both_multiSP.mat'), bestModel_table);
disp('All model parameters and AIC/BIC estimated and saved.');

% Plot proportion-best-fit bar chart, BMS probability/exceedance bars, and
% the winner-highlighted summary figure -- see plot_model_comparison.
fitSlider_ALLmodels.plot_model_comparison(AICBIC_table, bestModel_table, BMS_table, model_names, 'Both condition');

% Plot fitted alpha/kappa/sigma across models, plus summed AIC/BIC and
% mean delta-BIC. sigma is fitted (not fixed) here, unlike
% fitReducedModelSpace_fixedSigma.m -- sigma_title reflects that.
bic_ax_both_obj = fitSlider_ALLmodels.plot_fitted_params({params_basicRL, params_RLsigma, params_bayesianAgent}, model_names, AICBIC_table, 'Both condition', 'sigma (fitted)');

% Family-wise Model Comparison using VBA Toolbox: convert BIC to log model
% evidence and group models into families: Basic RL alone vs. the two
% LR-modulation models (RLsigma, BayesianAgent).
BIC_matrix = [AICBIC_table.BIC_basicRL, ...
                  AICBIC_table.BIC_RLsigma, ...
                  AICBIC_table.BIC_BayesianAgent]; % numSubjs x 3 matrix of BIC values, one column per model
lme = -0.5 * BIC_matrix; % convert BIC to (approximate) log model evidence
family_names = {'Basic_RL', 'LR_Modulation'};
families{1} = [1];       % BasicRL -- column index into lme/BIC_matrix/model_names
families{2} = [2, 3];    % LR modulation models (RLsigma, BayesianAgent)

fprintf('Running family-wise model comparison...\n');
% Runs VBA_groupBMC (falling back to a manual approximation if that
% toolbox call fails) to get family- and model-level probabilities.
[results_summary, posterior, out] = fitSlider_ALLmodels.compute_family_model_comparison(lme, families, family_names, model_names);
save(fullfile(save_dir, 'family_wise_BMS_results_both_multiSP.mat'), 'results_summary', 'posterior', 'out');
fprintf('\nResults saved to: family_wise_BMS_results_both_multiSP.mat\n');

% Plots family probabilities/exceedance probabilities, individual model
% probabilities, a within-family stacked bar, and prints a text summary.
fitSlider_ALLmodels.plot_family_model_comparison(results_summary, 'Both condition');

%% --- Perceptual condition ---
nll_basicRL = importdata(fullfile(save_dir, "nll_basicRL_sigma_RBVoi_Perceptual_multiSP.mat"));
nll_RLsigma = importdata(fullfile(save_dir, "nll_RLSigma_RBVoi_Perceptual_multiSP.mat"));
nll_bayesianAgent = importdata(fullfile(save_dir, "nll_bayesianAgent_integratedPerceptual_multiSP.mat"));
% Reloaded for the same reason as the Both-condition block above: these
% variable names were last overwritten by this script's own fitting
% blocks, so reloading from disk guarantees the Perceptual-condition fit.
params_basicRL = importdata(fullfile(save_dir, "params_basicRL_RBVoi_Perceptual_multiSP.mat"));
params_RLsigma = importdata(fullfile(save_dir, "params_RLSigma_RBVoi_Perceptual_multiSP.mat"));
params_bayesianAgent = importdata(fullfile(save_dir, "params_bayesianAgent_integratedPerceptual_multiSP.mat"));

disp('Computing AIC and BIC for all models (perceptual condition)...');
num_trials_perceptual = NaN(numSubjs, 1);
for n = 1:numSubjs
    num_trials_perceptual(n) = length(preprocess_fitSlider(dataPerceptual, uniqueID(n)).mu_hat);
end
num_params = [3, 3, 2];
model_names = {'basicRL','RLsigma','BayesianAgent'};
[AICBIC_table, bestModel_table, BMS_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL, nll_RLsigma, nll_bayesianAgent], model_names, num_params, num_trials_perceptual, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_integratedPerceptual_reducedMS_multiSP.mat'), AICBIC_table);
safe_saveall(fullfile(save_dir, 'bestModel_Perceptual_multiSP.mat'), bestModel_table);
disp('All model parameters and AIC/BIC estimated and saved (perceptual condition).');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_table, bestModel_table, BMS_table, model_names, 'Perceptual condition');

bic_ax_perc_obj = fitSlider_ALLmodels.plot_fitted_params({params_basicRL, params_RLsigma, params_bayesianAgent}, model_names, AICBIC_table, 'Perceptual condition', 'sigma (fitted)');

BIC_matrix = [AICBIC_table.BIC_basicRL, ...
                  AICBIC_table.BIC_RLsigma, ...
                  AICBIC_table.BIC_BayesianAgent];
lme = -0.5 * BIC_matrix;
family_names = {'Basic_RL', 'LR_Modulation'};
families{1} = [1];
families{2} = [2, 3];

fprintf('Running family-wise model comparison (perceptual condition)...\n');
[results_summary, posterior, out] = fitSlider_ALLmodels.compute_family_model_comparison(lme, families, family_names, model_names);
save(fullfile(save_dir, 'family_wise_BMS_results_perceptual_multiSP.mat'), 'results_summary', 'posterior', 'out');
fprintf('\nResults saved to: family_wise_BMS_results_perceptual_multiSP.mat\n');

fitSlider_ALLmodels.plot_family_model_comparison(results_summary, 'Perceptual condition');

%% ========================================================================
%  SECTION 2 OF 4: SUBJECTIVE PE, INTEGRATED
%  ------------------------------------------------------------------------
%  Same basicRL / RLsigma models as Section 1, but using
%  nll_basicRL_integrated_subjectivePE / nll_RLsigma_VOI_subjectivePE
%  instead of nll_basicRL_integrated / nll_RLsigma_VOI: each trial's
%  delta-rule anchors to the subject's own actual previous slider report
%  (mu_hat(n-1)) rather than the model's internally-tracked running belief
%  -- see fitSlider_ALLmodels.m's nll_basicRL_integrated_subjectivePE
%  header for the full rationale. bayesianAgent has no subjective-PE
%  variant (its belief update isn't a simple alpha-driven delta rule), so
%  it's excluded from this section -- a 2-model comparison throughout.
%  ========================================================================

%% ----------------- Both condition: fitting ------------------------------
%% --- Basic RL model, subjective PE ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_basicRL_subjectivePE = NaN(numSubjs, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Basic RL subjective PE (Both)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Basic RL subjective PE (Both)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataBoth, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated_subjectivePE(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,3);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp)];
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]);
    end

    alphaParameter(n) = best_params(1);
    kappaParameter(n) = best_params(2);
    sigmaParameter(n) = best_params(3);
    nll_basicRL_subjectivePE(n) = best_nll;
end
params_basicRL_subjectivePE.alpha = alphaParameter;
params_basicRL_subjectivePE.kappa = kappaParameter;
params_basicRL_subjectivePE.sigma = sigmaParameter;
safe_saveall(fullfile(save_dir, 'params_basicRL_subjectivePEv2_Both_multiSP.mat'), params_basicRL_subjectivePE);
safe_saveall(fullfile(save_dir, 'nll_basicRL_subjectivePEv2_Both_multiSP.mat'), nll_basicRL_subjectivePE);
end

%% --- RL + Est. Sensitivity model, subjective PE ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_RLsigma_subjectivePE = NaN(numSubjs, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'RL + Est. Sensitivity subjective PE (Both)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'RL + Est. Sensitivity subjective PE (Both)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataBoth, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI_subjectivePE(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,3);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp)];
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]);
    end

    alphaParameter(n) = best_params(1);
    kappaParameter(n) = best_params(2);
    sigmaParameter(n) = best_params(3);
    nll_RLsigma_subjectivePE(n) = best_nll;
end
params_RLsigma_subjectivePE.alpha = alphaParameter;
params_RLsigma_subjectivePE.kappa = kappaParameter;
params_RLsigma_subjectivePE.sigma = sigmaParameter;
safe_saveall(fullfile(save_dir, 'params_RLsigma_subjectivePEv2_Both_multiSP.mat'), params_RLsigma_subjectivePE);
safe_saveall(fullfile(save_dir, 'nll_RLsigma_subjectivePEv2_Both_multiSP.mat'), nll_RLsigma_subjectivePE);
end

%% ----------------- Perceptual condition: fitting -------------------------
%% --- Basic RL model, subjective PE (Perceptual) ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_basicRL_subjectivePE = NaN(numSubjs, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Basic RL subjective PE (Perceptual)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Basic RL subjective PE (Perceptual)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated_subjectivePE(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,3);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp)];
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]);
    end

    alphaParameter(n) = best_params(1);
    kappaParameter(n) = best_params(2);
    sigmaParameter(n) = best_params(3);
    nll_basicRL_subjectivePE(n) = best_nll;
end
params_basicRL_subjectivePE.alpha = alphaParameter;
params_basicRL_subjectivePE.kappa = kappaParameter;
params_basicRL_subjectivePE.sigma = sigmaParameter;
safe_saveall(fullfile(save_dir, 'params_basicRL_subjectivePEv2_Perceptual_multiSP.mat'), params_basicRL_subjectivePE);
safe_saveall(fullfile(save_dir, 'nll_basicRL_subjectivePEv2_Perceptual_multiSP.mat'), nll_basicRL_subjectivePE);
end

%% --- RL + Est. Sensitivity model, subjective PE (Perceptual) ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_RLsigma_subjectivePE = NaN(numSubjs, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'RL + Est. Sensitivity subjective PE (Perceptual)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'RL + Est. Sensitivity subjective PE (Perceptual)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI_subjectivePE(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,3);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp)];
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]);
    end

    alphaParameter(n) = best_params(1);
    kappaParameter(n) = best_params(2);
    sigmaParameter(n) = best_params(3);
    nll_RLsigma_subjectivePE(n) = best_nll;
end
params_RLsigma_subjectivePE.alpha = alphaParameter;
params_RLsigma_subjectivePE.kappa = kappaParameter;
params_RLsigma_subjectivePE.sigma = sigmaParameter;
safe_saveall(fullfile(save_dir, 'params_RLsigma_subjectivePEv2_Perceptual_multiSP.mat'), params_RLsigma_subjectivePE);
safe_saveall(fullfile(save_dir, 'nll_RLsigma_subjectivePEv2_Perceptual_multiSP.mat'), nll_RLsigma_subjectivePE);
end

%% ----------------- Model comparison ---------------------------------------
%  2-way AIC/BIC/bestModel/BMS comparison between the two subjective-PE
%  models (basicRL_subjectivePE vs RLsigma_subjectivePE). No family-wise
%  VBA_groupBMC step here -- with only two models, a family-wise grouping
%  would just duplicate the model-level results below, so it's omitted
%  (same convention as fitReducedModelSpace_fixedSigma.m).
%% --- Both condition ---
nll_basicRL_subjectivePE = importdata(fullfile(save_dir, "nll_basicRL_subjectivePEv2_Both_multiSP.mat"));
nll_RLsigma_subjectivePE = importdata(fullfile(save_dir, "nll_RLsigma_subjectivePEv2_Both_multiSP.mat"));
params_basicRL_subjectivePE = importdata(fullfile(save_dir, "params_basicRL_subjectivePEv2_Both_multiSP.mat"));
params_RLsigma_subjectivePE = importdata(fullfile(save_dir, "params_RLsigma_subjectivePEv2_Both_multiSP.mat"));

disp('Computing AIC and BIC for subjective PE models...');
% Recomputed explicitly (not reused from Section 1 above) -- dataBoth and
% dataPerceptual can give different per-subject trial counts after
% requireMuHat filtering, so this must not risk picking up whichever
% condition's count happened to be last computed above.
num_trials_both = NaN(numSubjs, 1);
for n = 1:numSubjs
    num_trials_both(n) = length(preprocess_fitSlider(dataBoth, uniqueID(n)).mu_hat);
end
num_params_subjectivePE = [3, 3]; % both subjective-PE models have 3 free params (alpha, kappa, sigma)
model_names_subjectivePE = {'basicRL_subjectivePE','RLsigma_subjectivePE'};
[AICBIC_subjectivePE_table, bestModel_subjectivePE_table, BMS_subjectivePE_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_subjectivePE, nll_RLsigma_subjectivePE], model_names_subjectivePE, num_params_subjectivePE, num_trials_both, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_subjectivePEv2_Both_reducedMS_multiSP.mat'), AICBIC_subjectivePE_table);
safe_saveall(fullfile(save_dir, 'bestModel_subjectivePEv2_Both_multiSP.mat'), bestModel_subjectivePE_table);
disp('Subjective PE model AIC/BIC estimated and saved.');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_subjectivePE_table, bestModel_subjectivePE_table, BMS_subjectivePE_table, model_names_subjectivePE, 'Both condition, Subjective PE');

bic_ax_both_subj = fitSlider_ALLmodels.plot_fitted_params({params_basicRL_subjectivePE, params_RLsigma_subjectivePE}, model_names_subjectivePE, AICBIC_subjectivePE_table, 'Both condition, Subjective PE', 'sigma (fitted)');

%% --- Perceptual condition ---
nll_basicRL_subjectivePE = importdata(fullfile(save_dir, "nll_basicRL_subjectivePEv2_Perceptual_multiSP.mat"));
nll_RLsigma_subjectivePE = importdata(fullfile(save_dir, "nll_RLsigma_subjectivePEv2_Perceptual_multiSP.mat"));
params_basicRL_subjectivePE = importdata(fullfile(save_dir, "params_basicRL_subjectivePEv2_Perceptual_multiSP.mat"));
params_RLsigma_subjectivePE = importdata(fullfile(save_dir, "params_RLsigma_subjectivePEv2_Perceptual_multiSP.mat"));

disp('Computing AIC and BIC for subjective PE models (Perceptual)...');
num_trials_perceptual = NaN(numSubjs, 1);
for n = 1:numSubjs
    num_trials_perceptual(n) = length(preprocess_fitSlider(dataPerceptual, uniqueID(n)).mu_hat);
end
num_params_subjectivePE = [3, 3];
model_names_subjectivePE = {'basicRL_subjectivePE','RLsigma_subjectivePE'};
[AICBIC_subjectivePE_table, bestModel_subjectivePE_table, BMS_subjectivePE_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_subjectivePE, nll_RLsigma_subjectivePE], model_names_subjectivePE, num_params_subjectivePE, num_trials_perceptual, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_subjectivePEv2_Perceptual_reducedMS_multiSP.mat'), AICBIC_subjectivePE_table);
safe_saveall(fullfile(save_dir, 'bestModel_subjectivePEv2_Perceptual_multiSP.mat'), bestModel_subjectivePE_table);
disp('Subjective PE model AIC/BIC estimated and saved (Perceptual).');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_subjectivePE_table, bestModel_subjectivePE_table, BMS_subjectivePE_table, model_names_subjectivePE, 'Perceptual condition, Subjective PE');

bic_ax_perc_subj = fitSlider_ALLmodels.plot_fitted_params({params_basicRL_subjectivePE, params_RLsigma_subjectivePE}, model_names_subjectivePE, AICBIC_subjectivePE_table, 'Perceptual condition, Subjective PE', 'sigma (fitted)');

%% ========================================================================
%  SECTION 3 OF 4: OBJECTIVE PE, NO INTEGRATION
%  ------------------------------------------------------------------------
%  Same basicRL / RLsigma models as Section 1, but using each model's
%  point-observation (non-integrated) likelihood instead of its integrated
%  one: nll_basicRL_pointObs instead of nll_basicRL_integrated, and
%  nll_RLsigma instead of nll_RLsigma_VOI. Belief each trial comes from a
%  single point estimate of the observation (normcdf(condiff(n), 0,
%  sigma)) rather than integrated over a discretized grid of hypothetical
%  observations -- see fitSlider_ALLmodels.m's nll_basicRL_pointObs header
%  for the full rationale. bayesianAgent is excluded (its belief update
%  goes through the Agent class's own mechanism, not the set_o grid this
%  section's integration concerns) -- a 2-model comparison throughout.
%  ========================================================================

%% ----------------- Both condition: fitting ------------------------------
%% --- Basic RL model, no integration ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_basicRL_noIntegration = NaN(numSubjs, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Basic RL, no integration (Both)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Basic RL, no integration (Both)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataBoth, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_pointObs(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,3);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp)];
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]);
    end

    alphaParameter(n) = best_params(1);
    kappaParameter(n) = best_params(2);
    sigmaParameter(n) = best_params(3);
    nll_basicRL_noIntegration(n) = best_nll;
end
params_basicRL_noIntegration.alpha = alphaParameter;
params_basicRL_noIntegration.kappa = kappaParameter;
params_basicRL_noIntegration.sigma = sigmaParameter;
safe_saveall(fullfile(save_dir, 'params_basicRL_noIntegration_Both_multiSP.mat'), params_basicRL_noIntegration);
safe_saveall(fullfile(save_dir, 'nll_basicRL_noIntegration_Both_multiSP.mat'), nll_basicRL_noIntegration);
end

%% --- RL + Est. Sensitivity model, no integration ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_RLsigma_noIntegration = NaN(numSubjs, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'RL + Est. Sensitivity, no integration (Both)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'RL + Est. Sensitivity, no integration (Both)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataBoth, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,3);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp)];
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]);
    end

    alphaParameter(n) = best_params(1);
    kappaParameter(n) = best_params(2);
    sigmaParameter(n) = best_params(3);
    nll_RLsigma_noIntegration(n) = best_nll;
end
params_RLsigma_noIntegration.alpha = alphaParameter;
params_RLsigma_noIntegration.sigma = sigmaParameter;
params_RLsigma_noIntegration.kappa = kappaParameter;
safe_saveall(fullfile(save_dir, 'params_RLsigma_noIntegration_Both_multiSP.mat'), params_RLsigma_noIntegration);
safe_saveall(fullfile(save_dir, 'nll_RLsigma_noIntegration_Both_multiSP.mat'), nll_RLsigma_noIntegration);
end

%% ----------------- Perceptual condition: fitting -------------------------
%% --- Basic RL model, no integration (Perceptual) ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_basicRL_noIntegration = NaN(numSubjs, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Basic RL, no integration (Perceptual)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Basic RL, no integration (Perceptual)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_pointObs(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,3);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp)];
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]);
    end

    alphaParameter(n) = best_params(1);
    kappaParameter(n) = best_params(2);
    sigmaParameter(n) = best_params(3);
    nll_basicRL_noIntegration(n) = best_nll;
end
params_basicRL_noIntegration.alpha = alphaParameter;
params_basicRL_noIntegration.kappa = kappaParameter;
params_basicRL_noIntegration.sigma = sigmaParameter;
safe_saveall(fullfile(save_dir, 'params_basicRL_noIntegration_Perceptual_multiSP.mat'), params_basicRL_noIntegration);
safe_saveall(fullfile(save_dir, 'nll_basicRL_noIntegration_Perceptual_multiSP.mat'), nll_basicRL_noIntegration);
end

%% --- RL + Est. Sensitivity model, no integration (Perceptual) ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_RLsigma_noIntegration = NaN(numSubjs, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'RL + Est. Sensitivity, no integration (Perceptual)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'RL + Est. Sensitivity, no integration (Perceptual)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,3);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp)];
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]);
    end

    alphaParameter(n) = best_params(1);
    kappaParameter(n) = best_params(2);
    sigmaParameter(n) = best_params(3);
    nll_RLsigma_noIntegration(n) = best_nll;
end
params_RLsigma_noIntegration.alpha = alphaParameter;
params_RLsigma_noIntegration.sigma = sigmaParameter;
params_RLsigma_noIntegration.kappa = kappaParameter;
safe_saveall(fullfile(save_dir, 'params_RLsigma_noIntegration_Perceptual_multiSP.mat'), params_RLsigma_noIntegration);
safe_saveall(fullfile(save_dir, 'nll_RLsigma_noIntegration_Perceptual_multiSP.mat'), nll_RLsigma_noIntegration);
end

%% ----------------- Model comparison ---------------------------------------
%% --- Both condition ---
nll_basicRL_noIntegration = importdata(fullfile(save_dir, "nll_basicRL_noIntegration_Both_multiSP.mat"));
nll_RLsigma_noIntegration = importdata(fullfile(save_dir, "nll_RLsigma_noIntegration_Both_multiSP.mat"));
params_basicRL_noIntegration = importdata(fullfile(save_dir, "params_basicRL_noIntegration_Both_multiSP.mat"));
params_RLsigma_noIntegration = importdata(fullfile(save_dir, "params_RLsigma_noIntegration_Both_multiSP.mat"));

disp('Computing AIC and BIC for no-integration models...');
num_trials_both = NaN(numSubjs, 1);
for n = 1:numSubjs
    num_trials_both(n) = length(preprocess_fitSlider(dataBoth, uniqueID(n)).mu_hat);
end
num_params_noIntegration = [3, 3]; % both no-integration models have 3 free params (alpha, kappa, sigma)
model_names_noIntegration = {'basicRL_noIntegration','RLsigma_noIntegration'};
[AICBIC_noIntegration_table, bestModel_noIntegration_table, BMS_noIntegration_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_noIntegration, nll_RLsigma_noIntegration], model_names_noIntegration, num_params_noIntegration, num_trials_both, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_noIntegration_Both_reducedMS_multiSP.mat'), AICBIC_noIntegration_table);
safe_saveall(fullfile(save_dir, 'bestModel_noIntegration_Both_multiSP.mat'), bestModel_noIntegration_table);
disp('No-integration model AIC/BIC estimated and saved.');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_noIntegration_table, bestModel_noIntegration_table, BMS_noIntegration_table, model_names_noIntegration, 'Both condition, No Integration');

bic_ax_both_obj_noInt = fitSlider_ALLmodels.plot_fitted_params({params_basicRL_noIntegration, params_RLsigma_noIntegration}, model_names_noIntegration, AICBIC_noIntegration_table, 'Both condition, No Integration', 'sigma (fitted)');

%% --- Perceptual condition ---
nll_basicRL_noIntegration = importdata(fullfile(save_dir, "nll_basicRL_noIntegration_Perceptual_multiSP.mat"));
nll_RLsigma_noIntegration = importdata(fullfile(save_dir, "nll_RLsigma_noIntegration_Perceptual_multiSP.mat"));
params_basicRL_noIntegration = importdata(fullfile(save_dir, "params_basicRL_noIntegration_Perceptual_multiSP.mat"));
params_RLsigma_noIntegration = importdata(fullfile(save_dir, "params_RLsigma_noIntegration_Perceptual_multiSP.mat"));

disp('Computing AIC and BIC for no-integration models (Perceptual)...');
num_trials_perceptual = NaN(numSubjs, 1);
for n = 1:numSubjs
    num_trials_perceptual(n) = length(preprocess_fitSlider(dataPerceptual, uniqueID(n)).mu_hat);
end
num_params_noIntegration = [3, 3];
model_names_noIntegration = {'basicRL_noIntegration','RLsigma_noIntegration'};
[AICBIC_noIntegration_table, bestModel_noIntegration_table, BMS_noIntegration_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_noIntegration, nll_RLsigma_noIntegration], model_names_noIntegration, num_params_noIntegration, num_trials_perceptual, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_noIntegration_Perceptual_reducedMS_multiSP.mat'), AICBIC_noIntegration_table);
safe_saveall(fullfile(save_dir, 'bestModel_noIntegration_Perceptual_multiSP.mat'), bestModel_noIntegration_table);
disp('No-integration model AIC/BIC estimated and saved (Perceptual).');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_noIntegration_table, bestModel_noIntegration_table, BMS_noIntegration_table, model_names_noIntegration, 'Perceptual condition, No Integration');

bic_ax_perc_obj_noInt = fitSlider_ALLmodels.plot_fitted_params({params_basicRL_noIntegration, params_RLsigma_noIntegration}, model_names_noIntegration, AICBIC_noIntegration_table, 'Perceptual condition, No Integration', 'sigma (fitted)');

%% ========================================================================
%  SECTION 4 OF 4: SUBJECTIVE PE, NO INTEGRATION
%  ------------------------------------------------------------------------
%  Combines Section 2's subjective-PE anchoring (mu_hat(n-1)) with Section
%  3's point-observation belief: nll_basicRL_pointObs_subjectivePE instead
%  of nll_basicRL_integrated_subjectivePE, nll_RLsigma_subjectivePE
%  instead of nll_RLsigma_VOI_subjectivePE. bayesianAgent is excluded (see
%  Sections 2 and 3 above) -- a 2-model comparison throughout.
%  ========================================================================

%% ----------------- Both condition: fitting ------------------------------
%% --- Basic RL model, subjective PE, no integration ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_basicRL_subjectivePE_noIntegration = NaN(numSubjs, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Basic RL subjective PE, no integration (Both)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Basic RL subjective PE, no integration (Both)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataBoth, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_pointObs_subjectivePE(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,3);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp)];
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]);
    end

    alphaParameter(n) = best_params(1);
    kappaParameter(n) = best_params(2);
    sigmaParameter(n) = best_params(3);
    nll_basicRL_subjectivePE_noIntegration(n) = best_nll;
end
params_basicRL_subjectivePE_noIntegration.alpha = alphaParameter;
params_basicRL_subjectivePE_noIntegration.kappa = kappaParameter;
params_basicRL_subjectivePE_noIntegration.sigma = sigmaParameter;
safe_saveall(fullfile(save_dir, 'params_basicRL_subjectivePE_noIntegration_Both_multiSP.mat'), params_basicRL_subjectivePE_noIntegration);
safe_saveall(fullfile(save_dir, 'nll_basicRL_subjectivePE_noIntegration_Both_multiSP.mat'), nll_basicRL_subjectivePE_noIntegration);
end

%% --- RL + Est. Sensitivity model, subjective PE, no integration ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_RLsigma_subjectivePE_noIntegration = NaN(numSubjs, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'RL + Est. Sensitivity subjective PE, no integration (Both)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'RL + Est. Sensitivity subjective PE, no integration (Both)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataBoth, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_subjectivePE(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,3);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp)];
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]);
    end

    alphaParameter(n) = best_params(1);
    kappaParameter(n) = best_params(2);
    sigmaParameter(n) = best_params(3);
    nll_RLsigma_subjectivePE_noIntegration(n) = best_nll;
end
params_RLsigma_subjectivePE_noIntegration.alpha = alphaParameter;
params_RLsigma_subjectivePE_noIntegration.sigma = sigmaParameter;
params_RLsigma_subjectivePE_noIntegration.kappa = kappaParameter;
safe_saveall(fullfile(save_dir, 'params_RLsigma_subjectivePE_noIntegration_Both_multiSP.mat'), params_RLsigma_subjectivePE_noIntegration);
safe_saveall(fullfile(save_dir, 'nll_RLsigma_subjectivePE_noIntegration_Both_multiSP.mat'), nll_RLsigma_subjectivePE_noIntegration);
end

%% ----------------- Perceptual condition: fitting -------------------------
%% --- Basic RL model, subjective PE, no integration (Perceptual) ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_basicRL_subjectivePE_noIntegration = NaN(numSubjs, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Basic RL subjective PE, no integration (Perceptual)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Basic RL subjective PE, no integration (Perceptual)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_pointObs_subjectivePE(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,3);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp)];
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]);
    end

    alphaParameter(n) = best_params(1);
    kappaParameter(n) = best_params(2);
    sigmaParameter(n) = best_params(3);
    nll_basicRL_subjectivePE_noIntegration(n) = best_nll;
end
params_basicRL_subjectivePE_noIntegration.alpha = alphaParameter;
params_basicRL_subjectivePE_noIntegration.kappa = kappaParameter;
params_basicRL_subjectivePE_noIntegration.sigma = sigmaParameter;
safe_saveall(fullfile(save_dir, 'params_basicRL_subjectivePE_noIntegration_Perceptual_multiSP.mat'), params_basicRL_subjectivePE_noIntegration);
safe_saveall(fullfile(save_dir, 'nll_basicRL_subjectivePE_noIntegration_Perceptual_multiSP.mat'), nll_basicRL_subjectivePE_noIntegration);
end

%% --- RL + Est. Sensitivity model, subjective PE, no integration (Perceptual) ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
nll_RLsigma_subjectivePE_noIntegration = NaN(numSubjs, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'RL + Est. Sensitivity subjective PE, no integration (Perceptual)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'RL + Est. Sensitivity subjective PE, no integration (Perceptual)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_subjectivePE(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,3);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp)];
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]);
    end

    alphaParameter(n) = best_params(1);
    kappaParameter(n) = best_params(2);
    sigmaParameter(n) = best_params(3);
    nll_RLsigma_subjectivePE_noIntegration(n) = best_nll;
end
params_RLsigma_subjectivePE_noIntegration.alpha = alphaParameter;
params_RLsigma_subjectivePE_noIntegration.sigma = sigmaParameter;
params_RLsigma_subjectivePE_noIntegration.kappa = kappaParameter;
safe_saveall(fullfile(save_dir, 'params_RLsigma_subjectivePE_noIntegration_Perceptual_multiSP.mat'), params_RLsigma_subjectivePE_noIntegration);
safe_saveall(fullfile(save_dir, 'nll_RLsigma_subjectivePE_noIntegration_Perceptual_multiSP.mat'), nll_RLsigma_subjectivePE_noIntegration);
end

fitSlider_ALLmodels.progress_bar('close'); % done fitting -- close the waitbar window (safe no-op if run_fitting was false or nothing above ran)

%% ----------------- Model comparison ---------------------------------------
%% --- Both condition ---
nll_basicRL_subjectivePE_noIntegration = importdata(fullfile(save_dir, "nll_basicRL_subjectivePE_noIntegration_Both_multiSP.mat"));
nll_RLsigma_subjectivePE_noIntegration = importdata(fullfile(save_dir, "nll_RLsigma_subjectivePE_noIntegration_Both_multiSP.mat"));
params_basicRL_subjectivePE_noIntegration = importdata(fullfile(save_dir, "params_basicRL_subjectivePE_noIntegration_Both_multiSP.mat"));
params_RLsigma_subjectivePE_noIntegration = importdata(fullfile(save_dir, "params_RLsigma_subjectivePE_noIntegration_Both_multiSP.mat"));

disp('Computing AIC and BIC for subjective PE, no-integration models...');
num_trials_both = NaN(numSubjs, 1);
for n = 1:numSubjs
    num_trials_both(n) = length(preprocess_fitSlider(dataBoth, uniqueID(n)).mu_hat);
end
num_params_subjectivePE_noIntegration = [3, 3];
model_names_subjectivePE_noIntegration = {'basicRL_subjectivePE_noIntegration','RLsigma_subjectivePE_noIntegration'};
[AICBIC_subjectivePE_noIntegration_table, bestModel_subjectivePE_noIntegration_table, BMS_subjectivePE_noIntegration_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_subjectivePE_noIntegration, nll_RLsigma_subjectivePE_noIntegration], model_names_subjectivePE_noIntegration, num_params_subjectivePE_noIntegration, num_trials_both, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_subjectivePE_noIntegration_Both_reducedMS_multiSP.mat'), AICBIC_subjectivePE_noIntegration_table);
safe_saveall(fullfile(save_dir, 'bestModel_subjectivePE_noIntegration_Both_multiSP.mat'), bestModel_subjectivePE_noIntegration_table);
disp('Subjective PE, no-integration model AIC/BIC estimated and saved.');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_subjectivePE_noIntegration_table, bestModel_subjectivePE_noIntegration_table, BMS_subjectivePE_noIntegration_table, model_names_subjectivePE_noIntegration, 'Both condition, Subjective PE, No Integration');

bic_ax_both_subj_noInt = fitSlider_ALLmodels.plot_fitted_params({params_basicRL_subjectivePE_noIntegration, params_RLsigma_subjectivePE_noIntegration}, model_names_subjectivePE_noIntegration, AICBIC_subjectivePE_noIntegration_table, 'Both condition, Subjective PE, No Integration', 'sigma (fitted)');

%% --- Perceptual condition ---
nll_basicRL_subjectivePE_noIntegration = importdata(fullfile(save_dir, "nll_basicRL_subjectivePE_noIntegration_Perceptual_multiSP.mat"));
nll_RLsigma_subjectivePE_noIntegration = importdata(fullfile(save_dir, "nll_RLsigma_subjectivePE_noIntegration_Perceptual_multiSP.mat"));
params_basicRL_subjectivePE_noIntegration = importdata(fullfile(save_dir, "params_basicRL_subjectivePE_noIntegration_Perceptual_multiSP.mat"));
params_RLsigma_subjectivePE_noIntegration = importdata(fullfile(save_dir, "params_RLsigma_subjectivePE_noIntegration_Perceptual_multiSP.mat"));

disp('Computing AIC and BIC for subjective PE, no-integration models (Perceptual)...');
num_trials_perceptual = NaN(numSubjs, 1);
for n = 1:numSubjs
    num_trials_perceptual(n) = length(preprocess_fitSlider(dataPerceptual, uniqueID(n)).mu_hat);
end
num_params_subjectivePE_noIntegration = [3, 3];
model_names_subjectivePE_noIntegration = {'basicRL_subjectivePE_noIntegration','RLsigma_subjectivePE_noIntegration'};
[AICBIC_subjectivePE_noIntegration_table, bestModel_subjectivePE_noIntegration_table, BMS_subjectivePE_noIntegration_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_subjectivePE_noIntegration, nll_RLsigma_subjectivePE_noIntegration], model_names_subjectivePE_noIntegration, num_params_subjectivePE_noIntegration, num_trials_perceptual, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_subjectivePE_noIntegration_Perceptual_reducedMS_multiSP.mat'), AICBIC_subjectivePE_noIntegration_table);
safe_saveall(fullfile(save_dir, 'bestModel_subjectivePE_noIntegration_Perceptual_multiSP.mat'), bestModel_subjectivePE_noIntegration_table);
disp('Subjective PE, no-integration model AIC/BIC estimated and saved (Perceptual).');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_subjectivePE_noIntegration_table, bestModel_subjectivePE_noIntegration_table, BMS_subjectivePE_noIntegration_table, model_names_subjectivePE_noIntegration, 'Perceptual condition, Subjective PE, No Integration');

bic_ax_perc_subj_noInt = fitSlider_ALLmodels.plot_fitted_params({params_basicRL_subjectivePE_noIntegration, params_RLsigma_subjectivePE_noIntegration}, model_names_subjectivePE_noIntegration, AICBIC_subjectivePE_noIntegration_table, 'Perceptual condition, Subjective PE, No Integration', 'sigma (fitted)');

%% ========================================================================
%  CROSS-SECTION COMPARISON PLOTS
%  ------------------------------------------------------------------------
%  Everything below compares results ACROSS the four sections above, and
%  always runs (all four sections' outputs always exist, since there are
%  no selector toggles).
%  ========================================================================

%% =================== MATCH SUM-BIC Y-AXES ACROSS ALL PLOTS ==============
% plot_fitted_params auto-scales each "Sum of BIC" bar plot to its own
% data, which makes Both vs Perceptual (or objective vs subjective PE, or
% integrated vs no-integration) visually misleading to compare side by
% side. Equalize all eight to the widest range needed.
bic_axes = [bic_ax_both_obj, bic_ax_perc_obj, ...
            bic_ax_both_subj, bic_ax_perc_subj, ...
            bic_ax_both_obj_noInt, bic_ax_perc_obj_noInt, ...
            bic_ax_both_subj_noInt, bic_ax_perc_subj_noInt];
bic_ylims = cell2mat(arrayfun(@(ax) ylim(ax), bic_axes, 'UniformOutput', false)');
shared_bic_ylim = [min(bic_ylims(:,1)), max(bic_ylims(:,2))];
for i = 1:length(bic_axes)
    ylim(bic_axes(i), shared_bic_ylim);
end

%% ========================================================================
%  OBJECTIVE VS SUBJECTIVE PE PARAMETER COMPARISON
%  ------------------------------------------------------------------------
%  Scatters each subject's fitted alpha/kappa from the subjective-PE fit
%  (Section 2) against the same model's objective-PE fit (Section 1).
%  Only basicRL and RLsigma have a subjective-PE variant (bayesianAgent
%  has none), so this is a 2 (model) x 2 (parameter) grid, one combined
%  figure per condition.
%  Reloaded from disk (not reused in-memory) for the same reason as every
%  other params_* reload in this script.
%  ========================================================================
conditions_pe_compare = {'Both', 'Perceptual'};
model_names_pe_compare = {'basicRL', 'RLsigma'};
param_names_pe_compare = {'alpha', 'kappa'};

for c = 1:length(conditions_pe_compare)
    cond_label = conditions_pe_compare{c};

    params_obj = { ...
        importdata(fullfile(save_dir, sprintf('params_basicRL_RBVoi_%s_multiSP.mat', cond_label))), ...
        importdata(fullfile(save_dir, sprintf('params_RLSigma_RBVoi_%s_multiSP.mat', cond_label)))};
    params_subj = { ...
        importdata(fullfile(save_dir, sprintf('params_basicRL_subjectivePEv2_%s_multiSP.mat', cond_label))), ...
        importdata(fullfile(save_dir, sprintf('params_RLsigma_subjectivePEv2_%s_multiSP.mat', cond_label)))};

    figure('Position', [100, 100, 1200, 700]);
    sgtitle(sprintf('Objective vs Subjective PE: Fitted Parameters (%s condition)', cond_label));

    for m = 1:length(model_names_pe_compare)
        for p = 1:length(param_names_pe_compare)
            subplot(length(model_names_pe_compare), length(param_names_pe_compare), (m-1)*length(param_names_pe_compare) + p);
            x = params_obj{m}.(param_names_pe_compare{p});
            y = params_subj{m}.(param_names_pe_compare{p});
            fitSlider_ALLmodels.plot_scatter_spearman(x, y, ...
                sprintf('%s (objective PE)', param_names_pe_compare{p}), ...
                sprintf('%s (subjective PE)', param_names_pe_compare{p}), ...
                sprintf('%s: %s', model_names_pe_compare{m}, param_names_pe_compare{p}));
        end
    end
end

%% ========================================================================
%  INTEGRATED VS NO-INTEGRATION PARAMETER COMPARISON
%  ------------------------------------------------------------------------
%  Scatters each subject's fitted alpha/kappa from the no-integration
%  (point-observation) fit (Sections 3/4) against the same model's
%  integrated fit (Sections 1/2), for basicRL and RLsigma -- a 2 (model) x
%  2 (parameter) grid, one combined figure per condition per PE variant
%  (objective, subjective).
%  Reloaded from disk (not reused in-memory) for the same reason as every
%  other params_* reload in this script.
%  ========================================================================
conditions_int_compare = {'Both', 'Perceptual'};
model_names_int_compare = {'basicRL', 'RLsigma'};
param_names_int_compare = {'alpha', 'kappa'};

for c = 1:length(conditions_int_compare)
    cond_label = conditions_int_compare{c};

    % --- Objective PE: integrated vs no-integration ---
    params_integrated = { ...
        importdata(fullfile(save_dir, sprintf('params_basicRL_RBVoi_%s_multiSP.mat', cond_label))), ...
        importdata(fullfile(save_dir, sprintf('params_RLSigma_RBVoi_%s_multiSP.mat', cond_label)))};
    params_noInt = { ...
        importdata(fullfile(save_dir, sprintf('params_basicRL_noIntegration_%s_multiSP.mat', cond_label))), ...
        importdata(fullfile(save_dir, sprintf('params_RLsigma_noIntegration_%s_multiSP.mat', cond_label)))};

    figure('Position', [100, 100, 1200, 700]);
    sgtitle(sprintf('Integrated vs No-Integration: Fitted Parameters, Objective PE (%s condition)', cond_label));

    for m = 1:length(model_names_int_compare)
        for p = 1:length(param_names_int_compare)
            subplot(length(model_names_int_compare), length(param_names_int_compare), (m-1)*length(param_names_int_compare) + p);
            x = params_integrated{m}.(param_names_int_compare{p});
            y = params_noInt{m}.(param_names_int_compare{p});
            fitSlider_ALLmodels.plot_scatter_spearman(x, y, ...
                sprintf('%s (integrated)', param_names_int_compare{p}), ...
                sprintf('%s (no integration)', param_names_int_compare{p}), ...
                sprintf('%s: %s', model_names_int_compare{m}, param_names_int_compare{p}));
        end
    end

    % --- Subjective PE: integrated vs no-integration ---
    params_integrated_subj = { ...
        importdata(fullfile(save_dir, sprintf('params_basicRL_subjectivePEv2_%s_multiSP.mat', cond_label))), ...
        importdata(fullfile(save_dir, sprintf('params_RLsigma_subjectivePEv2_%s_multiSP.mat', cond_label)))};
    params_noInt_subj = { ...
        importdata(fullfile(save_dir, sprintf('params_basicRL_subjectivePE_noIntegration_%s_multiSP.mat', cond_label))), ...
        importdata(fullfile(save_dir, sprintf('params_RLsigma_subjectivePE_noIntegration_%s_multiSP.mat', cond_label)))};

    figure('Position', [100, 100, 1200, 700]);
    sgtitle(sprintf('Integrated vs No-Integration: Fitted Parameters, Subjective PE (%s condition)', cond_label));

    for m = 1:length(model_names_int_compare)
        for p = 1:length(param_names_int_compare)
            subplot(length(model_names_int_compare), length(param_names_int_compare), (m-1)*length(param_names_int_compare) + p);
            x = params_integrated_subj{m}.(param_names_int_compare{p});
            y = params_noInt_subj{m}.(param_names_int_compare{p});
            fitSlider_ALLmodels.plot_scatter_spearman(x, y, ...
                sprintf('%s (integrated)', param_names_int_compare{p}), ...
                sprintf('%s (no integration)', param_names_int_compare{p}), ...
                sprintf('%s: %s (subjective PE)', model_names_int_compare{m}, param_names_int_compare{p}));
        end
    end
end
