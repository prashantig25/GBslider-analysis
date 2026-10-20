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
%  A fifth section follows the cross-section comparison plots: basicRL and
%  RLsigma (subjective PE, free sigma), both integrated and no-integration,
%  refit to the normative Bayesian Agent's own WITH-integration simulations
%  (Agent/script_sims.m's data_agent_condition{1,2}_pupil0.06.txt) instead
%  of real subject data -- see that section's header for the rationale.
%  A sixth section fits a basicRL/RLsigma mixture model (free mixture
%  weight lambda, GbAgent.py-style -- see fitSlider_ALLmodels.m's
%  nll_mixture_subjectivePE/nll_mixture_integrated_subjectivePE headers)
%  to real subject data, both integrated and no-integration, and runs a
%  3-way comparison against Section 2's/4's pure basicRL_subjectivePE and
%  RLsigma_subjectivePE fits.
%  bayesianAgent appears in Sections 1 and 3 (objective PE, integrated and
%  no-integration respectively) but not 2/4 -- it has no subjective-PE
%  variant (its belief update isn't a simple alpha-driven delta rule). It
%  does have a no-integration variant (nll_bayesianAgent_pointObs), since
%  its belief update turns out to route through the same set_o grid-
%  integration mechanism internally (Agent.learn() -> Agent.integrate_voi
%  when task_agent_analysis==1) that Sections 3/4 otherwise bypass for
%  basicRL/RLsigma -- see nll_bayesianAgent_pointObs's header in
%  fitSlider_ALLmodels.m for the full rationale.
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

% Recompute desiredPath/save_dir fresh so this section works correctly
% even when run standalone via Run Section, without having first run the
% script's own top -- other scripts in this repo reuse the variable name
% save_dir for different folders, so a stale value can otherwise linger
% in the workspace.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');

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
% Recompute desiredPath/save_dir fresh -- robust to jumping here directly
% via Run Section (a separate cell from the section's top above), see
% Section 1's top-of-section block for the full rationale.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');
% Recompute dataBoth/dataPerceptual/uniqueID/numSubjs fresh too (cheap --
% a single ~1MB file read), since this is a separate Run-Section cell from
% the top-level "LOAD AND PREPARE DATA" cell that normally defines them.
[dataBoth, dataPerceptual, uniqueID, numSubjs] = fitSlider_ALLmodels.load_fitting_data();

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
% Recompute save_dir/dataBoth/dataPerceptual/uniqueID/numSubjs fresh too
% -- robust to jumping here directly via Run Section, see Section 1's
% Both-condition Model comparison cell for the full rationale.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');
[dataBoth, dataPerceptual, uniqueID, numSubjs] = fitSlider_ALLmodels.load_fitting_data();

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

% Recompute desiredPath/save_dir fresh -- see Section 1's identical block
% for why.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');

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
% Recompute desiredPath/save_dir fresh -- robust to jumping here directly
% via Run Section, see Section 1's top-of-section block for the rationale.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');
% Recompute dataBoth/dataPerceptual/uniqueID/numSubjs fresh too -- see
% Section 1's Model comparison cell for why.
[dataBoth, dataPerceptual, uniqueID, numSubjs] = fitSlider_ALLmodels.load_fitting_data();

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
% Recompute save_dir/dataBoth/dataPerceptual/uniqueID/numSubjs fresh too
% -- see Section 1's Both-condition Model comparison cell for the rationale.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');
[dataBoth, dataPerceptual, uniqueID, numSubjs] = fitSlider_ALLmodels.load_fitting_data();

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
%  Same basicRL / RLsigma / bayesianAgent models as Section 1, but using
%  each model's point-observation (non-integrated) likelihood instead of
%  its integrated one: nll_basicRL_pointObs instead of
%  nll_basicRL_integrated, nll_RLsigma instead of nll_RLsigma_VOI, and
%  nll_bayesianAgent_pointObs instead of nll_bayesianAgent. Belief each
%  trial comes from a single point estimate of the observation
%  (normcdf(condiff(n), 0, sigma) for basicRL/RLsigma; Agent.p_s_giv_o's
%  own point-observation formula for bayesianAgent, with
%  agent.task_agent_analysis set to 0 instead of 1 so Agent.learn() skips
%  its internal set_o grid-integration step too) rather than integrated
%  over a discretized grid of hypothetical observations -- see fitSlider_
%  ALLmodels.m's nll_basicRL_pointObs and nll_bayesianAgent_pointObs
%  headers for the full rationale. Unlike Sections 2/4 (which exclude
%  bayesianAgent entirely, since it has no subjective-PE variant), this
%  section is a 3-model comparison with its own family-wise step, exactly
%  mirroring Section 1's structure.
%  ========================================================================

% Recompute desiredPath/save_dir fresh -- see Section 1's identical block
% for why.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');

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

%% --- Bayesian Agent model, no integration ---
if run_fitting
sigmaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
nll_bayesianAgent_noIntegration = NaN(numSubjs, 1);
lb = [1, 0];
ub = [100, 0.1];
initKappa = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Bayesian Agent, no integration (Both)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Bayesian Agent, no integration (Both)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataBoth, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_bayesianAgent_pointObs(params, subj.mu_hat, subj.dataTable, ...
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
    nll_bayesianAgent_noIntegration(n) = best_nll;
end
params_bayesianAgent_noIntegration.sigma = sigmaParameter;
params_bayesianAgent_noIntegration.kappa = kappaParameter;
safe_saveall(fullfile(save_dir, 'params_bayesianAgent_noIntegration_Both_multiSP.mat'), params_bayesianAgent_noIntegration);
safe_saveall(fullfile(save_dir, 'nll_bayesianAgent_noIntegration_Both_multiSP.mat'), nll_bayesianAgent_noIntegration);
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

%% --- Bayesian Agent model, no integration (Perceptual) ---
if run_fitting
sigmaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
nll_bayesianAgent_noIntegration = NaN(numSubjs, 1);
lb = [1, 0];
ub = [100, 0.1];
initKappa = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Bayesian Agent, no integration (Perceptual)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Bayesian Agent, no integration (Perceptual)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_bayesianAgent_pointObs(params, subj.mu_hat, subj.dataTable, ...
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
    nll_bayesianAgent_noIntegration(n) = best_nll;
end
params_bayesianAgent_noIntegration.sigma = sigmaParameter;
params_bayesianAgent_noIntegration.kappa = kappaParameter;
safe_saveall(fullfile(save_dir, 'params_bayesianAgent_noIntegration_Perceptual_multiSP.mat'), params_bayesianAgent_noIntegration);
safe_saveall(fullfile(save_dir, 'nll_bayesianAgent_noIntegration_Perceptual_multiSP.mat'), nll_bayesianAgent_noIntegration);
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
% Recompute desiredPath/save_dir fresh -- robust to jumping here directly
% via Run Section, see Section 1's top-of-section block for the rationale.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');
% Recompute dataBoth/dataPerceptual/uniqueID/numSubjs fresh too -- see
% Section 1's Model comparison cell for why.
[dataBoth, dataPerceptual, uniqueID, numSubjs] = fitSlider_ALLmodels.load_fitting_data();

%% --- Both condition ---
nll_basicRL_noIntegration = importdata(fullfile(save_dir, "nll_basicRL_noIntegration_Both_multiSP.mat"));
nll_RLsigma_noIntegration = importdata(fullfile(save_dir, "nll_RLsigma_noIntegration_Both_multiSP.mat"));
nll_bayesianAgent_noIntegration = importdata(fullfile(save_dir, "nll_bayesianAgent_noIntegration_Both_multiSP.mat"));
params_basicRL_noIntegration = importdata(fullfile(save_dir, "params_basicRL_noIntegration_Both_multiSP.mat"));
params_RLsigma_noIntegration = importdata(fullfile(save_dir, "params_RLsigma_noIntegration_Both_multiSP.mat"));
params_bayesianAgent_noIntegration = importdata(fullfile(save_dir, "params_bayesianAgent_noIntegration_Both_multiSP.mat"));

disp('Computing AIC and BIC for no-integration models...');
num_trials_both = NaN(numSubjs, 1);
for n = 1:numSubjs
    num_trials_both(n) = length(preprocess_fitSlider(dataBoth, uniqueID(n)).mu_hat);
end
num_params_noIntegration = [3, 3, 2]; % [basicRL, RLsigma, BayesianAgent] -- sigma is free here, matching Section 1
model_names_noIntegration = {'basicRL_noIntegration','RLsigma_noIntegration','BayesianAgent_noIntegration'};
[AICBIC_noIntegration_table, bestModel_noIntegration_table, BMS_noIntegration_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_noIntegration, nll_RLsigma_noIntegration, nll_bayesianAgent_noIntegration], model_names_noIntegration, num_params_noIntegration, num_trials_both, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_noIntegration_Both_reducedMS_multiSP.mat'), AICBIC_noIntegration_table);
safe_saveall(fullfile(save_dir, 'bestModel_noIntegration_Both_multiSP.mat'), bestModel_noIntegration_table);
disp('No-integration model AIC/BIC estimated and saved.');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_noIntegration_table, bestModel_noIntegration_table, BMS_noIntegration_table, model_names_noIntegration, 'Both condition, No Integration');

bic_ax_both_obj_noInt = fitSlider_ALLmodels.plot_fitted_params({params_basicRL_noIntegration, params_RLsigma_noIntegration, params_bayesianAgent_noIntegration}, model_names_noIntegration, AICBIC_noIntegration_table, 'Both condition, No Integration', 'sigma (fitted)');

% Family-wise Model Comparison using VBA Toolbox (see Section 1 for the
% full rationale): Basic RL alone vs. the two LR-modulation models
% (RLsigma, BayesianAgent), both no-integration.
BIC_matrix = [AICBIC_noIntegration_table.BIC_basicRL_noIntegration, ...
                  AICBIC_noIntegration_table.BIC_RLsigma_noIntegration, ...
                  AICBIC_noIntegration_table.BIC_BayesianAgent_noIntegration];
lme = -0.5 * BIC_matrix;
family_names = {'Basic_RL', 'LR_Modulation'};
families{1} = [1];
families{2} = [2, 3];

fprintf('Running family-wise model comparison (no integration)...\n');
[results_summary, posterior, out] = fitSlider_ALLmodels.compute_family_model_comparison(lme, families, family_names, model_names_noIntegration);
save(fullfile(save_dir, 'family_wise_BMS_results_noIntegration_both_multiSP.mat'), 'results_summary', 'posterior', 'out');
fprintf('\nResults saved to: family_wise_BMS_results_noIntegration_both_multiSP.mat\n');

fitSlider_ALLmodels.plot_family_model_comparison(results_summary, 'Both condition, No Integration');

%% --- Perceptual condition ---
% Recompute save_dir/dataBoth/dataPerceptual/uniqueID/numSubjs fresh too
% -- see Section 1's Both-condition Model comparison cell for the rationale.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');
[dataBoth, dataPerceptual, uniqueID, numSubjs] = fitSlider_ALLmodels.load_fitting_data();

nll_basicRL_noIntegration = importdata(fullfile(save_dir, "nll_basicRL_noIntegration_Perceptual_multiSP.mat"));
nll_RLsigma_noIntegration = importdata(fullfile(save_dir, "nll_RLsigma_noIntegration_Perceptual_multiSP.mat"));
nll_bayesianAgent_noIntegration = importdata(fullfile(save_dir, "nll_bayesianAgent_noIntegration_Perceptual_multiSP.mat"));
params_basicRL_noIntegration = importdata(fullfile(save_dir, "params_basicRL_noIntegration_Perceptual_multiSP.mat"));
params_RLsigma_noIntegration = importdata(fullfile(save_dir, "params_RLsigma_noIntegration_Perceptual_multiSP.mat"));
params_bayesianAgent_noIntegration = importdata(fullfile(save_dir, "params_bayesianAgent_noIntegration_Perceptual_multiSP.mat"));

disp('Computing AIC and BIC for no-integration models (Perceptual)...');
num_trials_perceptual = NaN(numSubjs, 1);
for n = 1:numSubjs
    num_trials_perceptual(n) = length(preprocess_fitSlider(dataPerceptual, uniqueID(n)).mu_hat);
end
num_params_noIntegration = [3, 3, 2];
model_names_noIntegration = {'basicRL_noIntegration','RLsigma_noIntegration','BayesianAgent_noIntegration'};
[AICBIC_noIntegration_table, bestModel_noIntegration_table, BMS_noIntegration_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_noIntegration, nll_RLsigma_noIntegration, nll_bayesianAgent_noIntegration], model_names_noIntegration, num_params_noIntegration, num_trials_perceptual, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_noIntegration_Perceptual_reducedMS_multiSP.mat'), AICBIC_noIntegration_table);
safe_saveall(fullfile(save_dir, 'bestModel_noIntegration_Perceptual_multiSP.mat'), bestModel_noIntegration_table);
disp('No-integration model AIC/BIC estimated and saved (Perceptual).');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_noIntegration_table, bestModel_noIntegration_table, BMS_noIntegration_table, model_names_noIntegration, 'Perceptual condition, No Integration');

bic_ax_perc_obj_noInt = fitSlider_ALLmodels.plot_fitted_params({params_basicRL_noIntegration, params_RLsigma_noIntegration, params_bayesianAgent_noIntegration}, model_names_noIntegration, AICBIC_noIntegration_table, 'Perceptual condition, No Integration', 'sigma (fitted)');

BIC_matrix = [AICBIC_noIntegration_table.BIC_basicRL_noIntegration, ...
                  AICBIC_noIntegration_table.BIC_RLsigma_noIntegration, ...
                  AICBIC_noIntegration_table.BIC_BayesianAgent_noIntegration];
lme = -0.5 * BIC_matrix;
family_names = {'Basic_RL', 'LR_Modulation'};
families{1} = [1];
families{2} = [2, 3];

fprintf('Running family-wise model comparison (no integration, perceptual condition)...\n');
[results_summary, posterior, out] = fitSlider_ALLmodels.compute_family_model_comparison(lme, families, family_names, model_names_noIntegration);
save(fullfile(save_dir, 'family_wise_BMS_results_noIntegration_perceptual_multiSP.mat'), 'results_summary', 'posterior', 'out');
fprintf('\nResults saved to: family_wise_BMS_results_noIntegration_perceptual_multiSP.mat\n');

fitSlider_ALLmodels.plot_family_model_comparison(results_summary, 'Perceptual condition, No Integration');

%% ========================================================================
%  SECTION 4 OF 4: SUBJECTIVE PE, NO INTEGRATION
%  ------------------------------------------------------------------------
%  Combines Section 2's subjective-PE anchoring (mu_hat(n-1)) with Section
%  3's point-observation belief: nll_basicRL_pointObs_subjectivePE instead
%  of nll_basicRL_integrated_subjectivePE, nll_RLsigma_subjectivePE
%  instead of nll_RLsigma_VOI_subjectivePE. bayesianAgent is excluded (see
%  Sections 2 and 3 above) -- a 2-model comparison throughout.
%  ========================================================================

% Recompute desiredPath/save_dir fresh -- see Section 1's identical block
% for why.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');

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
% Recompute desiredPath/save_dir fresh -- robust to jumping here directly
% via Run Section, see Section 1's top-of-section block for the rationale.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');
% Recompute dataBoth/dataPerceptual/uniqueID/numSubjs fresh too -- see
% Section 1's Model comparison cell for why.
[dataBoth, dataPerceptual, uniqueID, numSubjs] = fitSlider_ALLmodels.load_fitting_data();

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
% Recompute save_dir/dataBoth/dataPerceptual/uniqueID/numSubjs fresh too
% -- see Section 1's Both-condition Model comparison cell for the rationale.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');
[dataBoth, dataPerceptual, uniqueID, numSubjs] = fitSlider_ALLmodels.load_fitting_data();

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

%% ========================================================================
%  SECTION 5: AGENT SIMULATIONS (WITH INTEGRATION), SUBJECTIVE PE, FREE
%  SIGMA -- BASIC RL & RL SIGMA, WITH AND WITHOUT INTEGRATION
%  ------------------------------------------------------------------------
%  Fits basicRL and RLsigma (subjective PE, free sigma -- same two models
%  as Sections 2/4 above) to data generated by the normative Bayesian
%  Agent WITH observation integration (task_agent_analysis = 1 --
%  Agent/script_sims.m's data_agent_condition1_pupil0.06.txt /
%  data_agent_condition2_pupil0.06.txt), instead of real subject data.
%  Agent/script_sims.m generates 300 independent simulations per
%  condition; only the first 100 (numSubjs_agent below) are fit here for
%  speed, each treated as one "subject" for fitting purposes. Both the integrated
%  (nll_basicRL_integrated_subjectivePE / nll_RLsigma_VOI_subjectivePE)
%  and no-integration (nll_basicRL_pointObs_subjectivePE /
%  nll_RLsigma_subjectivePE) likelihoods are fit to this SAME
%  agent-generated dataset, so the two model comparisons below directly
%  test whether basic RL's integration-driven "free" confidence-weighting
%  (see conversation re: integration giving basicRL's grid-marginalization
%  an implicit, emergent confidence-scaled learning rate even without an
%  explicit belief-weighting term) shows up even when the ground-truth
%  generative process is the Agent itself.
%  mu_hat here is the agent's own tracked belief (data.mu = agent.G), and
%  reward is recoded via recode_rewards_choice (not recode_rewards) since
%  this is simulated data with no real-data presentation-flip artifact to
%  correct for -- same convention as recovery_ReducedModelSpace.m's
%  refits of simulated choice data (see fitSlider_ALLmodels.
%  recode_rewards_choice's header).
%  ========================================================================

% Recompute desiredPath/save_dir fresh -- see Section 1's identical block
% for why.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');

%% =================== LOAD AGENT SIMULATIONS (WITH INTEGRATION) ==========
agent_data_dir = fullfile(desiredPath, 'Data', 'agent simulations');
trials_per_block_agent = 25; % matches Agent/script_sims.m's num_trials
blocks_per_sim_agent = 4;    % matches Agent/script_sims.m's num_blocks
trials_per_sim_agent = trials_per_block_agent * blocks_per_sim_agent; % 100
numSubjs_agent = 100; % subset of Agent/script_sims.m's 300 simulated "subjects" per condition, for faster fitting
uniqueID_agent = (1:numSubjs_agent)';
num_trials_agent = repmat(trials_per_sim_agent, numSubjs_agent, 1); % every simulation has exactly 100 trials, no missing data

agent_data_files = {'data_agent_condition1_pupil0.06.txt', 'data_agent_condition2_pupil0.06.txt'}; % condition 1 = Both, condition 2 = Perceptual

% Each simulation is trials_per_sim_agent contiguous rows (confirmed via
% the trials column resetting every 25 rows), so keeping only the first
% numSubjs_agent*trials_per_sim_agent rows keeps exactly the first
% numSubjs_agent simulations -- both the data table and sim_id_agent
% below must be truncated together so sim_id_agent stays the same height
% as data_agent for the == uniqueID_agent(n) indexing in the fitting loops.
data_agent_Both = readtable(fullfile(agent_data_dir, agent_data_files{1}));
data_agent_Both = data_agent_Both(1:numSubjs_agent*trials_per_sim_agent, :);
sim_id_agent_Both = repelem(uniqueID_agent, trials_per_sim_agent);
data_agent_Perceptual = readtable(fullfile(agent_data_dir, agent_data_files{2}));
data_agent_Perceptual = data_agent_Perceptual(1:numSubjs_agent*trials_per_sim_agent, :);
sim_id_agent_Perceptual = repelem(uniqueID_agent, trials_per_sim_agent);

agent_data_by_cond = {data_agent_Both, data_agent_Perceptual};
sim_id_by_cond = {sim_id_agent_Both, sim_id_agent_Perceptual};

%% ----------------- Both condition: fitting ------------------------------
%% --- Basic RL model, subjective PE, integrated, agent simulations ---
if run_fitting
data_agent = agent_data_by_cond{1};
sim_id_agent = sim_id_by_cond{1};
alphaParameter = NaN(numSubjs_agent, 1);
kappaParameter = NaN(numSubjs_agent, 1);
sigmaParameter = NaN(numSubjs_agent, 1);
nll_basicRL_subjectivePE_agentSim = NaN(numSubjs_agent, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs_agent, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs_agent, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs_agent, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs_agent, n_startingPoints, 'Basic RL subjective PE, agent sim, integrated (Both)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs_agent, n_startingPoints, 'Basic RL subjective PE, agent sim, integrated (Both)'));
parfor n = 1:numSubjs_agent
    subj_data = data_agent(sim_id_agent == uniqueID_agent(n), :);
    mu_hat = subj_data.mu;
    blocks = cumsum(subj_data.trials == 1);
    condiff = subj_data.contrast_diff;
    recoded_reward = fitSlider_ALLmodels.recode_rewards_choice(subj_data.reward, subj_data.action);
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated_subjectivePE(params, mu_hat, blocks, recoded_reward, condiff);
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
    nll_basicRL_subjectivePE_agentSim(n) = best_nll;
end
params_basicRL_subjectivePE_agentSim.alpha = alphaParameter;
params_basicRL_subjectivePE_agentSim.kappa = kappaParameter;
params_basicRL_subjectivePE_agentSim.sigma = sigmaParameter;
safe_saveall(fullfile(save_dir, 'params_basicRL_subjectivePE_agentSim_Both_multiSP.mat'), params_basicRL_subjectivePE_agentSim);
safe_saveall(fullfile(save_dir, 'nll_basicRL_subjectivePE_agentSim_Both_multiSP.mat'), nll_basicRL_subjectivePE_agentSim);
end

%% --- RL + Est. Sensitivity model, subjective PE, integrated, agent simulations ---
if run_fitting
data_agent = agent_data_by_cond{1};
sim_id_agent = sim_id_by_cond{1};
alphaParameter = NaN(numSubjs_agent, 1);
kappaParameter = NaN(numSubjs_agent, 1);
sigmaParameter = NaN(numSubjs_agent, 1);
nll_RLsigma_subjectivePE_agentSim = NaN(numSubjs_agent, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs_agent, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs_agent, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs_agent, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs_agent, n_startingPoints, 'RL + Est. Sensitivity subjective PE, agent sim, integrated (Both)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs_agent, n_startingPoints, 'RL + Est. Sensitivity subjective PE, agent sim, integrated (Both)'));
parfor n = 1:numSubjs_agent
    subj_data = data_agent(sim_id_agent == uniqueID_agent(n), :);
    mu_hat = subj_data.mu;
    blocks = cumsum(subj_data.trials == 1);
    condiff = subj_data.contrast_diff;
    recoded_reward = fitSlider_ALLmodels.recode_rewards_choice(subj_data.reward, subj_data.action);
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI_subjectivePE(params, mu_hat, blocks, recoded_reward, condiff);
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
    nll_RLsigma_subjectivePE_agentSim(n) = best_nll;
end
params_RLsigma_subjectivePE_agentSim.alpha = alphaParameter;
params_RLsigma_subjectivePE_agentSim.kappa = kappaParameter;
params_RLsigma_subjectivePE_agentSim.sigma = sigmaParameter;
safe_saveall(fullfile(save_dir, 'params_RLsigma_subjectivePE_agentSim_Both_multiSP.mat'), params_RLsigma_subjectivePE_agentSim);
safe_saveall(fullfile(save_dir, 'nll_RLsigma_subjectivePE_agentSim_Both_multiSP.mat'), nll_RLsigma_subjectivePE_agentSim);
end

%% --- Basic RL model, subjective PE, no integration, agent simulations ---
if run_fitting
data_agent = agent_data_by_cond{1};
sim_id_agent = sim_id_by_cond{1};
alphaParameter = NaN(numSubjs_agent, 1);
kappaParameter = NaN(numSubjs_agent, 1);
sigmaParameter = NaN(numSubjs_agent, 1);
nll_basicRL_subjectivePE_noIntegration_agentSim = NaN(numSubjs_agent, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs_agent, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs_agent, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs_agent, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs_agent, n_startingPoints, 'Basic RL subjective PE, agent sim, no integration (Both)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs_agent, n_startingPoints, 'Basic RL subjective PE, agent sim, no integration (Both)'));
parfor n = 1:numSubjs_agent
    subj_data = data_agent(sim_id_agent == uniqueID_agent(n), :);
    mu_hat = subj_data.mu;
    blocks = cumsum(subj_data.trials == 1);
    condiff = subj_data.contrast_diff;
    recoded_reward = fitSlider_ALLmodels.recode_rewards_choice(subj_data.reward, subj_data.action);
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_pointObs_subjectivePE(params, mu_hat, blocks, recoded_reward, condiff);
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
    nll_basicRL_subjectivePE_noIntegration_agentSim(n) = best_nll;
end
params_basicRL_subjectivePE_noIntegration_agentSim.alpha = alphaParameter;
params_basicRL_subjectivePE_noIntegration_agentSim.kappa = kappaParameter;
params_basicRL_subjectivePE_noIntegration_agentSim.sigma = sigmaParameter;
safe_saveall(fullfile(save_dir, 'params_basicRL_subjectivePE_noIntegration_agentSim_Both_multiSP.mat'), params_basicRL_subjectivePE_noIntegration_agentSim);
safe_saveall(fullfile(save_dir, 'nll_basicRL_subjectivePE_noIntegration_agentSim_Both_multiSP.mat'), nll_basicRL_subjectivePE_noIntegration_agentSim);
end

%% --- RL + Est. Sensitivity model, subjective PE, no integration, agent simulations ---
if run_fitting
data_agent = agent_data_by_cond{1};
sim_id_agent = sim_id_by_cond{1};
alphaParameter = NaN(numSubjs_agent, 1);
kappaParameter = NaN(numSubjs_agent, 1);
sigmaParameter = NaN(numSubjs_agent, 1);
nll_RLsigma_subjectivePE_noIntegration_agentSim = NaN(numSubjs_agent, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs_agent, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs_agent, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs_agent, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs_agent, n_startingPoints, 'RL + Est. Sensitivity subjective PE, agent sim, no integration (Both)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs_agent, n_startingPoints, 'RL + Est. Sensitivity subjective PE, agent sim, no integration (Both)'));
parfor n = 1:numSubjs_agent
    subj_data = data_agent(sim_id_agent == uniqueID_agent(n), :);
    mu_hat = subj_data.mu;
    blocks = cumsum(subj_data.trials == 1);
    condiff = subj_data.contrast_diff;
    recoded_reward = fitSlider_ALLmodels.recode_rewards_choice(subj_data.reward, subj_data.action);
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_subjectivePE(params, mu_hat, blocks, recoded_reward, condiff);
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
    nll_RLsigma_subjectivePE_noIntegration_agentSim(n) = best_nll;
end
params_RLsigma_subjectivePE_noIntegration_agentSim.alpha = alphaParameter;
params_RLsigma_subjectivePE_noIntegration_agentSim.kappa = kappaParameter;
params_RLsigma_subjectivePE_noIntegration_agentSim.sigma = sigmaParameter;
safe_saveall(fullfile(save_dir, 'params_RLsigma_subjectivePE_noIntegration_agentSim_Both_multiSP.mat'), params_RLsigma_subjectivePE_noIntegration_agentSim);
safe_saveall(fullfile(save_dir, 'nll_RLsigma_subjectivePE_noIntegration_agentSim_Both_multiSP.mat'), nll_RLsigma_subjectivePE_noIntegration_agentSim);
end

%% ----------------- Perceptual condition: fitting -------------------------
%% --- Basic RL model, subjective PE, integrated, agent simulations (Perceptual) ---
if run_fitting
data_agent = agent_data_by_cond{2};
sim_id_agent = sim_id_by_cond{2};
alphaParameter = NaN(numSubjs_agent, 1);
kappaParameter = NaN(numSubjs_agent, 1);
sigmaParameter = NaN(numSubjs_agent, 1);
nll_basicRL_subjectivePE_agentSim = NaN(numSubjs_agent, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs_agent, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs_agent, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs_agent, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs_agent, n_startingPoints, 'Basic RL subjective PE, agent sim, integrated (Perceptual)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs_agent, n_startingPoints, 'Basic RL subjective PE, agent sim, integrated (Perceptual)'));
parfor n = 1:numSubjs_agent
    subj_data = data_agent(sim_id_agent == uniqueID_agent(n), :);
    mu_hat = subj_data.mu;
    blocks = cumsum(subj_data.trials == 1);
    condiff = subj_data.contrast_diff;
    recoded_reward = fitSlider_ALLmodels.recode_rewards_choice(subj_data.reward, subj_data.action);
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated_subjectivePE(params, mu_hat, blocks, recoded_reward, condiff);
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
    nll_basicRL_subjectivePE_agentSim(n) = best_nll;
end
params_basicRL_subjectivePE_agentSim.alpha = alphaParameter;
params_basicRL_subjectivePE_agentSim.kappa = kappaParameter;
params_basicRL_subjectivePE_agentSim.sigma = sigmaParameter;
safe_saveall(fullfile(save_dir, 'params_basicRL_subjectivePE_agentSim_Perceptual_multiSP.mat'), params_basicRL_subjectivePE_agentSim);
safe_saveall(fullfile(save_dir, 'nll_basicRL_subjectivePE_agentSim_Perceptual_multiSP.mat'), nll_basicRL_subjectivePE_agentSim);
end

%% --- RL + Est. Sensitivity model, subjective PE, integrated, agent simulations (Perceptual) ---
if run_fitting
data_agent = agent_data_by_cond{2};
sim_id_agent = sim_id_by_cond{2};
alphaParameter = NaN(numSubjs_agent, 1);
kappaParameter = NaN(numSubjs_agent, 1);
sigmaParameter = NaN(numSubjs_agent, 1);
nll_RLsigma_subjectivePE_agentSim = NaN(numSubjs_agent, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs_agent, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs_agent, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs_agent, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs_agent, n_startingPoints, 'RL + Est. Sensitivity subjective PE, agent sim, integrated (Perceptual)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs_agent, n_startingPoints, 'RL + Est. Sensitivity subjective PE, agent sim, integrated (Perceptual)'));
parfor n = 1:numSubjs_agent
    subj_data = data_agent(sim_id_agent == uniqueID_agent(n), :);
    mu_hat = subj_data.mu;
    blocks = cumsum(subj_data.trials == 1);
    condiff = subj_data.contrast_diff;
    recoded_reward = fitSlider_ALLmodels.recode_rewards_choice(subj_data.reward, subj_data.action);
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI_subjectivePE(params, mu_hat, blocks, recoded_reward, condiff);
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
    nll_RLsigma_subjectivePE_agentSim(n) = best_nll;
end
params_RLsigma_subjectivePE_agentSim.alpha = alphaParameter;
params_RLsigma_subjectivePE_agentSim.kappa = kappaParameter;
params_RLsigma_subjectivePE_agentSim.sigma = sigmaParameter;
safe_saveall(fullfile(save_dir, 'params_RLsigma_subjectivePE_agentSim_Perceptual_multiSP.mat'), params_RLsigma_subjectivePE_agentSim);
safe_saveall(fullfile(save_dir, 'nll_RLsigma_subjectivePE_agentSim_Perceptual_multiSP.mat'), nll_RLsigma_subjectivePE_agentSim);
end

%% --- Basic RL model, subjective PE, no integration, agent simulations (Perceptual) ---
if run_fitting
data_agent = agent_data_by_cond{2};
sim_id_agent = sim_id_by_cond{2};
alphaParameter = NaN(numSubjs_agent, 1);
kappaParameter = NaN(numSubjs_agent, 1);
sigmaParameter = NaN(numSubjs_agent, 1);
nll_basicRL_subjectivePE_noIntegration_agentSim = NaN(numSubjs_agent, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs_agent, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs_agent, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs_agent, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs_agent, n_startingPoints, 'Basic RL subjective PE, agent sim, no integration (Perceptual)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs_agent, n_startingPoints, 'Basic RL subjective PE, agent sim, no integration (Perceptual)'));
parfor n = 1:numSubjs_agent
    subj_data = data_agent(sim_id_agent == uniqueID_agent(n), :);
    mu_hat = subj_data.mu;
    blocks = cumsum(subj_data.trials == 1);
    condiff = subj_data.contrast_diff;
    recoded_reward = fitSlider_ALLmodels.recode_rewards_choice(subj_data.reward, subj_data.action);
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_pointObs_subjectivePE(params, mu_hat, blocks, recoded_reward, condiff);
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
    nll_basicRL_subjectivePE_noIntegration_agentSim(n) = best_nll;
end
params_basicRL_subjectivePE_noIntegration_agentSim.alpha = alphaParameter;
params_basicRL_subjectivePE_noIntegration_agentSim.kappa = kappaParameter;
params_basicRL_subjectivePE_noIntegration_agentSim.sigma = sigmaParameter;
safe_saveall(fullfile(save_dir, 'params_basicRL_subjectivePE_noIntegration_agentSim_Perceptual_multiSP.mat'), params_basicRL_subjectivePE_noIntegration_agentSim);
safe_saveall(fullfile(save_dir, 'nll_basicRL_subjectivePE_noIntegration_agentSim_Perceptual_multiSP.mat'), nll_basicRL_subjectivePE_noIntegration_agentSim);
end

%% --- RL + Est. Sensitivity model, subjective PE, no integration, agent simulations (Perceptual) ---
if run_fitting
data_agent = agent_data_by_cond{2};
sim_id_agent = sim_id_by_cond{2};
alphaParameter = NaN(numSubjs_agent, 1);
kappaParameter = NaN(numSubjs_agent, 1);
sigmaParameter = NaN(numSubjs_agent, 1);
nll_RLsigma_subjectivePE_noIntegration_agentSim = NaN(numSubjs_agent, 1);
lb = [0, 1, 0];
ub = [1, 100, 0.1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs_agent, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs_agent, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs_agent, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs_agent, n_startingPoints, 'RL + Est. Sensitivity subjective PE, agent sim, no integration (Perceptual)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs_agent, n_startingPoints, 'RL + Est. Sensitivity subjective PE, agent sim, no integration (Perceptual)'));
parfor n = 1:numSubjs_agent
    subj_data = data_agent(sim_id_agent == uniqueID_agent(n), :);
    mu_hat = subj_data.mu;
    blocks = cumsum(subj_data.trials == 1);
    condiff = subj_data.contrast_diff;
    recoded_reward = fitSlider_ALLmodels.recode_rewards_choice(subj_data.reward, subj_data.action);
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_subjectivePE(params, mu_hat, blocks, recoded_reward, condiff);
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
    nll_RLsigma_subjectivePE_noIntegration_agentSim(n) = best_nll;
end
params_RLsigma_subjectivePE_noIntegration_agentSim.alpha = alphaParameter;
params_RLsigma_subjectivePE_noIntegration_agentSim.kappa = kappaParameter;
params_RLsigma_subjectivePE_noIntegration_agentSim.sigma = sigmaParameter;
safe_saveall(fullfile(save_dir, 'params_RLsigma_subjectivePE_noIntegration_agentSim_Perceptual_multiSP.mat'), params_RLsigma_subjectivePE_noIntegration_agentSim);
safe_saveall(fullfile(save_dir, 'nll_RLsigma_subjectivePE_noIntegration_agentSim_Perceptual_multiSP.mat'), nll_RLsigma_subjectivePE_noIntegration_agentSim);
end

fitSlider_ALLmodels.progress_bar('close'); % done fitting -- close the waitbar window (safe no-op if run_fitting was false or nothing above ran)

%% ----------------- Model comparison ---------------------------------------
%  Two separate 2-way comparisons per condition (integrated vs
%  no-integration are NOT compared against each other here -- they use
%  different likelihood functions with the same param count, and this
%  section's purpose is basicRL vs RLsigma WITHIN each integration level,
%  mirroring Sections 2 and 4's real-data comparisons exactly).
% Recompute desiredPath/save_dir fresh -- robust to jumping here directly
% via Run Section, see Section 1's top-of-section block for the rationale.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');
% Recompute the agent-simulation constants fresh too (cheap -- no file
% I/O, just the same fixed numbers from this section's "LOAD AGENT
% SIMULATIONS" cell above), since that's a separate Run-Section cell and
% num_trials_agent/uniqueID_agent are otherwise undefined here.
trials_per_block_agent = 25;
blocks_per_sim_agent = 4;
trials_per_sim_agent = trials_per_block_agent * blocks_per_sim_agent;
numSubjs_agent = 100;
uniqueID_agent = (1:numSubjs_agent)';
num_trials_agent = repmat(trials_per_sim_agent, numSubjs_agent, 1);

%% --- Both condition, integrated ---
nll_basicRL_subjectivePE_agentSim = importdata(fullfile(save_dir, "nll_basicRL_subjectivePE_agentSim_Both_multiSP.mat"));
nll_RLsigma_subjectivePE_agentSim = importdata(fullfile(save_dir, "nll_RLsigma_subjectivePE_agentSim_Both_multiSP.mat"));
params_basicRL_subjectivePE_agentSim = importdata(fullfile(save_dir, "params_basicRL_subjectivePE_agentSim_Both_multiSP.mat"));
params_RLsigma_subjectivePE_agentSim = importdata(fullfile(save_dir, "params_RLsigma_subjectivePE_agentSim_Both_multiSP.mat"));

disp('Computing AIC and BIC for agent-simulation subjective PE models (Both, integrated)...');
num_params_agentSim = [3, 3];
model_names_agentSim = {'basicRL_subjectivePE_agentSim','RLsigma_subjectivePE_agentSim'};
[AICBIC_agentSim_table, bestModel_agentSim_table, BMS_agentSim_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_subjectivePE_agentSim, nll_RLsigma_subjectivePE_agentSim], model_names_agentSim, num_params_agentSim, num_trials_agent, uniqueID_agent);
safe_saveall(fullfile(save_dir, 'AICBIC_subjectivePE_agentSim_Both_reducedMS_multiSP.mat'), AICBIC_agentSim_table);
safe_saveall(fullfile(save_dir, 'bestModel_subjectivePE_agentSim_Both_multiSP.mat'), bestModel_agentSim_table);
disp('Agent-simulation subjective PE model AIC/BIC estimated and saved (Both, integrated).');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_agentSim_table, bestModel_agentSim_table, BMS_agentSim_table, model_names_agentSim, 'Both condition, Subjective PE, Agent Sim, Integrated');
fitSlider_ALLmodels.plot_fitted_params({params_basicRL_subjectivePE_agentSim, params_RLsigma_subjectivePE_agentSim}, model_names_agentSim, AICBIC_agentSim_table, 'Both condition, Subjective PE, Agent Sim, Integrated', 'sigma (fitted)');

%% --- Both condition, no integration ---
nll_basicRL_subjectivePE_noIntegration_agentSim = importdata(fullfile(save_dir, "nll_basicRL_subjectivePE_noIntegration_agentSim_Both_multiSP.mat"));
nll_RLsigma_subjectivePE_noIntegration_agentSim = importdata(fullfile(save_dir, "nll_RLsigma_subjectivePE_noIntegration_agentSim_Both_multiSP.mat"));
params_basicRL_subjectivePE_noIntegration_agentSim = importdata(fullfile(save_dir, "params_basicRL_subjectivePE_noIntegration_agentSim_Both_multiSP.mat"));
params_RLsigma_subjectivePE_noIntegration_agentSim = importdata(fullfile(save_dir, "params_RLsigma_subjectivePE_noIntegration_agentSim_Both_multiSP.mat"));

disp('Computing AIC and BIC for agent-simulation subjective PE models (Both, no integration)...');
model_names_agentSim_noInt = {'basicRL_subjectivePE_noIntegration_agentSim','RLsigma_subjectivePE_noIntegration_agentSim'};
[AICBIC_agentSim_noInt_table, bestModel_agentSim_noInt_table, BMS_agentSim_noInt_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_subjectivePE_noIntegration_agentSim, nll_RLsigma_subjectivePE_noIntegration_agentSim], model_names_agentSim_noInt, num_params_agentSim, num_trials_agent, uniqueID_agent);
safe_saveall(fullfile(save_dir, 'AICBIC_subjectivePE_noIntegration_agentSim_Both_reducedMS_multiSP.mat'), AICBIC_agentSim_noInt_table);
safe_saveall(fullfile(save_dir, 'bestModel_subjectivePE_noIntegration_agentSim_Both_multiSP.mat'), bestModel_agentSim_noInt_table);
disp('Agent-simulation subjective PE model AIC/BIC estimated and saved (Both, no integration).');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_agentSim_noInt_table, bestModel_agentSim_noInt_table, BMS_agentSim_noInt_table, model_names_agentSim_noInt, 'Both condition, Subjective PE, Agent Sim, No Integration');
fitSlider_ALLmodels.plot_fitted_params({params_basicRL_subjectivePE_noIntegration_agentSim, params_RLsigma_subjectivePE_noIntegration_agentSim}, model_names_agentSim_noInt, AICBIC_agentSim_noInt_table, 'Both condition, Subjective PE, Agent Sim, No Integration', 'sigma (fitted)');

%% --- Perceptual condition, integrated ---
% Recompute save_dir and the agent-simulation constants/model names fresh
% too -- see Section 5's Both-condition Model comparison cell for why.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');
trials_per_block_agent = 25;
blocks_per_sim_agent = 4;
trials_per_sim_agent = trials_per_block_agent * blocks_per_sim_agent;
numSubjs_agent = 100;
uniqueID_agent = (1:numSubjs_agent)';
num_trials_agent = repmat(trials_per_sim_agent, numSubjs_agent, 1);
num_params_agentSim = [3, 3];
model_names_agentSim = {'basicRL_subjectivePE_agentSim','RLsigma_subjectivePE_agentSim'};

nll_basicRL_subjectivePE_agentSim = importdata(fullfile(save_dir, "nll_basicRL_subjectivePE_agentSim_Perceptual_multiSP.mat"));
nll_RLsigma_subjectivePE_agentSim = importdata(fullfile(save_dir, "nll_RLsigma_subjectivePE_agentSim_Perceptual_multiSP.mat"));
params_basicRL_subjectivePE_agentSim = importdata(fullfile(save_dir, "params_basicRL_subjectivePE_agentSim_Perceptual_multiSP.mat"));
params_RLsigma_subjectivePE_agentSim = importdata(fullfile(save_dir, "params_RLsigma_subjectivePE_agentSim_Perceptual_multiSP.mat"));

disp('Computing AIC and BIC for agent-simulation subjective PE models (Perceptual, integrated)...');
[AICBIC_agentSim_table, bestModel_agentSim_table, BMS_agentSim_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_subjectivePE_agentSim, nll_RLsigma_subjectivePE_agentSim], model_names_agentSim, num_params_agentSim, num_trials_agent, uniqueID_agent);
safe_saveall(fullfile(save_dir, 'AICBIC_subjectivePE_agentSim_Perceptual_reducedMS_multiSP.mat'), AICBIC_agentSim_table);
safe_saveall(fullfile(save_dir, 'bestModel_subjectivePE_agentSim_Perceptual_multiSP.mat'), bestModel_agentSim_table);
disp('Agent-simulation subjective PE model AIC/BIC estimated and saved (Perceptual, integrated).');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_agentSim_table, bestModel_agentSim_table, BMS_agentSim_table, model_names_agentSim, 'Perceptual condition, Subjective PE, Agent Sim, Integrated');
fitSlider_ALLmodels.plot_fitted_params({params_basicRL_subjectivePE_agentSim, params_RLsigma_subjectivePE_agentSim}, model_names_agentSim, AICBIC_agentSim_table, 'Perceptual condition, Subjective PE, Agent Sim, Integrated', 'sigma (fitted)');

%% --- Perceptual condition, no integration ---
% Recompute save_dir and the agent-simulation constants/model names fresh
% too -- see Section 5's Both-condition Model comparison cell for why.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');
trials_per_block_agent = 25;
blocks_per_sim_agent = 4;
trials_per_sim_agent = trials_per_block_agent * blocks_per_sim_agent;
numSubjs_agent = 100;
uniqueID_agent = (1:numSubjs_agent)';
num_trials_agent = repmat(trials_per_sim_agent, numSubjs_agent, 1);
num_params_agentSim = [3, 3];
model_names_agentSim_noInt = {'basicRL_subjectivePE_noIntegration_agentSim','RLsigma_subjectivePE_noIntegration_agentSim'};

nll_basicRL_subjectivePE_noIntegration_agentSim = importdata(fullfile(save_dir, "nll_basicRL_subjectivePE_noIntegration_agentSim_Perceptual_multiSP.mat"));
nll_RLsigma_subjectivePE_noIntegration_agentSim = importdata(fullfile(save_dir, "nll_RLsigma_subjectivePE_noIntegration_agentSim_Perceptual_multiSP.mat"));
params_basicRL_subjectivePE_noIntegration_agentSim = importdata(fullfile(save_dir, "params_basicRL_subjectivePE_noIntegration_agentSim_Perceptual_multiSP.mat"));
params_RLsigma_subjectivePE_noIntegration_agentSim = importdata(fullfile(save_dir, "params_RLsigma_subjectivePE_noIntegration_agentSim_Perceptual_multiSP.mat"));

disp('Computing AIC and BIC for agent-simulation subjective PE models (Perceptual, no integration)...');
[AICBIC_agentSim_noInt_table, bestModel_agentSim_noInt_table, BMS_agentSim_noInt_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_subjectivePE_noIntegration_agentSim, nll_RLsigma_subjectivePE_noIntegration_agentSim], model_names_agentSim_noInt, num_params_agentSim, num_trials_agent, uniqueID_agent);
safe_saveall(fullfile(save_dir, 'AICBIC_subjectivePE_noIntegration_agentSim_Perceptual_reducedMS_multiSP.mat'), AICBIC_agentSim_noInt_table);
safe_saveall(fullfile(save_dir, 'bestModel_subjectivePE_noIntegration_agentSim_Perceptual_multiSP.mat'), bestModel_agentSim_noInt_table);
disp('Agent-simulation subjective PE model AIC/BIC estimated and saved (Perceptual, no integration).');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_agentSim_noInt_table, bestModel_agentSim_noInt_table, BMS_agentSim_noInt_table, model_names_agentSim_noInt, 'Perceptual condition, Subjective PE, Agent Sim, No Integration');
fitSlider_ALLmodels.plot_fitted_params({params_basicRL_subjectivePE_noIntegration_agentSim, params_RLsigma_subjectivePE_noIntegration_agentSim}, model_names_agentSim_noInt, AICBIC_agentSim_noInt_table, 'Perceptual condition, Subjective PE, Agent Sim, No Integration', 'sigma (fitted)');

%% ========================================================================
%  SECTION 6: MIXTURE MODEL (BASIC RL / RL SIGMA), SUBJECTIVE PE, FREE
%  SIGMA -- WITH AND WITHOUT INTEGRATION
%  ------------------------------------------------------------------------
%  Fits nll_mixture_subjectivePE / nll_mixture_integrated_subjectivePE
%  (fitSlider_ALLmodels.m -- see their headers for the GbAgent.py-style
%  compute_mixture rationale) to real subject data, same data/conditions
%  as Sections 2 and 4. Four free params: [alpha, kappa, sigma, lambda],
%  lambda bounded to [0,1] directly (no logit transform), matching
%  GbEstVars.py's l_bnds for its own lambda_param.
%  Model comparison below is 3-way (basicRL_subjectivePE vs
%  RLsigma_subjectivePE vs mixture_subjectivePE, reloading Section 2's and
%  4's saved NLLs) rather than the usual 2-way, since the whole point of
%  fitting the mixture is to see whether letting lambda vary continuously
%  actually beats committing to one pure model (lambda=0 or lambda=1) once
%  the extra parameter is BIC-penalized -- not just to log its own fit.
%  ========================================================================

% Recompute desiredPath/save_dir fresh -- see Section 1's identical block
% for why.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');

%% ----------------- Both condition: fitting ------------------------------
%% --- Mixture model, subjective PE, integrated ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
lambdaParameter = NaN(numSubjs, 1);
nll_mixture_subjectivePE = NaN(numSubjs, 1);
lb = [0, 1, 0, 0];
ub = [1, 100, 0.1, 1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
initLambda = unifrnd(lb(4), ub(4), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Mixture model subjective PE, integrated (Both)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Mixture model subjective PE, integrated (Both)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataBoth, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_mixture_integrated_subjectivePE(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,4);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp), initLambda(n,sp)];
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
    lambdaParameter(n) = best_params(4);
    nll_mixture_subjectivePE(n) = best_nll;
end
params_mixture_subjectivePE.alpha = alphaParameter;
params_mixture_subjectivePE.kappa = kappaParameter;
params_mixture_subjectivePE.sigma = sigmaParameter;
params_mixture_subjectivePE.lambda = lambdaParameter;
safe_saveall(fullfile(save_dir, 'params_mixture_subjectivePE_Both_multiSP.mat'), params_mixture_subjectivePE);
safe_saveall(fullfile(save_dir, 'nll_mixture_subjectivePE_Both_multiSP.mat'), nll_mixture_subjectivePE);
end

%% --- Mixture model, subjective PE, no integration ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
lambdaParameter = NaN(numSubjs, 1);
nll_mixture_subjectivePE_noIntegration = NaN(numSubjs, 1);
lb = [0, 1, 0, 0];
ub = [1, 100, 0.1, 1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
initLambda = unifrnd(lb(4), ub(4), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Mixture model subjective PE, no integration (Both)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Mixture model subjective PE, no integration (Both)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataBoth, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_mixture_subjectivePE(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,4);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp), initLambda(n,sp)];
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
    lambdaParameter(n) = best_params(4);
    nll_mixture_subjectivePE_noIntegration(n) = best_nll;
end
params_mixture_subjectivePE_noIntegration.alpha = alphaParameter;
params_mixture_subjectivePE_noIntegration.kappa = kappaParameter;
params_mixture_subjectivePE_noIntegration.sigma = sigmaParameter;
params_mixture_subjectivePE_noIntegration.lambda = lambdaParameter;
safe_saveall(fullfile(save_dir, 'params_mixture_subjectivePE_noIntegration_Both_multiSP.mat'), params_mixture_subjectivePE_noIntegration);
safe_saveall(fullfile(save_dir, 'nll_mixture_subjectivePE_noIntegration_Both_multiSP.mat'), nll_mixture_subjectivePE_noIntegration);
end

%% ----------------- Perceptual condition: fitting -------------------------
%% --- Mixture model, subjective PE, integrated (Perceptual) ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
lambdaParameter = NaN(numSubjs, 1);
nll_mixture_subjectivePE = NaN(numSubjs, 1);
lb = [0, 1, 0, 0];
ub = [1, 100, 0.1, 1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
initLambda = unifrnd(lb(4), ub(4), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Mixture model subjective PE, integrated (Perceptual)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Mixture model subjective PE, integrated (Perceptual)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_mixture_integrated_subjectivePE(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,4);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp), initLambda(n,sp)];
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
    lambdaParameter(n) = best_params(4);
    nll_mixture_subjectivePE(n) = best_nll;
end
params_mixture_subjectivePE.alpha = alphaParameter;
params_mixture_subjectivePE.kappa = kappaParameter;
params_mixture_subjectivePE.sigma = sigmaParameter;
params_mixture_subjectivePE.lambda = lambdaParameter;
safe_saveall(fullfile(save_dir, 'params_mixture_subjectivePE_Perceptual_multiSP.mat'), params_mixture_subjectivePE);
safe_saveall(fullfile(save_dir, 'nll_mixture_subjectivePE_Perceptual_multiSP.mat'), nll_mixture_subjectivePE);
end

%% --- Mixture model, subjective PE, no integration (Perceptual) ---
if run_fitting
alphaParameter = NaN(numSubjs, 1);
kappaParameter = NaN(numSubjs, 1);
sigmaParameter = NaN(numSubjs, 1);
lambdaParameter = NaN(numSubjs, 1);
nll_mixture_subjectivePE_noIntegration = NaN(numSubjs, 1);
lb = [0, 1, 0, 0];
ub = [1, 100, 0.1, 1];
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]);
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]);
initSigma = unifrnd(lb(3), ub(3), [numSubjs, n_startingPoints]);
initLambda = unifrnd(lb(4), ub(4), [numSubjs, n_startingPoints]);
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Mixture model subjective PE, no integration (Perceptual)');
progressQueue = parallel.pool.DataQueue;
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Mixture model subjective PE, no integration (Perceptual)'));
parfor n = 1:numSubjs
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n));
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast);
    nll_fun = @(params) fitSlider_ALLmodels.nll_mixture_subjectivePE(params, subj.mu_hat, subj.blocks, rewards, subj.condiff);
    options = optimset('Display', 'off');

    best_nll = Inf;
    best_params = NaN(1,4);
    for sp = 1:n_startingPoints
        start_params = [initAlpha(n,sp), initKappa(n,sp), initSigma(n,sp), initLambda(n,sp)];
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
    lambdaParameter(n) = best_params(4);
    nll_mixture_subjectivePE_noIntegration(n) = best_nll;
end
params_mixture_subjectivePE_noIntegration.alpha = alphaParameter;
params_mixture_subjectivePE_noIntegration.kappa = kappaParameter;
params_mixture_subjectivePE_noIntegration.sigma = sigmaParameter;
params_mixture_subjectivePE_noIntegration.lambda = lambdaParameter;
safe_saveall(fullfile(save_dir, 'params_mixture_subjectivePE_noIntegration_Perceptual_multiSP.mat'), params_mixture_subjectivePE_noIntegration);
safe_saveall(fullfile(save_dir, 'nll_mixture_subjectivePE_noIntegration_Perceptual_multiSP.mat'), nll_mixture_subjectivePE_noIntegration);
end

fitSlider_ALLmodels.progress_bar('close'); % done fitting -- close the waitbar window (safe no-op if run_fitting was false or nothing above ran)

%% ----------------- Model comparison: basicRL vs RLsigma vs mixture --------
% Recompute desiredPath/save_dir fresh -- robust to jumping here directly
% via Run Section, see Section 1's top-of-section block for the rationale.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');
% Recompute dataBoth/dataPerceptual/uniqueID/numSubjs fresh too -- see
% Section 1's Model comparison cell for why.
[dataBoth, dataPerceptual, uniqueID, numSubjs] = fitSlider_ALLmodels.load_fitting_data();

%% --- Both condition, integrated ---
nll_basicRL_subjectivePE = importdata(fullfile(save_dir, "nll_basicRL_subjectivePEv2_Both_multiSP.mat"));
nll_RLsigma_subjectivePE = importdata(fullfile(save_dir, "nll_RLsigma_subjectivePEv2_Both_multiSP.mat"));
nll_mixture_subjectivePE = importdata(fullfile(save_dir, "nll_mixture_subjectivePE_Both_multiSP.mat"));
params_basicRL_subjectivePE = importdata(fullfile(save_dir, "params_basicRL_subjectivePEv2_Both_multiSP.mat"));
params_RLsigma_subjectivePE = importdata(fullfile(save_dir, "params_RLsigma_subjectivePEv2_Both_multiSP.mat"));
params_mixture_subjectivePE = importdata(fullfile(save_dir, "params_mixture_subjectivePE_Both_multiSP.mat"));

disp('Computing AIC and BIC for basicRL vs RLsigma vs mixture (Both, integrated)...');
num_trials_both = NaN(numSubjs, 1);
for n = 1:numSubjs
    num_trials_both(n) = length(preprocess_fitSlider(dataBoth, uniqueID(n)).mu_hat);
end
num_params_mixtureComp = [3, 3, 4]; % basicRL_subjectivePE, RLsigma_subjectivePE, mixture_subjectivePE
model_names_mixtureComp = {'basicRL_subjectivePE','RLsigma_subjectivePE','mixture_subjectivePE'};
[AICBIC_mixtureComp_table, bestModel_mixtureComp_table, BMS_mixtureComp_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_subjectivePE, nll_RLsigma_subjectivePE, nll_mixture_subjectivePE], model_names_mixtureComp, num_params_mixtureComp, num_trials_both, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_mixture_subjectivePE_Both_reducedMS_multiSP.mat'), AICBIC_mixtureComp_table);
safe_saveall(fullfile(save_dir, 'bestModel_mixture_subjectivePE_Both_multiSP.mat'), bestModel_mixtureComp_table);
disp('basicRL vs RLsigma vs mixture AIC/BIC estimated and saved (Both, integrated).');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_mixtureComp_table, bestModel_mixtureComp_table, BMS_mixtureComp_table, model_names_mixtureComp, 'Both condition, Subjective PE, Integrated (incl. Mixture)');
fitSlider_ALLmodels.plot_fitted_params({params_basicRL_subjectivePE, params_RLsigma_subjectivePE, params_mixture_subjectivePE}, model_names_mixtureComp, AICBIC_mixtureComp_table, 'Both condition, Subjective PE, Integrated (incl. Mixture)', 'sigma (fitted)');

%% --- Both condition, no integration ---
nll_basicRL_subjectivePE_noIntegration = importdata(fullfile(save_dir, "nll_basicRL_subjectivePE_noIntegration_Both_multiSP.mat"));
nll_RLsigma_subjectivePE_noIntegration = importdata(fullfile(save_dir, "nll_RLsigma_subjectivePE_noIntegration_Both_multiSP.mat"));
nll_mixture_subjectivePE_noIntegration = importdata(fullfile(save_dir, "nll_mixture_subjectivePE_noIntegration_Both_multiSP.mat"));
params_basicRL_subjectivePE_noIntegration = importdata(fullfile(save_dir, "params_basicRL_subjectivePE_noIntegration_Both_multiSP.mat"));
params_RLsigma_subjectivePE_noIntegration = importdata(fullfile(save_dir, "params_RLsigma_subjectivePE_noIntegration_Both_multiSP.mat"));
params_mixture_subjectivePE_noIntegration = importdata(fullfile(save_dir, "params_mixture_subjectivePE_noIntegration_Both_multiSP.mat"));

disp('Computing AIC and BIC for basicRL vs RLsigma vs mixture (Both, no integration)...');
model_names_mixtureComp_noInt = {'basicRL_subjectivePE_noIntegration','RLsigma_subjectivePE_noIntegration','mixture_subjectivePE_noIntegration'};
[AICBIC_mixtureComp_noInt_table, bestModel_mixtureComp_noInt_table, BMS_mixtureComp_noInt_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_subjectivePE_noIntegration, nll_RLsigma_subjectivePE_noIntegration, nll_mixture_subjectivePE_noIntegration], model_names_mixtureComp_noInt, num_params_mixtureComp, num_trials_both, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_mixture_subjectivePE_noIntegration_Both_reducedMS_multiSP.mat'), AICBIC_mixtureComp_noInt_table);
safe_saveall(fullfile(save_dir, 'bestModel_mixture_subjectivePE_noIntegration_Both_multiSP.mat'), bestModel_mixtureComp_noInt_table);
disp('basicRL vs RLsigma vs mixture AIC/BIC estimated and saved (Both, no integration).');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_mixtureComp_noInt_table, bestModel_mixtureComp_noInt_table, BMS_mixtureComp_noInt_table, model_names_mixtureComp_noInt, 'Both condition, Subjective PE, No Integration (incl. Mixture)');
fitSlider_ALLmodels.plot_fitted_params({params_basicRL_subjectivePE_noIntegration, params_RLsigma_subjectivePE_noIntegration, params_mixture_subjectivePE_noIntegration}, model_names_mixtureComp_noInt, AICBIC_mixtureComp_noInt_table, 'Both condition, Subjective PE, No Integration (incl. Mixture)', 'sigma (fitted)');

%% --- Perceptual condition, integrated ---
% Recompute save_dir/dataBoth/dataPerceptual/uniqueID/numSubjs and the
% model-name/param-count constants fresh too -- see Section 6's
% Both-condition Model comparison cell for why.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');
[dataBoth, dataPerceptual, uniqueID, numSubjs] = fitSlider_ALLmodels.load_fitting_data();
num_params_mixtureComp = [3, 3, 4];
model_names_mixtureComp = {'basicRL_subjectivePE','RLsigma_subjectivePE','mixture_subjectivePE'};

nll_basicRL_subjectivePE = importdata(fullfile(save_dir, "nll_basicRL_subjectivePEv2_Perceptual_multiSP.mat"));
nll_RLsigma_subjectivePE = importdata(fullfile(save_dir, "nll_RLsigma_subjectivePEv2_Perceptual_multiSP.mat"));
nll_mixture_subjectivePE = importdata(fullfile(save_dir, "nll_mixture_subjectivePE_Perceptual_multiSP.mat"));
params_basicRL_subjectivePE = importdata(fullfile(save_dir, "params_basicRL_subjectivePEv2_Perceptual_multiSP.mat"));
params_RLsigma_subjectivePE = importdata(fullfile(save_dir, "params_RLsigma_subjectivePEv2_Perceptual_multiSP.mat"));
params_mixture_subjectivePE = importdata(fullfile(save_dir, "params_mixture_subjectivePE_Perceptual_multiSP.mat"));

disp('Computing AIC and BIC for basicRL vs RLsigma vs mixture (Perceptual, integrated)...');
num_trials_perceptual = NaN(numSubjs, 1);
for n = 1:numSubjs
    num_trials_perceptual(n) = length(preprocess_fitSlider(dataPerceptual, uniqueID(n)).mu_hat);
end
[AICBIC_mixtureComp_table, bestModel_mixtureComp_table, BMS_mixtureComp_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_subjectivePE, nll_RLsigma_subjectivePE, nll_mixture_subjectivePE], model_names_mixtureComp, num_params_mixtureComp, num_trials_perceptual, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_mixture_subjectivePE_Perceptual_reducedMS_multiSP.mat'), AICBIC_mixtureComp_table);
safe_saveall(fullfile(save_dir, 'bestModel_mixture_subjectivePE_Perceptual_multiSP.mat'), bestModel_mixtureComp_table);
disp('basicRL vs RLsigma vs mixture AIC/BIC estimated and saved (Perceptual, integrated).');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_mixtureComp_table, bestModel_mixtureComp_table, BMS_mixtureComp_table, model_names_mixtureComp, 'Perceptual condition, Subjective PE, Integrated (incl. Mixture)');
fitSlider_ALLmodels.plot_fitted_params({params_basicRL_subjectivePE, params_RLsigma_subjectivePE, params_mixture_subjectivePE}, model_names_mixtureComp, AICBIC_mixtureComp_table, 'Perceptual condition, Subjective PE, Integrated (incl. Mixture)', 'sigma (fitted)');

%% --- Perceptual condition, no integration ---
% Recompute save_dir/dataBoth/dataPerceptual/uniqueID/numSubjs and the
% model-name/param-count constants fresh too -- see Section 6's
% Both-condition Model comparison cell for why.
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting');
[dataBoth, dataPerceptual, uniqueID, numSubjs] = fitSlider_ALLmodels.load_fitting_data();
num_params_mixtureComp = [3, 3, 4];
model_names_mixtureComp_noInt = {'basicRL_subjectivePE_noIntegration','RLsigma_subjectivePE_noIntegration','mixture_subjectivePE_noIntegration'};

nll_basicRL_subjectivePE_noIntegration = importdata(fullfile(save_dir, "nll_basicRL_subjectivePE_noIntegration_Perceptual_multiSP.mat"));
nll_RLsigma_subjectivePE_noIntegration = importdata(fullfile(save_dir, "nll_RLsigma_subjectivePE_noIntegration_Perceptual_multiSP.mat"));
nll_mixture_subjectivePE_noIntegration = importdata(fullfile(save_dir, "nll_mixture_subjectivePE_noIntegration_Perceptual_multiSP.mat"));
params_basicRL_subjectivePE_noIntegration = importdata(fullfile(save_dir, "params_basicRL_subjectivePE_noIntegration_Perceptual_multiSP.mat"));
params_RLsigma_subjectivePE_noIntegration = importdata(fullfile(save_dir, "params_RLsigma_subjectivePE_noIntegration_Perceptual_multiSP.mat"));
params_mixture_subjectivePE_noIntegration = importdata(fullfile(save_dir, "params_mixture_subjectivePE_noIntegration_Perceptual_multiSP.mat"));

disp('Computing AIC and BIC for basicRL vs RLsigma vs mixture (Perceptual, no integration)...');
[AICBIC_mixtureComp_noInt_table, bestModel_mixtureComp_noInt_table, BMS_mixtureComp_noInt_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_subjectivePE_noIntegration, nll_RLsigma_subjectivePE_noIntegration, nll_mixture_subjectivePE_noIntegration], model_names_mixtureComp_noInt, num_params_mixtureComp, num_trials_perceptual, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_mixture_subjectivePE_noIntegration_Perceptual_reducedMS_multiSP.mat'), AICBIC_mixtureComp_noInt_table);
safe_saveall(fullfile(save_dir, 'bestModel_mixture_subjectivePE_noIntegration_Perceptual_multiSP.mat'), bestModel_mixtureComp_noInt_table);
disp('basicRL vs RLsigma vs mixture AIC/BIC estimated and saved (Perceptual, no integration).');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_mixtureComp_noInt_table, bestModel_mixtureComp_noInt_table, BMS_mixtureComp_noInt_table, model_names_mixtureComp_noInt, 'Perceptual condition, Subjective PE, No Integration (incl. Mixture)');
fitSlider_ALLmodels.plot_fitted_params({params_basicRL_subjectivePE_noIntegration, params_RLsigma_subjectivePE_noIntegration, params_mixture_subjectivePE_noIntegration}, model_names_mixtureComp_noInt, AICBIC_mixtureComp_noInt_table, 'Perceptual condition, Subjective PE, No Integration (incl. Mixture)', 'sigma (fitted)');
