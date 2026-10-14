%% ========================================================================
%  Script: Fit RL Models with Sigma Fixed from the Perceptual-Choice Fit (+AIC/BIC)
%  ------------------------------------------------------------------------
%  Same pipeline as fitReducedModelSpace.m, except sigma (perceptual
%  sensitivity) is not fit jointly with each model's other parameters.
%  Instead, each subject's sigma comes from their perceptual-choice fit
%  (fitPerceptualChoice.m, saved to sigma_perceptualChoice.mat) and is held
%  fixed while only the remaining parameters (alpha/kappa) are estimated
%  from the slider data. This reduces each model's free-parameter count by
%  one relative to fitReducedModelSpace.m, which is reflected in the
%  AIC/BIC computation below.
%  ------------------------------------------------------------------------
%  After the perceptual-sigma fit, structured as four self-contained
%  sections, one per combination of:
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
%  section's previously-saved *_fixedSigma_*_multiSP.mat files.
%  bayesianAgent only appears in Section 1 -- it has no subjective-PE or
%  no-integration variant (its belief update goes through the Agent
%  class's own internal mechanism, not a simple alpha-driven delta rule or
%  the set_o grid-integration Sections 3/4 use instead).
%  ========================================================================
clc; clearvars; % clear the command window and workspace so old variables can't leak into this run
rng(123); % fix the random seed so starting points / shuffling are reproducible

% PATH STUFF -- anchor the read location to Agent/model fitting/, where
% fitPerceptualChoice.m saves sigma_perceptualChoice.mat, regardless of
% MATLAB's current folder.
currentDir = cd; % MATLAB's current working folder
reqPath = 'Reward-learning-analysis (code_review)'; % to which directory one must save in
pathParts = strsplit(currentDir, filesep); % split the path into its folder names
if strcmp(pathParts{end}, reqPath) % already standing inside the repo root?
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath); % otherwise, walk up/down to find it
end
save_dir = fullfile(desiredPath, 'Agent', 'model fitting'); % folder where every .mat file below is read from / written to

%% =================== LOAD AND PREPARE DATA =============================
% dataBoth/dataPerceptual: trial-level tables for the two conditions this
% script fits separately. uniqueID: subject ID list (same order used for
% every per-subject array below). numSubjs: how many subjects there are.
% data: full dataset (all conditions), needed below for the integrated
% perceptual-choice sigma fit.
[dataBoth, dataPerceptual, uniqueID, numSubjs, data] = fitSlider_ALLmodels.load_fitting_data();

% Multi-start: fmincon is a local optimizer, so draw multiple random
% starting points per subject within each model's [lb, ub] and keep the
% lowest-NLL fit, to reduce the risk of converging to a local minimum.
% Sigma is fixed here, but the remaining free parameters (alpha/kappa)
% still carry the same local-minimum risk as in fitReducedModelSpace.m.
n_startingPoints = 10; % how many random restarts per subject, per model

% run_fitting toggles whether the fitting blocks below (re)run at all, in
% all four sections of this script (plus the perceptual-sigma fit above
% them):
%   true (default)  - run every parfor fit and save fresh params_*/nll_*
%                      .mat files, as before.
%   false            - skip all fitting blocks entirely. The AIC/BIC
%                      sections further down already reload nll_basicRL/
%                      nll_RLsigma/nll_bayesianAgent from disk via
%                      importdata() before using them, so with this set to
%                      false the whole rest of the script (AIC/BIC,
%                      model-selection, and all plots) runs straight off
%                      each section's previously-saved *_fixedSigma_*_
%                      multiSP.mat files, without waiting through the
%                      fitting loops again.
run_fitting = true;

%% ========================================================================
%  PERCEPTUAL SENSITIVITY (SIGMA) FITTING
%  ------------------------------------------------------------------------
%  Integrated from fitPerceptualChoice.m (rather than relying on a
%  separately-run copy of that script) so sigma_perceptualChoice.mat is
%  always refit fresh from the current data before being used as fixedSigma
%  below. Previously, if fitPerceptualChoice.m was rerun on its own without
%  also rerunning this script, the rest of the fixedSigma fits would
%  silently keep using whatever sigma was already on disk -- exactly what
%  happened to bayesianAgent/RLsigma (13 subjects fit against stale sigma
%  values after a perceptual-choice refit this script never picked up).
%  Also reuses run_fitting: when false, skips straight to reloading the
%  previously-saved sigma_perceptualChoice.mat, same as before.
%  ========================================================================
if run_fitting

% Preallocate per-subject outputs
sigmaParameter = NaN(numSubjs, 1);    % Best-fitting sigma for each subject
nll_perceptualChoice = NaN(numSubjs, 1); % Negative log-likelihood at that fit
matchRate = NaN(numSubjs, 1);         % Posterior predictive check: per-subject
                                       % fraction of trials where the simulated
                                       % choice matches the observed choice

% fmincon settings: single free parameter (sigma) with bounds
lb = 0.001;
ub = 0.1;

% Multi-start optimization: fmincon is a local optimizer, so a single
% starting point risks converging to a local rather than global minimum.
% Draw multiple random starting points per subject within [lb, ub] and
% keep whichever converges to the lowest NLL. Uses its own starting-point
% count (15, matching fitPerceptualChoice.m's original) rather than the
% shared n_startingPoints (10, set below for the alpha/kappa model fits).
n_startingPoints_sigma = 15;
initSigma = unifrnd(lb, ub, [numSubjs, n_startingPoints_sigma]);

% Graphical progress bar: this loop is serial, not parfor, so the waitbar
% can be called directly without a DataQueue/afterEach.
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints_sigma, 'Perceptual sigma');

% Accumulators for the posterior predictive check below: pooled across all
% subjects' trials, so trial counts must line up 1:1 across the 3 arrays.
allCondiff = [];
allChoicesObserved = [];
allChoicesSimulated = [];

for n = 1:numSubjs
    % Extract and preprocess this subject's trials (pupil flag unused here, set to 0).
    % requireMuHat = false: this model only needs choice/condiff/blocks, so
    % trials with a valid choice but a missing slider (mu) response are kept.
    subj = preprocess_fitSlider(data, uniqueID(n), 0, false);

    % Objective function: NLL of this subject's perceptual choices given sigma
    nll_fun = @(params) fitSlider_ALLmodels.nll_perceptualChoice(params, subj.dataTable, ...
        length(unique(subj.blocks)), 20, unique(subj.blocks));

    options = optimset('Display', 'off');   % Suppress fmincon iteration output

    % Fit sigma from each starting point, keeping the best (lowest-NLL) fit
    best_sigma = NaN;
    best_nll = Inf;
    for sp = 1:n_startingPoints_sigma
        [params, nll] = fmincon(nll_fun, initSigma(n, sp), [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_sigma = params(1);
        end
        fitSlider_ALLmodels.progress_bar('update', [n, sp], numSubjs, n_startingPoints_sigma, 'Perceptual sigma');
    end

    sigmaParameter(n) = best_sigma;   % Store this subject's best-fitting sigma
    nll_perceptualChoice(n) = best_nll;  % Store NLL at the best fit

    % Posterior predictive check: simulate one replicate of this subject's
    % choices using their own real trial sequence (condiff, blocks) and
    % their just-fitted sigma, for comparison against their observed
    % choices below.
    nBlocksSubj = length(unique(subj.blocks));
    simulated_choices = fitSlider_ALLmodels.simulate_perceptualChoice(best_sigma, subj.condiff, subj.blocks, nBlocksSubj);
    allCondiff = [allCondiff; subj.condiff];
    allChoicesObserved = [allChoicesObserved; subj.choices];
    allChoicesSimulated = [allChoicesSimulated; simulated_choices(:)];
    matchRate(n) = mean(simulated_choices(:) == subj.choices); % this subject's trial-level agreement
end
fitSlider_ALLmodels.progress_bar('close');

% Package fitted sigmas into a struct (named to match the convention used
% elsewhere, e.g. plotSigma.m, even though this isn't a full learning-model fit)
params_perceptualChoice.sigma = sigmaParameter;

% Persist results to disk (consumed by the rest of this script, which
% fixes sigma to these fitted values instead of fitting it jointly)
safe_saveall(fullfile(save_dir, 'sigma_perceptualChoice.mat'), params_perceptualChoice);
safe_saveall(fullfile(save_dir, 'nll_perceptualChoice.mat'), nll_perceptualChoice);

fixedSigma = sigmaParameter; % use the just-fitted sigma directly -- no need to reload from disk

% Plot fitted sigma parameter - will remove it later. just here to see quick results
% Mean and SEM of the fitted sigma across subjects
mean_sigma = nanmean(sigmaParameter);
SEM_sigma = nanstd(sigmaParameter) ./ sqrt(sum(~isnan(sigmaParameter)));

figure('Position', [100, 100, 200, 200]);
% Bar plot with per-subject scatter dots, mean marker, and SEM error bar
bar_plots_pval(sigmaParameter, mean_sigma, SEM_sigma, numSubjs, 1, 1, ...
    {'Perceptual choice model'}, 1, {'sigma'}, 'Fitted perceptual sigma', ...
    '', 'sigma', 0, 1, 20, 1, 12, 1, 'Arial', 1, lines(1));

% POSTERIOR PREDICTIVE CHECK -- single replicate - will remove it later
% too.
% Compares the psychometric curve (P(choice = 1) vs. contrast difference)
% between observed choices and one simulated replicate per subject,
% pooled across all subjects' trials. Each replicate was generated above
% using that subject's own fitted sigma and real trial sequence, so this
% checks whether the fitted model reproduces the real data's psychometric
% pattern.
nBins = 10;
edges = linspace(min(allCondiff), max(allCondiff), nBins+1);
binCenters = (edges(1:end-1) + edges(2:end)) / 2;
binIdx = discretize(allCondiff, edges);

propObserved = NaN(nBins,1);
propSimulated = NaN(nBins,1);
for b = 1:nBins
    inBin = binIdx == b;
    propObserved(b) = mean(allChoicesObserved(inBin));
    propSimulated(b) = mean(allChoicesSimulated(inBin));
end

figure('Position', [100, 100, 400, 350]);
plot(binCenters, propObserved, 'o-', 'LineWidth', 1.5, 'Color', [0.2 0.2 0.2], 'DisplayName', 'Observed');
hold on;
plot(binCenters, propSimulated, 's--', 'LineWidth', 1.5, 'Color', [0.8 0.3 0.3], 'DisplayName', 'Simulated (1 replicate)');
xlabel('Contrast difference (condiff relative)');
ylabel('P(choice = 1)');
title('Posterior predictive check: perceptual choice');
legend('Location', 'best');
grid on;

% POSTERIOR PREDICTIVE CHECK -- group-level summary
% The pooled psychometric curve above can look fine even if it's masking
% subjects whose fit is bad in opposite directions (one too sensitive, one
% not sensitive enough, averaging out). matchRate(n), computed in the
% fitting loop above, is a per-subject summary that doesn't have this
% problem: the fraction of that subject's trials where the (single-
% replicate) simulated choice matches their actual observed choice. A
% subject whose fitted sigma badly misses their real behavior will show up
% here even if the group-average curve looks reasonable.
mean_matchRate = nanmean(matchRate);
SEM_matchRate = nanstd(matchRate) ./ sqrt(sum(~isnan(matchRate)));
fprintf('Posterior predictive check: mean simulated-observed choice match rate = %.3f (SEM = %.3f) across %d subjects.\n', ...
    mean_matchRate, SEM_matchRate, numSubjs);
fprintf('Range across subjects: [%.3f, %.3f]. Subjects below 0.7 match rate: %d/%d.\n', ...
    min(matchRate), max(matchRate), sum(matchRate < 0.7), numSubjs);

figure('Position', [100, 100, 200, 200]);
bar_plots_pval(matchRate, mean_matchRate, SEM_matchRate, numSubjs, 1, 1, ...
    {'Perceptual choice model'}, 1, {'match rate'}, 'Simulated-observed choice agreement', ...
    '', 'proportion trials matched', 0, 1, 20, 1, 12, 1, 'Arial', 1, lines(1));

% PARAMETER RECOVERY -- validate the fit above by simulating synthetic
% choice data at known sigma values, re-fitting sigma from that data, and
% comparing recovered vs. true sigma (fitSlider_ALLmodels.m owns the
% simulation/fitting/plotting logic; this just chooses the study parameters).
n_parameters = 20;     % number of synthetic subjects/sigma values to test
n_recoveryStartingPoints = 20;  % random fmincon starting points per subject
n_recoveryTrials = 1000;        % simulated trials per subject

[sigmaRange, best_recovered_sigmas, best_nlls] = fitSlider_ALLmodels.recover_perceptualChoiceSigma(n_parameters, n_recoveryStartingPoints, n_recoveryTrials);
fitSlider_ALLmodels.plot_perceptualChoiceRecovery(sigmaRange, best_recovered_sigmas);

else
% run_fitting == false: skip re-fitting sigma and just reload the
% previously-saved perceptual-choice fit from disk, same as before.
perceptualSigma = importdata(fullfile(save_dir, 'sigma_perceptualChoice.mat')); % load the saved struct
fixedSigma = perceptualSigma.sigma; % numSubjs x 1 vector of fixed sigma values, one per subject
end

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

alphaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit alpha (learning rate)
kappaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit kappa (report precision)
sigmaParameter = NaN(numSubjs, 1); % will hold each subject's fixed sigma, copied in for record-keeping
nll_basicRL = NaN(numSubjs, 1); % will hold each subject's best (lowest) negative log-likelihood
lb = [0, 1]; % lower bounds for [alpha, kappa]
ub = [1, 100]; % upper bounds for [alpha, kappa]
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]); % random alpha starting points: numSubjs x n_startingPoints
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]); % random kappa starting points: numSubjs x n_startingPoints
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Basic RL (Both, fixed sigma)'); % open/relabel the waitbar
progressQueue = parallel.pool.DataQueue; % channel parfor workers use to report progress back to the client
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Basic RL (Both, fixed sigma)')); % what to do each time a worker reports progress

parfor n = 1:numSubjs % fit every subject independently and in parallel

    subj = preprocess_fitSlider(dataBoth, uniqueID(n)); % pull this subject's trial-level data (mu_hat, blocks, condiff, etc.)
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast); % recode rewards into the model's fixed state-0 frame
    sigma_n = fixedSigma(n); % this subject's fixed sigma (not estimated here)
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated( ... % build this subject's objective function;
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff); % params(1)=alpha, params(2)=kappa, sigma is fixed in
    options = optimset('Display', 'off'); % keep fmincon quiet

    best_nll = Inf; % track the lowest NLL found across starting points so far
    best_params = NaN(1,2); % track the [alpha, kappa] that achieved best_nll
    for sp = 1:n_startingPoints % try each random starting point in turn
        start_params = [initAlpha(n,sp), initKappa(n,sp)]; % this restart's starting [alpha, kappa]
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options); % local optimization within [lb, ub]
        if nll < best_nll % did this restart do better than anything seen so far?
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]); % tell the client this (subject, start point) is done
    end

    alphaParameter(n) = best_params(1); % store this subject's best-fit alpha
    kappaParameter(n) = best_params(2); % store this subject's best-fit kappa
    sigmaParameter(n) = sigma_n; % store this subject's fixed sigma alongside the fitted params
    nll_basicRL(n) = best_nll; % store this subject's best (lowest) NLL

end
params_basicRL.alpha = alphaParameter; % bundle all subjects' fitted alphas into one struct
params_basicRL.kappa = kappaParameter; % ...and kappas
params_basicRL.sigma = sigmaParameter; % ...and the fixed sigmas
safe_saveall(fullfile(save_dir, 'params_basicRL_fixedSigma_Both_multiSP.mat'), params_basicRL); % save fitted parameters to disk
safe_saveall(fullfile(save_dir, 'nll_basicRL_fixedSigma_Both_multiSP.mat'), nll_basicRL); % save per-subject NLLs to disk
end

%% --- Bayesian Agent model ---
if run_fitting

kappaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit kappa (this model has no alpha -- belief updates are parameter-free)
sigmaParameter = NaN(numSubjs, 1); % will hold each subject's fixed sigma
nll_bayesianAgent = NaN(numSubjs, 1); % will hold each subject's best NLL
lb = 1; % lower bound for kappa
ub = 100; % upper bound for kappa
initKappa = unifrnd(lb, ub, [numSubjs, n_startingPoints]); % random kappa starting points: numSubjs x n_startingPoints
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Bayesian Agent (Both, fixed sigma)'); % open/relabel the waitbar
progressQueue = parallel.pool.DataQueue; % fresh progress channel for this block
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Bayesian Agent (Both, fixed sigma)'));

parfor n = 1:numSubjs % fit every subject independently and in parallel

    subj = preprocess_fitSlider(dataBoth, uniqueID(n)); % this subject's trial-level data
    rewards = fitSlider_ALLmodels.recode_rewards(subj.rewards, subj.contrast); % recode rewards into the model's fixed frame
    sigma_n = fixedSigma(n); % this subject's fixed sigma
    nll_fun = @(params) fitSlider_ALLmodels.nll_bayesianAgent([params(1), sigma_n], subj.mu_hat, subj.dataTable, ... % params(1)=kappa, sigma fixed in
        length(unique(subj.blocks)), 25, unique(subj.blocks), rewards); % plus block count/size/IDs needed by the Bayesian belief update
    options = optimset('Display', 'off'); % keep fmincon quiet

    best_nll = Inf; % best NLL seen so far across restarts
    best_params = NaN; % kappa value that achieved best_nll
    for sp = 1:n_startingPoints % try each random starting kappa
        [params, nll] = fmincon(nll_fun, initKappa(n,sp), [], [], [], [], lb, ub, [], options); % optimize kappa within [lb, ub]
        if nll < best_nll % keep this restart if it's the best so far
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]); % report progress
    end

    kappaParameter(n) = best_params(1); % store this subject's best-fit kappa
    sigmaParameter(n) = sigma_n; % store this subject's fixed sigma
    nll_bayesianAgent(n) = best_nll; % store this subject's best NLL

end

params_bayesianAgent.sigma = sigmaParameter; % bundle sigmas into a struct
params_bayesianAgent.kappa = kappaParameter; % ...and kappas
safe_saveall(fullfile(save_dir, 'params_bayesianAgent_fixedSigma_Both_multiSP.mat'), params_bayesianAgent); % save fitted parameters
safe_saveall(fullfile(save_dir, 'nll_bayesianAgent_fixedSigma_Both_multiSP.mat'), nll_bayesianAgent); % save per-subject NLLs
end

%% --- RL + Est. Sensitivity model ---
if run_fitting

alphaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit alpha
kappaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit kappa
sigmaParameter = NaN(numSubjs, 1); % will hold each subject's fixed sigma
nll_RLsigma = NaN(numSubjs, 1); % will hold each subject's best NLL
lb = [0, 1]; % lower bounds for [alpha, kappa]
ub = [1, 100]; % upper bounds for [alpha, kappa]
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]); % random alpha starting points
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]); % random kappa starting points
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'RL + Est. Sensitivity (Both, fixed sigma)'); % open/relabel the waitbar
progressQueue = parallel.pool.DataQueue; % fresh progress channel for this block
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'RL + Est. Sensitivity (Both, fixed sigma)'));

parfor n = 1:numSubjs % fit every subject independently and in parallel

    subj = preprocess_fitSlider(dataBoth, uniqueID(n)); % this subject's trial-level data
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast); % recode rewards into the model's fixed frame
    sigma_n = fixedSigma(n); % this subject's fixed sigma
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI( ... % this model's belief-weighted learning rate variant of basicRL
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff); % params(1)=alpha, params(2)=kappa, sigma fixed in
    options = optimset('Display', 'off'); % keep fmincon quiet

    best_nll = Inf; % best NLL seen so far across restarts
    best_params = NaN(1,2); % [alpha, kappa] that achieved best_nll
    for sp = 1:n_startingPoints % try each random starting point
        start_params = [initAlpha(n,sp), initKappa(n,sp)]; % this restart's starting [alpha, kappa]
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options); % local optimization within [lb, ub]
        if nll < best_nll % keep this restart if it's the best so far
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]); % report progress
    end

    alphaParameter(n) = best_params(1); % store this subject's best-fit alpha
    kappaParameter(n) = best_params(2); % store this subject's best-fit kappa
    sigmaParameter(n) = sigma_n; % store this subject's fixed sigma
    nll_RLsigma(n) = best_nll; % store this subject's best NLL

end

params_RLsigma.alpha = alphaParameter; % bundle alphas into a struct
params_RLsigma.sigma = sigmaParameter; % ...and sigmas
params_RLsigma.kappa = kappaParameter; % ...and kappas
safe_saveall(fullfile(save_dir, 'params_RLsigma_fixedSigma_Both_multiSP.mat'), params_RLsigma); % save fitted parameters
safe_saveall(fullfile(save_dir, 'nll_RLsigma_fixedSigma_Both_multiSP.mat'), nll_RLsigma); % save per-subject NLLs
end

%% ----------------- Perceptual condition: fitting -------------------------
%% --- Basic RL model (Perceptual) ---
if run_fitting
alphaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit alpha
kappaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit kappa
sigmaParameter = NaN(numSubjs, 1); % will hold each subject's fixed sigma
nll_basicRL = NaN(numSubjs, 1); % will hold each subject's best NLL
lb = [0, 1]; % lower bounds for [alpha, kappa]
ub = [1, 100]; % upper bounds for [alpha, kappa]
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]); % random alpha starting points
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]); % random kappa starting points
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Basic RL (Perceptual, fixed sigma)'); % open/relabel the waitbar
progressQueue = parallel.pool.DataQueue; % fresh progress channel for this block
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Basic RL (Perceptual, fixed sigma)'));
parfor n = 1:numSubjs % fit every subject independently and in parallel

    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n)); % this subject's trial-level data, Perceptual condition
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast); % recode rewards into the model's fixed frame
    sigma_n = fixedSigma(n); % this subject's fixed sigma
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated( ... % objective-PE basicRL objective
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff); % params(1)=alpha, params(2)=kappa, sigma fixed in
    options = optimset('Display', 'off'); % keep fmincon quiet

    best_nll = Inf; % best NLL seen so far across restarts
    best_params = NaN(1,2); % [alpha, kappa] that achieved best_nll
    for sp = 1:n_startingPoints % try each random starting point
        start_params = [initAlpha(n,sp), initKappa(n,sp)]; % this restart's starting [alpha, kappa]
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options); % local optimization within [lb, ub]
        if nll < best_nll % keep this restart if it's the best so far
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]); % report progress
    end

    alphaParameter(n) = best_params(1); % store this subject's best-fit alpha
    kappaParameter(n) = best_params(2); % store this subject's best-fit kappa
    sigmaParameter(n) = sigma_n; % store this subject's fixed sigma
    nll_basicRL(n) = best_nll; % store this subject's best NLL

end
params_basicRL.alpha = alphaParameter; % bundle alphas into a struct
params_basicRL.kappa = kappaParameter; % ...and kappas
params_basicRL.sigma = sigmaParameter; % ...and sigmas
safe_saveall(fullfile(save_dir, 'params_basicRL_fixedSigma_Perceptual_multiSP.mat'), params_basicRL); % save fitted parameters
safe_saveall(fullfile(save_dir, 'nll_basicRL_fixedSigma_Perceptual_multiSP.mat'), nll_basicRL); % save per-subject NLLs
end

%% --- Bayesian Agent model (Perceptual) ---
if run_fitting
kappaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit kappa
sigmaParameter = NaN(numSubjs, 1); % will hold each subject's fixed sigma
nll_bayesianAgent = NaN(numSubjs, 1); % will hold each subject's best NLL
lb = 1; % lower bound for kappa
ub = 100; % upper bound for kappa
initKappa = unifrnd(lb, ub, [numSubjs, n_startingPoints]); % random kappa starting points
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Bayesian Agent (Perceptual, fixed sigma)'); % open/relabel the waitbar
progressQueue = parallel.pool.DataQueue; % fresh progress channel for this block
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Bayesian Agent (Perceptual, fixed sigma)'));
parfor n = 1:numSubjs % fit every subject independently and in parallel
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n)); % this subject's trial-level data, Perceptual condition
    rewards = fitSlider_ALLmodels.recode_rewards(subj.rewards, subj.contrast); % recode rewards into the model's fixed frame
    sigma_n = fixedSigma(n); % this subject's fixed sigma
    nll_fun = @(params) fitSlider_ALLmodels.nll_bayesianAgent([params(1), sigma_n], subj.mu_hat, subj.dataTable, ... % params(1)=kappa, sigma fixed in
        length(unique(subj.blocks)), 25, unique(subj.blocks), rewards); % plus block count/size/IDs needed by the Bayesian belief update
    options = optimset('Display', 'off'); % keep fmincon quiet

    best_nll = Inf; % best NLL seen so far across restarts
    best_params = NaN; % kappa value that achieved best_nll
    for sp = 1:n_startingPoints % try each random starting kappa
        [params, nll] = fmincon(nll_fun, initKappa(n,sp), [], [], [], [], lb, ub, [], options); % optimize kappa within [lb, ub]
        if nll < best_nll % keep this restart if it's the best so far
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]); % report progress
    end

    kappaParameter(n) = best_params(1); % store this subject's best-fit kappa
    sigmaParameter(n) = sigma_n; % store this subject's fixed sigma
    nll_bayesianAgent(n) = best_nll; % store this subject's best NLL
end
params_bayesianAgent.sigma = sigmaParameter; % bundle sigmas into a struct
params_bayesianAgent.kappa = kappaParameter; % ...and kappas
safe_saveall(fullfile(save_dir, 'params_bayesianAgent_fixedSigma_Perceptual_multiSP.mat'), params_bayesianAgent); % save fitted parameters
safe_saveall(fullfile(save_dir, 'nll_bayesianAgent_fixedSigma_Perceptual_multiSP.mat'), nll_bayesianAgent); % save per-subject NLLs
end

%% --- RL + Est. Sensitivity model (Perceptual) ---
if run_fitting
alphaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit alpha
kappaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit kappa
sigmaParameter = NaN(numSubjs, 1); % will hold each subject's fixed sigma
nll_RLsigma = NaN(numSubjs, 1); % will hold each subject's best NLL
lb = [0, 1]; % lower bounds for [alpha, kappa]
ub = [1, 100]; % upper bounds for [alpha, kappa]
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]); % random alpha starting points
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]); % random kappa starting points
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'RL + Est. Sensitivity (Perceptual, fixed sigma)'); % open/relabel the waitbar
progressQueue = parallel.pool.DataQueue; % fresh progress channel for this block
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'RL + Est. Sensitivity (Perceptual, fixed sigma)'));
parfor n = 1:numSubjs % fit every subject independently and in parallel
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n)); % this subject's trial-level data, Perceptual condition
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast); % recode rewards into the model's fixed frame
    sigma_n = fixedSigma(n); % this subject's fixed sigma
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI( ... % belief-weighted learning rate variant of basicRL
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff); % params(1)=alpha, params(2)=kappa, sigma fixed in
    options = optimset('Display', 'off'); % keep fmincon quiet

    best_nll = Inf; % best NLL seen so far across restarts
    best_params = NaN(1,2); % [alpha, kappa] that achieved best_nll
    for sp = 1:n_startingPoints % try each random starting point
        start_params = [initAlpha(n,sp), initKappa(n,sp)]; % this restart's starting [alpha, kappa]
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options); % local optimization within [lb, ub]
        if nll < best_nll % keep this restart if it's the best so far
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]); % report progress
    end

    alphaParameter(n) = best_params(1); % store this subject's best-fit alpha
    kappaParameter(n) = best_params(2); % store this subject's best-fit kappa
    sigmaParameter(n) = sigma_n; % store this subject's fixed sigma
    nll_RLsigma(n) = best_nll; % store this subject's best NLL
end
params_RLsigma.alpha = alphaParameter; % bundle alphas into a struct
params_RLsigma.sigma = sigmaParameter; % ...and sigmas
params_RLsigma.kappa = kappaParameter; % ...and kappas
safe_saveall(fullfile(save_dir, 'params_RLsigma_fixedSigma_Perceptual_multiSP.mat'), params_RLsigma); % save fitted parameters
safe_saveall(fullfile(save_dir, 'nll_RLsigma_fixedSigma_Perceptual_multiSP.mat'), nll_RLsigma); % save per-subject NLLs
end

%% ----------------- Model comparison ---------------------------------------
%% --- Both condition ---
% Reload each model's per-subject NLLs (works whether this run just fit
% them above or run_fitting was false and they're from an earlier run).
nll_basicRL = importdata(fullfile(save_dir, "nll_basicRL_fixedSigma_Both_multiSP.mat"));
nll_RLsigma = importdata(fullfile(save_dir, "nll_RLsigma_fixedSigma_Both_multiSP.mat"));
nll_bayesianAgent = importdata(fullfile(save_dir, "nll_bayesianAgent_fixedSigma_Both_multiSP.mat"));
% Reloaded (rather than reused in-memory) for the same reason as the NLLs
% above: params_basicRL/params_RLsigma/params_bayesianAgent get overwritten
% by the Perceptual-condition fitting blocks further up the script, so the
% in-memory values here would actually be the Perceptual condition's.
params_basicRL = importdata(fullfile(save_dir, "params_basicRL_fixedSigma_Both_multiSP.mat"));
params_RLsigma = importdata(fullfile(save_dir, "params_RLsigma_fixedSigma_Both_multiSP.mat"));
params_bayesianAgent = importdata(fullfile(save_dir, "params_bayesianAgent_fixedSigma_Both_multiSP.mat"));

disp('Computing AIC and BIC for all models (sigma fixed)...');
% Number of trials per subject -- NOT a fixed 200: each condition (Both/
% Perceptual) only gets 4 of the task's 12 blocks (100 trials), and
% requireMuHat further drops any trials with a missing slider response,
% so this is computed per subject rather than assumed. Named per-condition
% (num_trials_both, not a shared num_trials) so the Perceptual block below
% can't silently overwrite the value the subjective-PE comparison further
% down also needs for this condition.
num_trials_both = NaN(numSubjs, 1);
for n = 1:numSubjs
    num_trials_both(n) = length(preprocess_fitSlider(dataBoth, uniqueID(n)).mu_hat);
end
% Model parameter counts -- one fewer than fitReducedModelSpace.m, since
% sigma is fixed rather than estimated. [basicRL, RLsigma, BayesianAgent]
num_params = [2, 2, 1]; % free-parameter count per model, in the same column order as the NLLs below
model_names = {'basicRL','RLsigma','BayesianAgent'}; % display/column names, same column order as num_params
% One call computes AIC/BIC/delta-BIC per subject and model, finds each
% subject's best-fitting model, and runs Bayesian Model Selection (BMS)
% across subjects -- see fitSlider_ALLmodels.compute_model_comparison.
[AICBIC_table, bestModel_table, BMS_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL, nll_RLsigma, nll_bayesianAgent], model_names, num_params, num_trials_both, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_fixedSigma_Both_reducedMS_multiSP.mat'), AICBIC_table); % save per-subject AIC/BIC/delta-BIC
safe_saveall(fullfile(save_dir, 'bestModel_fixedSigma_Both_multiSP.mat'), bestModel_table); % save each subject's best-fitting model
disp('All model parameters and AIC/BIC estimated and saved (sigma fixed).');

% Plot proportion-best-fit bar chart, BMS probability/exceedance bars, and
% the winner-highlighted summary figure -- see plot_model_comparison.
fitSlider_ALLmodels.plot_model_comparison(AICBIC_table, bestModel_table, BMS_table, model_names, 'Both condition, Sigma Fixed');

% Plot fitted alpha/kappa/sigma across models, plus summed AIC/BIC and
% mean delta-BIC -- integrated from plotParams_reducedMS_fixedSigma.m.
bic_ax_both_obj = fitSlider_ALLmodels.plot_fitted_params({params_basicRL, params_RLsigma, params_bayesianAgent}, model_names, AICBIC_table, 'Both condition, Sigma Fixed', 'sigma (fixed, from perceptual-choice fit)');

% Family-wise Model Comparison using VBA Toolbox: convert BIC to log model
% evidence and group models into families: Basic RL alone vs. the two
% LR-modulation models (RLsigma, BayesianAgent).
BIC_matrix = [AICBIC_table.BIC_basicRL, ...
                  AICBIC_table.BIC_RLsigma, ...
                  AICBIC_table.BIC_BayesianAgent]; % numSubjs x 3 matrix of BIC values, one column per model
lme = -0.5 * BIC_matrix; % convert BIC to (approximate) log model evidence
family_names = {'Basic_RL', 'LR_Modulation'}; % display names for the two families
families{1} = [1];       % BasicRL -- column index into lme/BIC_matrix/model_names
families{2} = [2, 3];    % LR modulation models (RLsigma, BayesianAgent)

fprintf('Running family-wise model comparison (sigma fixed)...\n');
% Runs VBA_groupBMC (falling back to a manual approximation if that
% toolbox call fails) to get family- and model-level probabilities.
[results_summary, posterior, out] = fitSlider_ALLmodels.compute_family_model_comparison(lme, families, family_names, model_names);
save(fullfile(save_dir, 'family_wise_BMS_results_fixedSigma_both_multiSP.mat'), 'results_summary', 'posterior', 'out'); % save raw + summarized results
fprintf('\nResults saved to: family_wise_BMS_results_fixedSigma_both_multiSP.mat\n');

% Plots family probabilities/exceedance probabilities, individual model
% probabilities, a within-family stacked bar, and prints a text summary.
fitSlider_ALLmodels.plot_family_model_comparison(results_summary, 'Both condition, Sigma Fixed');

%% --- Perceptual condition ---
% Reload each model's per-subject NLLs for the Perceptual condition.
nll_basicRL = importdata(fullfile(save_dir, "nll_basicRL_fixedSigma_Perceptual_multiSP.mat"));
nll_RLsigma = importdata(fullfile(save_dir, "nll_RLsigma_fixedSigma_Perceptual_multiSP.mat"));
nll_bayesianAgent = importdata(fullfile(save_dir, "nll_bayesianAgent_fixedSigma_Perceptual_multiSP.mat"));
% Reloaded for the same reason as the Both-condition block above: these
% variable names were last overwritten by this script's own fitting
% blocks, so reloading from disk guarantees the Perceptual-condition fit.
params_basicRL = importdata(fullfile(save_dir, "params_basicRL_fixedSigma_Perceptual_multiSP.mat"));
params_RLsigma = importdata(fullfile(save_dir, "params_RLsigma_fixedSigma_Perceptual_multiSP.mat"));
params_bayesianAgent = importdata(fullfile(save_dir, "params_bayesianAgent_fixedSigma_Perceptual_multiSP.mat"));

disp('Computing AIC and BIC for all models (perceptual condition, sigma fixed)...');
% Number of trials per subject -- NOT a fixed 200: each condition (Both/
% Perceptual) only gets 4 of the task's 12 blocks (100 trials), and
% requireMuHat further drops any trials with a missing slider response,
% so this is computed per subject rather than assumed. Named per-condition
% (num_trials_perceptual, not a shared num_trials) for the same reason as
% the Both-condition block above.
num_trials_perceptual = NaN(numSubjs, 1);
for n = 1:numSubjs
    num_trials_perceptual(n) = length(preprocess_fitSlider(dataPerceptual, uniqueID(n)).mu_hat);
end
% Model parameter counts -- one fewer than fitReducedModelSpace.m, since
% sigma is fixed rather than estimated. [basicRL, RLsigma, BayesianAgent]
num_params = [2, 2, 1]; % free-parameter count per model, same column order as the NLLs below
model_names = {'basicRL','RLsigma','BayesianAgent'}; % display/column names, same column order as num_params
[AICBIC_table, bestModel_table, BMS_table] = fitSlider_ALLmodels.compute_model_comparison( ... % AIC/BIC/delta-BIC/bestModel/BMS in one call
    [nll_basicRL, nll_RLsigma, nll_bayesianAgent], model_names, num_params, num_trials_perceptual, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_fixedSigma_Perceptual_reducedMS_multiSP.mat'), AICBIC_table); % save per-subject AIC/BIC/delta-BIC
safe_saveall(fullfile(save_dir, 'bestModel_fixedSigma_Perceptual_multiSP.mat'), bestModel_table); % save each subject's best-fitting model
disp('All model parameters and AIC/BIC estimated and saved (perceptual condition, sigma fixed).');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_table, bestModel_table, BMS_table, model_names, 'Perceptual condition, Sigma Fixed'); % proportion-best-fit bar + BMS plots

% Plot fitted alpha/kappa/sigma across models, plus summed AIC/BIC and
% mean delta-BIC -- integrated from plotParams_reducedMS_fixedSigma.m.
bic_ax_perc_obj = fitSlider_ALLmodels.plot_fitted_params({params_basicRL, params_RLsigma, params_bayesianAgent}, model_names, AICBIC_table, 'Perceptual condition, Sigma Fixed', 'sigma (fixed, from perceptual-choice fit)');

%% Family-wise Model Comparison using VBA Toolbox (Perceptual)
BIC_matrix = [AICBIC_table.BIC_basicRL, ...
                  AICBIC_table.BIC_RLsigma, ...
                  AICBIC_table.BIC_BayesianAgent]; % numSubjs x 3 matrix of BIC values, one column per model
lme = -0.5 * BIC_matrix; % convert BIC to (approximate) log model evidence
family_names = {'Basic_RL', 'LR_Modulation'}; % display names for the two families
families{1} = [1];       % BasicRL -- column index into lme/BIC_matrix/model_names
families{2} = [2, 3];    % LR modulation models (RLsigma, BayesianAgent)

fprintf('Running family-wise model comparison (perceptual condition, sigma fixed)...\n');
[results_summary, posterior, out] = fitSlider_ALLmodels.compute_family_model_comparison(lme, families, family_names, model_names); % VBA_groupBMC (or manual fallback)
save(fullfile(save_dir, 'family_wise_BMS_results_fixedSigma_perceptual_multiSP.mat'), 'results_summary', 'posterior', 'out'); % save raw + summarized results
fprintf('\nResults saved to: family_wise_BMS_results_fixedSigma_perceptual_multiSP.mat\n');

fitSlider_ALLmodels.plot_family_model_comparison(results_summary, 'Perceptual condition, Sigma Fixed'); % family/model probability plots + text summary

%% ========================================================================
%  SECTION 2 OF 4: SUBJECTIVE PE, INTEGRATED
%  ------------------------------------------------------------------------
%  Same basicRL / RLsigma models and same fixed-sigma setup as Section 1,
%  but using nll_basicRL_integrated_subjectivePE / nll_RLsigma_VOI_
%  subjectivePE instead of nll_basicRL_integrated / nll_RLsigma_VOI: each
%  trial's delta-rule anchors to the subject's own actual previous slider
%  report (mu_hat(n-1)) rather than the model's internally-tracked running
%  belief -- see fitSlider_ALLmodels.m's nll_basicRL_integrated_
%  subjectivePE header for the full rationale. bayesianAgent has no
%  subjective-PE variant (its belief update, Agent.learn()'s polynomial-
%  coefficient tracking, isn't a simple alpha-driven delta rule), so it's
%  excluded from this section -- a 2-model comparison throughout.
%  ========================================================================

%% ----------------- Both condition: fitting ------------------------------
%% --- Basic RL model, subjective PE ---
if run_fitting
alphaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit alpha
kappaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit kappa
sigmaParameter = NaN(numSubjs, 1); % will hold each subject's fixed sigma
nll_basicRL_subjectivePE = NaN(numSubjs, 1); % will hold each subject's best NLL
lb = [0, 1]; % lower bounds for [alpha, kappa]
ub = [1, 100]; % upper bounds for [alpha, kappa]
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]); % random alpha starting points
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]); % random kappa starting points
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Basic RL subjective PE (Both, fixed sigma)'); % open/relabel the waitbar
progressQueue = parallel.pool.DataQueue; % fresh progress channel for this block
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Basic RL subjective PE (Both, fixed sigma)'));
parfor n = 1:numSubjs % fit every subject independently and in parallel

    subj = preprocess_fitSlider(dataBoth, uniqueID(n)); % this subject's trial-level data
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast); % recode rewards into the model's fixed frame
    sigma_n = fixedSigma(n); % this subject's fixed sigma
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated_subjectivePE( ... % subjective-PE version of basicRL's objective
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff); % params(1)=alpha, params(2)=kappa, sigma fixed in
    options = optimset('Display', 'off'); % keep fmincon quiet

    best_nll = Inf; % best NLL seen so far across restarts
    best_params = NaN(1,2); % [alpha, kappa] that achieved best_nll
    for sp = 1:n_startingPoints % try each random starting point
        start_params = [initAlpha(n,sp), initKappa(n,sp)]; % this restart's starting [alpha, kappa]
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options); % local optimization within [lb, ub]
        if nll < best_nll % keep this restart if it's the best so far
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]); % report progress
    end

    alphaParameter(n) = best_params(1); % store this subject's best-fit alpha
    kappaParameter(n) = best_params(2); % store this subject's best-fit kappa
    sigmaParameter(n) = sigma_n; % store this subject's fixed sigma
    nll_basicRL_subjectivePE(n) = best_nll; % store this subject's best NLL

end
params_basicRL_subjectivePE.alpha = alphaParameter; % bundle alphas into a struct
params_basicRL_subjectivePE.kappa = kappaParameter; % ...and kappas
params_basicRL_subjectivePE.sigma = sigmaParameter; % ...and sigmas
safe_saveall(fullfile(save_dir, 'params_basicRL_subjectivePEv2_fixedSigma_Both_multiSP.mat'), params_basicRL_subjectivePE); % save fitted parameters
safe_saveall(fullfile(save_dir, 'nll_basicRL_subjectivePEv2_fixedSigma_Both_multiSP.mat'), nll_basicRL_subjectivePE); % save per-subject NLLs
end

%% --- RL + Est. Sensitivity model, subjective PE ---
if run_fitting
alphaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit alpha
kappaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit kappa
sigmaParameter = NaN(numSubjs, 1); % will hold each subject's fixed sigma
nll_RLsigma_subjectivePE = NaN(numSubjs, 1); % will hold each subject's best NLL
lb = [0, 1]; % lower bounds for [alpha, kappa]
ub = [1, 100]; % upper bounds for [alpha, kappa]
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]); % random alpha starting points
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]); % random kappa starting points
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'RL + Est. Sensitivity subjective PE (Both, fixed sigma)'); % open/relabel the waitbar
progressQueue = parallel.pool.DataQueue; % fresh progress channel for this block
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'RL + Est. Sensitivity subjective PE (Both, fixed sigma)'));
parfor n = 1:numSubjs % fit every subject independently and in parallel
    subj = preprocess_fitSlider(dataBoth, uniqueID(n)); % this subject's trial-level data
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast); % recode rewards into the model's fixed frame
    sigma_n = fixedSigma(n); % this subject's fixed sigma
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI_subjectivePE( ... % subjective-PE version of RLsigma's objective
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff); % params(1)=alpha, params(2)=kappa, sigma fixed in
    options = optimset('Display', 'off'); % keep fmincon quiet

    best_nll = Inf; % best NLL seen so far across restarts
    best_params = NaN(1,2); % [alpha, kappa] that achieved best_nll
    for sp = 1:n_startingPoints % try each random starting point
        start_params = [initAlpha(n,sp), initKappa(n,sp)]; % this restart's starting [alpha, kappa]
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options); % local optimization within [lb, ub]
        if nll < best_nll % keep this restart if it's the best so far
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]); % report progress
    end

    alphaParameter(n) = best_params(1); % store this subject's best-fit alpha
    kappaParameter(n) = best_params(2); % store this subject's best-fit kappa
    sigmaParameter(n) = sigma_n; % store this subject's fixed sigma
    nll_RLsigma_subjectivePE(n) = best_nll; % store this subject's best NLL
end
params_RLsigma_subjectivePE.alpha = alphaParameter; % bundle alphas into a struct
params_RLsigma_subjectivePE.sigma = sigmaParameter; % ...and sigmas
params_RLsigma_subjectivePE.kappa = kappaParameter; % ...and kappas
safe_saveall(fullfile(save_dir, 'params_RLsigma_subjectivePEv2_fixedSigma_Both_multiSP.mat'), params_RLsigma_subjectivePE); % save fitted parameters
safe_saveall(fullfile(save_dir, 'nll_RLsigma_subjectivePEv2_fixedSigma_Both_multiSP.mat'), nll_RLsigma_subjectivePE); % save per-subject NLLs
end

%% ----------------- Perceptual condition: fitting -------------------------
%% --- Basic RL model, subjective PE (Perceptual) ---
if run_fitting
alphaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit alpha
kappaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit kappa
sigmaParameter = NaN(numSubjs, 1); % will hold each subject's fixed sigma
nll_basicRL_subjectivePE = NaN(numSubjs, 1); % will hold each subject's best NLL
lb = [0, 1]; % lower bounds for [alpha, kappa]
ub = [1, 100]; % upper bounds for [alpha, kappa]
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]); % random alpha starting points
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]); % random kappa starting points
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Basic RL subjective PE (Perceptual, fixed sigma)'); % open/relabel the waitbar
progressQueue = parallel.pool.DataQueue; % fresh progress channel for this block
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Basic RL subjective PE (Perceptual, fixed sigma)'));
parfor n = 1:numSubjs % fit every subject independently and in parallel
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n)); % this subject's trial-level data, Perceptual condition
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast); % recode rewards into the model's fixed frame
    sigma_n = fixedSigma(n); % this subject's fixed sigma
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated_subjectivePE( ... % subjective-PE version of basicRL's objective
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff); % params(1)=alpha, params(2)=kappa, sigma fixed in
    options = optimset('Display', 'off'); % keep fmincon quiet

    best_nll = Inf; % best NLL seen so far across restarts
    best_params = NaN(1,2); % [alpha, kappa] that achieved best_nll
    for sp = 1:n_startingPoints % try each random starting point
        start_params = [initAlpha(n,sp), initKappa(n,sp)]; % this restart's starting [alpha, kappa]
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options); % local optimization within [lb, ub]
        if nll < best_nll % keep this restart if it's the best so far
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]); % report progress
    end

    alphaParameter(n) = best_params(1); % store this subject's best-fit alpha
    kappaParameter(n) = best_params(2); % store this subject's best-fit kappa
    sigmaParameter(n) = sigma_n; % store this subject's fixed sigma
    nll_basicRL_subjectivePE(n) = best_nll; % store this subject's best NLL
end
params_basicRL_subjectivePE.alpha = alphaParameter; % bundle alphas into a struct
params_basicRL_subjectivePE.kappa = kappaParameter; % ...and kappas
params_basicRL_subjectivePE.sigma = sigmaParameter; % ...and sigmas
safe_saveall(fullfile(save_dir, 'params_basicRL_subjectivePEv2_fixedSigma_Perceptual_multiSP.mat'), params_basicRL_subjectivePE); % save fitted parameters
safe_saveall(fullfile(save_dir, 'nll_basicRL_subjectivePEv2_fixedSigma_Perceptual_multiSP.mat'), nll_basicRL_subjectivePE); % save per-subject NLLs
end

%% --- RL + Est. Sensitivity model, subjective PE (Perceptual) ---
if run_fitting
alphaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit alpha
kappaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit kappa
sigmaParameter = NaN(numSubjs, 1); % will hold each subject's fixed sigma
nll_RLsigma_subjectivePE = NaN(numSubjs, 1); % will hold each subject's best NLL
lb = [0, 1]; % lower bounds for [alpha, kappa]
ub = [1, 100]; % upper bounds for [alpha, kappa]
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]); % random alpha starting points
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]); % random kappa starting points
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'RL + Est. Sensitivity subjective PE (Perceptual, fixed sigma)'); % open/relabel the waitbar
progressQueue = parallel.pool.DataQueue; % fresh progress channel for this block
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'RL + Est. Sensitivity subjective PE (Perceptual, fixed sigma)'));
parfor n = 1:numSubjs % fit every subject independently and in parallel
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n)); % this subject's trial-level data, Perceptual condition
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast); % recode rewards into the model's fixed frame
    sigma_n = fixedSigma(n); % this subject's fixed sigma
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI_subjectivePE( ... % subjective-PE version of RLsigma's objective
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff); % params(1)=alpha, params(2)=kappa, sigma fixed in
    options = optimset('Display', 'off'); % keep fmincon quiet

    best_nll = Inf; % best NLL seen so far across restarts
    best_params = NaN(1,2); % [alpha, kappa] that achieved best_nll
    for sp = 1:n_startingPoints % try each random starting point
        start_params = [initAlpha(n,sp), initKappa(n,sp)]; % this restart's starting [alpha, kappa]
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options); % local optimization within [lb, ub]
        if nll < best_nll % keep this restart if it's the best so far
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]); % report progress
    end

    alphaParameter(n) = best_params(1); % store this subject's best-fit alpha
    kappaParameter(n) = best_params(2); % store this subject's best-fit kappa
    sigmaParameter(n) = sigma_n; % store this subject's fixed sigma
    nll_RLsigma_subjectivePE(n) = best_nll; % store this subject's best NLL
end
params_RLsigma_subjectivePE.alpha = alphaParameter; % bundle alphas into a struct
params_RLsigma_subjectivePE.sigma = sigmaParameter; % ...and sigmas
params_RLsigma_subjectivePE.kappa = kappaParameter; % ...and kappas
safe_saveall(fullfile(save_dir, 'params_RLsigma_subjectivePEv2_fixedSigma_Perceptual_multiSP.mat'), params_RLsigma_subjectivePE); % save fitted parameters
safe_saveall(fullfile(save_dir, 'nll_RLsigma_subjectivePEv2_fixedSigma_Perceptual_multiSP.mat'), nll_RLsigma_subjectivePE); % save per-subject NLLs
end

fitSlider_ALLmodels.progress_bar('close'); % done with Sections 1-2 fitting -- close the waitbar window (safe no-op if run_fitting was false)

%% ----------------- Model comparison ---------------------------------------
%  2-way AIC/BIC/bestModel/BMS comparison between the two subjective-PE
%  models (basicRL_subjectivePE vs RLsigma_subjectivePE) fit above, kept
%  fully separate from the 3-way objective-PE comparison in Section 1. No
%  family-wise VBA_groupBMC step here -- with only two models, a
%  family-wise grouping would just duplicate the model-level results
%  below, so it's omitted.
%% --- Both condition ---
nll_basicRL_subjectivePE = importdata(fullfile(save_dir, "nll_basicRL_subjectivePEv2_fixedSigma_Both_multiSP.mat")); % reload each subject's NLL
nll_RLsigma_subjectivePE = importdata(fullfile(save_dir, "nll_RLsigma_subjectivePEv2_fixedSigma_Both_multiSP.mat")); % under the subjective-PE models
% Reloaded (not reused in-memory) for the same reason as every params_*
% reload above: these variable names get overwritten by the Perceptual-
% condition subjective-PE fitting block further down the script.
params_basicRL_subjectivePE = importdata(fullfile(save_dir, "params_basicRL_subjectivePEv2_fixedSigma_Both_multiSP.mat"));
params_RLsigma_subjectivePE = importdata(fullfile(save_dir, "params_RLsigma_subjectivePEv2_fixedSigma_Both_multiSP.mat"));

disp('Computing AIC and BIC for subjective PE models (sigma fixed)...');
% num_trials_both already computed per subject in the Both-condition
% objective-PE AIC/BIC section above, under its own condition-specific
% name -- safe to reuse as-is since nothing else overwrites it (unlike the
% old shared num_trials, which the Perceptual-condition objective-PE
% section further up would silently overwrite with Perceptual's counts).
num_params_subjectivePE = [2, 2]; % both subjective-PE models have 2 free params (alpha, kappa)
model_names_subjectivePE = {'basicRL_subjectivePE','RLsigma_subjectivePE'}; % display/column names for this 2-way comparison
[AICBIC_subjectivePE_table, bestModel_subjectivePE_table, BMS_subjectivePE_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_subjectivePE, nll_RLsigma_subjectivePE], model_names_subjectivePE, num_params_subjectivePE, num_trials_both, uniqueID); % same generic comparison helper as above, just with 2 models instead of 3
safe_saveall(fullfile(save_dir, 'AICBIC_subjectivePEv2_fixedSigma_Both_reducedMS_multiSP.mat'), AICBIC_subjectivePE_table); % save per-subject AIC/BIC/delta-BIC
safe_saveall(fullfile(save_dir, 'bestModel_subjectivePEv2_fixedSigma_Both_multiSP.mat'), bestModel_subjectivePE_table); % save each subject's best-fitting subjective-PE model
disp('Subjective PE model AIC/BIC estimated and saved (sigma fixed).');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_subjectivePE_table, bestModel_subjectivePE_table, BMS_subjectivePE_table, model_names_subjectivePE, 'Both condition, Subjective PE, Sigma Fixed'); % same plots as the 3-way comparison, scaled to 2 models

% Plot fitted alpha/kappa/sigma across models, plus summed AIC/BIC and
% mean delta-BIC -- integrated from plotParams_reducedMS_fixedSigma.m.
bic_ax_both_subj = fitSlider_ALLmodels.plot_fitted_params({params_basicRL_subjectivePE, params_RLsigma_subjectivePE}, model_names_subjectivePE, AICBIC_subjectivePE_table, 'Both condition, Subjective PE, Sigma Fixed', 'sigma (fixed, from perceptual-choice fit)');

%% --- Perceptual condition ---
nll_basicRL_subjectivePE = importdata(fullfile(save_dir, "nll_basicRL_subjectivePEv2_fixedSigma_Perceptual_multiSP.mat")); % reload each subject's NLL
nll_RLsigma_subjectivePE = importdata(fullfile(save_dir, "nll_RLsigma_subjectivePEv2_fixedSigma_Perceptual_multiSP.mat")); % under the subjective-PE models
% Reloaded for the same reason as the Both-condition block above: these
% variable names were last overwritten by this script's own fitting
% blocks, so reloading from disk guarantees the Perceptual-condition fit.
params_basicRL_subjectivePE = importdata(fullfile(save_dir, "params_basicRL_subjectivePEv2_fixedSigma_Perceptual_multiSP.mat"));
params_RLsigma_subjectivePE = importdata(fullfile(save_dir, "params_RLsigma_subjectivePEv2_fixedSigma_Perceptual_multiSP.mat"));

disp('Computing AIC and BIC for subjective PE models (Perceptual, sigma fixed)...');
% num_trials_perceptual already computed per subject in the Perceptual-
% condition objective-PE AIC/BIC section above, under its own condition-
% specific name -- safe to reuse as-is (see the Both-condition block
% above for why the old shared num_trials wasn't).
num_params_subjectivePE = [2, 2]; % both subjective-PE models have 2 free params (alpha, kappa)
model_names_subjectivePE = {'basicRL_subjectivePE','RLsigma_subjectivePE'}; % display/column names for this 2-way comparison
[AICBIC_subjectivePE_table, bestModel_subjectivePE_table, BMS_subjectivePE_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_subjectivePE, nll_RLsigma_subjectivePE], model_names_subjectivePE, num_params_subjectivePE, num_trials_perceptual, uniqueID); % same generic comparison helper as above, with 2 models
safe_saveall(fullfile(save_dir, 'AICBIC_subjectivePEv2_fixedSigma_Perceptual_reducedMS_multiSP.mat'), AICBIC_subjectivePE_table); % save per-subject AIC/BIC/delta-BIC
safe_saveall(fullfile(save_dir, 'bestModel_subjectivePEv2_fixedSigma_Perceptual_multiSP.mat'), bestModel_subjectivePE_table); % save each subject's best-fitting subjective-PE model
disp('Subjective PE model AIC/BIC estimated and saved (Perceptual, sigma fixed).');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_subjectivePE_table, bestModel_subjectivePE_table, BMS_subjectivePE_table, model_names_subjectivePE, 'Perceptual condition, Subjective PE, Sigma Fixed'); % same plots as the 3-way comparison, scaled to 2 models

% Plot fitted alpha/kappa/sigma across models, plus summed AIC/BIC and
% mean delta-BIC -- integrated from plotParams_reducedMS_fixedSigma.m.
bic_ax_perc_subj = fitSlider_ALLmodels.plot_fitted_params({params_basicRL_subjectivePE, params_RLsigma_subjectivePE}, model_names_subjectivePE, AICBIC_subjectivePE_table, 'Perceptual condition, Subjective PE, Sigma Fixed', 'sigma (fixed, from perceptual-choice fit)');

%% ========================================================================
%  SECTION 3 OF 4: OBJECTIVE PE, NO INTEGRATION
%  ------------------------------------------------------------------------
%  Same basicRL / RLsigma models and same fixed-sigma setup as Section 1,
%  but using each model's point-observation (non-integrated) likelihood
%  instead of its integrated one: nll_basicRL_pointObs instead of
%  nll_basicRL_integrated, and nll_RLsigma instead of nll_RLsigma_VOI.
%  Belief each trial comes from a single point estimate of the observation
%  (normcdf(condiff(n), 0, sigma)) rather than integrated over a
%  discretized grid of hypothetical observations -- see fitSlider_
%  ALLmodels.m's nll_basicRL_pointObs header for the full rationale.
%  bayesianAgent is excluded (its belief update goes through the Agent
%  class's own mechanism, not the set_o grid this section's integration
%  concerns) -- a 2-model comparison throughout.
%  ========================================================================

%% ----------------- Both condition: fitting ------------------------------
%% --- Basic RL model, no integration ---
if run_fitting
alphaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit alpha
kappaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit kappa
sigmaParameter = NaN(numSubjs, 1); % will hold each subject's fixed sigma
nll_basicRL_noIntegration = NaN(numSubjs, 1); % will hold each subject's best NLL
lb = [0, 1]; % lower bounds for [alpha, kappa]
ub = [1, 100]; % upper bounds for [alpha, kappa]
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]); % random alpha starting points
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]); % random kappa starting points
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Basic RL, no integration (Both, fixed sigma)'); % open/relabel the waitbar
progressQueue = parallel.pool.DataQueue; % fresh progress channel for this block
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Basic RL, no integration (Both, fixed sigma)'));
parfor n = 1:numSubjs % fit every subject independently and in parallel
    subj = preprocess_fitSlider(dataBoth, uniqueID(n)); % this subject's trial-level data
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast); % recode rewards into the model's fixed frame
    sigma_n = fixedSigma(n); % this subject's fixed sigma
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_pointObs( ... % point-observation (no integration) basicRL objective
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff); % params(1)=alpha, params(2)=kappa, sigma fixed in
    options = optimset('Display', 'off'); % keep fmincon quiet

    best_nll = Inf; % best NLL seen so far across restarts
    best_params = NaN(1,2); % [alpha, kappa] that achieved best_nll
    for sp = 1:n_startingPoints % try each random starting point
        start_params = [initAlpha(n,sp), initKappa(n,sp)]; % this restart's starting [alpha, kappa]
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options); % local optimization within [lb, ub]
        if nll < best_nll % keep this restart if it's the best so far
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]); % report progress
    end

    alphaParameter(n) = best_params(1); % store this subject's best-fit alpha
    kappaParameter(n) = best_params(2); % store this subject's best-fit kappa
    sigmaParameter(n) = sigma_n; % store this subject's fixed sigma
    nll_basicRL_noIntegration(n) = best_nll; % store this subject's best NLL
end
params_basicRL_noIntegration.alpha = alphaParameter; % bundle alphas into a struct
params_basicRL_noIntegration.kappa = kappaParameter; % ...and kappas
params_basicRL_noIntegration.sigma = sigmaParameter; % ...and sigmas
safe_saveall(fullfile(save_dir, 'params_basicRL_noIntegration_fixedSigma_Both_multiSP.mat'), params_basicRL_noIntegration); % save fitted parameters
safe_saveall(fullfile(save_dir, 'nll_basicRL_noIntegration_fixedSigma_Both_multiSP.mat'), nll_basicRL_noIntegration); % save per-subject NLLs
end

%% --- RL + Est. Sensitivity model, no integration ---
if run_fitting
alphaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit alpha
kappaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit kappa
sigmaParameter = NaN(numSubjs, 1); % will hold each subject's fixed sigma
nll_RLsigma_noIntegration = NaN(numSubjs, 1); % will hold each subject's best NLL
lb = [0, 1]; % lower bounds for [alpha, kappa]
ub = [1, 100]; % upper bounds for [alpha, kappa]
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]); % random alpha starting points
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]); % random kappa starting points
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'RL + Est. Sensitivity, no integration (Both, fixed sigma)'); % open/relabel the waitbar
progressQueue = parallel.pool.DataQueue; % fresh progress channel for this block
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'RL + Est. Sensitivity, no integration (Both, fixed sigma)'));
parfor n = 1:numSubjs % fit every subject independently and in parallel
    subj = preprocess_fitSlider(dataBoth, uniqueID(n)); % this subject's trial-level data
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast); % recode rewards into the model's fixed frame
    sigma_n = fixedSigma(n); % this subject's fixed sigma
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma( ... % point-observation (no integration, already-existing) RLsigma objective
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff); % params(1)=alpha, params(2)=kappa, sigma fixed in
    options = optimset('Display', 'off'); % keep fmincon quiet

    best_nll = Inf; % best NLL seen so far across restarts
    best_params = NaN(1,2); % [alpha, kappa] that achieved best_nll
    for sp = 1:n_startingPoints % try each random starting point
        start_params = [initAlpha(n,sp), initKappa(n,sp)]; % this restart's starting [alpha, kappa]
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options); % local optimization within [lb, ub]
        if nll < best_nll % keep this restart if it's the best so far
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]); % report progress
    end

    alphaParameter(n) = best_params(1); % store this subject's best-fit alpha
    kappaParameter(n) = best_params(2); % store this subject's best-fit kappa
    sigmaParameter(n) = sigma_n; % store this subject's fixed sigma
    nll_RLsigma_noIntegration(n) = best_nll; % store this subject's best NLL
end
params_RLsigma_noIntegration.alpha = alphaParameter; % bundle alphas into a struct
params_RLsigma_noIntegration.sigma = sigmaParameter; % ...and sigmas
params_RLsigma_noIntegration.kappa = kappaParameter; % ...and kappas
safe_saveall(fullfile(save_dir, 'params_RLsigma_noIntegration_fixedSigma_Both_multiSP.mat'), params_RLsigma_noIntegration); % save fitted parameters
safe_saveall(fullfile(save_dir, 'nll_RLsigma_noIntegration_fixedSigma_Both_multiSP.mat'), nll_RLsigma_noIntegration); % save per-subject NLLs
end

%% ----------------- Perceptual condition: fitting -------------------------
%% --- Basic RL model, no integration (Perceptual) ---
if run_fitting
alphaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit alpha
kappaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit kappa
sigmaParameter = NaN(numSubjs, 1); % will hold each subject's fixed sigma
nll_basicRL_noIntegration = NaN(numSubjs, 1); % will hold each subject's best NLL
lb = [0, 1]; % lower bounds for [alpha, kappa]
ub = [1, 100]; % upper bounds for [alpha, kappa]
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]); % random alpha starting points
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]); % random kappa starting points
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Basic RL, no integration (Perceptual, fixed sigma)'); % open/relabel the waitbar
progressQueue = parallel.pool.DataQueue; % fresh progress channel for this block
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Basic RL, no integration (Perceptual, fixed sigma)'));
parfor n = 1:numSubjs % fit every subject independently and in parallel
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n)); % this subject's trial-level data, Perceptual condition
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast); % recode rewards into the model's fixed frame
    sigma_n = fixedSigma(n); % this subject's fixed sigma
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_pointObs( ... % point-observation (no integration) basicRL objective
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff); % params(1)=alpha, params(2)=kappa, sigma fixed in
    options = optimset('Display', 'off'); % keep fmincon quiet

    best_nll = Inf; % best NLL seen so far across restarts
    best_params = NaN(1,2); % [alpha, kappa] that achieved best_nll
    for sp = 1:n_startingPoints % try each random starting point
        start_params = [initAlpha(n,sp), initKappa(n,sp)]; % this restart's starting [alpha, kappa]
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options); % local optimization within [lb, ub]
        if nll < best_nll % keep this restart if it's the best so far
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]); % report progress
    end

    alphaParameter(n) = best_params(1); % store this subject's best-fit alpha
    kappaParameter(n) = best_params(2); % store this subject's best-fit kappa
    sigmaParameter(n) = sigma_n; % store this subject's fixed sigma
    nll_basicRL_noIntegration(n) = best_nll; % store this subject's best NLL
end
params_basicRL_noIntegration.alpha = alphaParameter; % bundle alphas into a struct
params_basicRL_noIntegration.kappa = kappaParameter; % ...and kappas
params_basicRL_noIntegration.sigma = sigmaParameter; % ...and sigmas
safe_saveall(fullfile(save_dir, 'params_basicRL_noIntegration_fixedSigma_Perceptual_multiSP.mat'), params_basicRL_noIntegration); % save fitted parameters
safe_saveall(fullfile(save_dir, 'nll_basicRL_noIntegration_fixedSigma_Perceptual_multiSP.mat'), nll_basicRL_noIntegration); % save per-subject NLLs
end

%% --- RL + Est. Sensitivity model, no integration (Perceptual) ---
if run_fitting
alphaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit alpha
kappaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit kappa
sigmaParameter = NaN(numSubjs, 1); % will hold each subject's fixed sigma
nll_RLsigma_noIntegration = NaN(numSubjs, 1); % will hold each subject's best NLL
lb = [0, 1]; % lower bounds for [alpha, kappa]
ub = [1, 100]; % upper bounds for [alpha, kappa]
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]); % random alpha starting points
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]); % random kappa starting points
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'RL + Est. Sensitivity, no integration (Perceptual, fixed sigma)'); % open/relabel the waitbar
progressQueue = parallel.pool.DataQueue; % fresh progress channel for this block
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'RL + Est. Sensitivity, no integration (Perceptual, fixed sigma)'));
parfor n = 1:numSubjs % fit every subject independently and in parallel
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n)); % this subject's trial-level data, Perceptual condition
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast); % recode rewards into the model's fixed frame
    sigma_n = fixedSigma(n); % this subject's fixed sigma
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma( ... % point-observation (no integration, already-existing) RLsigma objective
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff); % params(1)=alpha, params(2)=kappa, sigma fixed in
    options = optimset('Display', 'off'); % keep fmincon quiet

    best_nll = Inf; % best NLL seen so far across restarts
    best_params = NaN(1,2); % [alpha, kappa] that achieved best_nll
    for sp = 1:n_startingPoints % try each random starting point
        start_params = [initAlpha(n,sp), initKappa(n,sp)]; % this restart's starting [alpha, kappa]
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options); % local optimization within [lb, ub]
        if nll < best_nll % keep this restart if it's the best so far
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]); % report progress
    end

    alphaParameter(n) = best_params(1); % store this subject's best-fit alpha
    kappaParameter(n) = best_params(2); % store this subject's best-fit kappa
    sigmaParameter(n) = sigma_n; % store this subject's fixed sigma
    nll_RLsigma_noIntegration(n) = best_nll; % store this subject's best NLL
end
params_RLsigma_noIntegration.alpha = alphaParameter; % bundle alphas into a struct
params_RLsigma_noIntegration.sigma = sigmaParameter; % ...and sigmas
params_RLsigma_noIntegration.kappa = kappaParameter; % ...and kappas
safe_saveall(fullfile(save_dir, 'params_RLsigma_noIntegration_fixedSigma_Perceptual_multiSP.mat'), params_RLsigma_noIntegration); % save fitted parameters
safe_saveall(fullfile(save_dir, 'nll_RLsigma_noIntegration_fixedSigma_Perceptual_multiSP.mat'), nll_RLsigma_noIntegration); % save per-subject NLLs
end

%% ----------------- Model comparison ---------------------------------------
%% --- Both condition ---
nll_basicRL_noIntegration = importdata(fullfile(save_dir, "nll_basicRL_noIntegration_fixedSigma_Both_multiSP.mat"));
nll_RLsigma_noIntegration = importdata(fullfile(save_dir, "nll_RLsigma_noIntegration_fixedSigma_Both_multiSP.mat"));
params_basicRL_noIntegration = importdata(fullfile(save_dir, "params_basicRL_noIntegration_fixedSigma_Both_multiSP.mat"));
params_RLsigma_noIntegration = importdata(fullfile(save_dir, "params_RLsigma_noIntegration_fixedSigma_Both_multiSP.mat"));

disp('Computing AIC and BIC for no-integration models (sigma fixed)...');
num_trials_both = NaN(numSubjs, 1);
for n = 1:numSubjs
    num_trials_both(n) = length(preprocess_fitSlider(dataBoth, uniqueID(n)).mu_hat);
end
num_params_noIntegration = [2, 2]; % both no-integration models have 2 free params (alpha, kappa)
model_names_noIntegration = {'basicRL_noIntegration','RLsigma_noIntegration'};
[AICBIC_noIntegration_table, bestModel_noIntegration_table, BMS_noIntegration_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_noIntegration, nll_RLsigma_noIntegration], model_names_noIntegration, num_params_noIntegration, num_trials_both, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_noIntegration_fixedSigma_Both_reducedMS_multiSP.mat'), AICBIC_noIntegration_table);
safe_saveall(fullfile(save_dir, 'bestModel_noIntegration_fixedSigma_Both_multiSP.mat'), bestModel_noIntegration_table);
disp('No-integration model AIC/BIC estimated and saved (sigma fixed).');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_noIntegration_table, bestModel_noIntegration_table, BMS_noIntegration_table, model_names_noIntegration, 'Both condition, No Integration, Sigma Fixed');

bic_ax_both_obj_noInt = fitSlider_ALLmodels.plot_fitted_params({params_basicRL_noIntegration, params_RLsigma_noIntegration}, model_names_noIntegration, AICBIC_noIntegration_table, 'Both condition, No Integration, Sigma Fixed', 'sigma (fixed, from perceptual-choice fit)');

%% --- Perceptual condition ---
nll_basicRL_noIntegration = importdata(fullfile(save_dir, "nll_basicRL_noIntegration_fixedSigma_Perceptual_multiSP.mat"));
nll_RLsigma_noIntegration = importdata(fullfile(save_dir, "nll_RLsigma_noIntegration_fixedSigma_Perceptual_multiSP.mat"));
params_basicRL_noIntegration = importdata(fullfile(save_dir, "params_basicRL_noIntegration_fixedSigma_Perceptual_multiSP.mat"));
params_RLsigma_noIntegration = importdata(fullfile(save_dir, "params_RLsigma_noIntegration_fixedSigma_Perceptual_multiSP.mat"));

disp('Computing AIC and BIC for no-integration models (Perceptual, sigma fixed)...');
num_trials_perceptual = NaN(numSubjs, 1);
for n = 1:numSubjs
    num_trials_perceptual(n) = length(preprocess_fitSlider(dataPerceptual, uniqueID(n)).mu_hat);
end
num_params_noIntegration = [2, 2];
model_names_noIntegration = {'basicRL_noIntegration','RLsigma_noIntegration'};
[AICBIC_noIntegration_table, bestModel_noIntegration_table, BMS_noIntegration_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_noIntegration, nll_RLsigma_noIntegration], model_names_noIntegration, num_params_noIntegration, num_trials_perceptual, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_noIntegration_fixedSigma_Perceptual_reducedMS_multiSP.mat'), AICBIC_noIntegration_table);
safe_saveall(fullfile(save_dir, 'bestModel_noIntegration_fixedSigma_Perceptual_multiSP.mat'), bestModel_noIntegration_table);
disp('No-integration model AIC/BIC estimated and saved (Perceptual, sigma fixed).');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_noIntegration_table, bestModel_noIntegration_table, BMS_noIntegration_table, model_names_noIntegration, 'Perceptual condition, No Integration, Sigma Fixed');

bic_ax_perc_obj_noInt = fitSlider_ALLmodels.plot_fitted_params({params_basicRL_noIntegration, params_RLsigma_noIntegration}, model_names_noIntegration, AICBIC_noIntegration_table, 'Perceptual condition, No Integration, Sigma Fixed', 'sigma (fixed, from perceptual-choice fit)');

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
alphaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit alpha
kappaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit kappa
sigmaParameter = NaN(numSubjs, 1); % will hold each subject's fixed sigma
nll_basicRL_subjectivePE_noIntegration = NaN(numSubjs, 1); % will hold each subject's best NLL
lb = [0, 1]; % lower bound for [alpha, kappa]
ub = [1, 100]; % upper bounds for [alpha, kappa]
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]); % random alpha starting points
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]); % random kappa starting points
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Basic RL subjective PE, no integration (Both, fixed sigma)'); % open/relabel the waitbar
progressQueue = parallel.pool.DataQueue; % fresh progress channel for this block
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Basic RL subjective PE, no integration (Both, fixed sigma)'));
parfor n = 1:numSubjs % fit every subject independently and in parallel
    subj = preprocess_fitSlider(dataBoth, uniqueID(n)); % this subject's trial-level data
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast); % recode rewards into the model's fixed frame
    sigma_n = fixedSigma(n); % this subject's fixed sigma
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_pointObs_subjectivePE( ... % point-observation subjective-PE version of basicRL's objective
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff); % params(1)=alpha, params(2)=kappa, sigma fixed in
    options = optimset('Display', 'off'); % keep fmincon quiet

    best_nll = Inf; % best NLL seen so far across restarts
    best_params = NaN(1,2); % [alpha, kappa] that achieved best_nll
    for sp = 1:n_startingPoints % try each random starting point
        start_params = [initAlpha(n,sp), initKappa(n,sp)]; % this restart's starting [alpha, kappa]
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options); % local optimization within [lb, ub]
        if nll < best_nll % keep this restart if it's the best so far
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]); % report progress
    end

    alphaParameter(n) = best_params(1); % store this subject's best-fit alpha
    kappaParameter(n) = best_params(2); % store this subject's best-fit kappa
    sigmaParameter(n) = sigma_n; % store this subject's fixed sigma
    nll_basicRL_subjectivePE_noIntegration(n) = best_nll; % store this subject's best NLL
end
params_basicRL_subjectivePE_noIntegration.alpha = alphaParameter; % bundle alphas into a struct
params_basicRL_subjectivePE_noIntegration.kappa = kappaParameter; % ...and kappas
params_basicRL_subjectivePE_noIntegration.sigma = sigmaParameter; % ...and sigmas
safe_saveall(fullfile(save_dir, 'params_basicRL_subjectivePE_noIntegration_fixedSigma_Both_multiSP.mat'), params_basicRL_subjectivePE_noIntegration); % save fitted parameters
safe_saveall(fullfile(save_dir, 'nll_basicRL_subjectivePE_noIntegration_fixedSigma_Both_multiSP.mat'), nll_basicRL_subjectivePE_noIntegration); % save per-subject NLLs
end

%% --- RL + Est. Sensitivity model, subjective PE, no integration ---
if run_fitting
alphaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit alpha
kappaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit kappa
sigmaParameter = NaN(numSubjs, 1); % will hold each subject's fixed sigma
nll_RLsigma_subjectivePE_noIntegration = NaN(numSubjs, 1); % will hold each subject's best NLL
lb = [0, 1]; % lower bounds for [alpha, kappa]
ub = [1, 100]; % upper bounds for [alpha, kappa]
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]); % random alpha starting points
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]); % random kappa starting points
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'RL + Est. Sensitivity subjective PE, no integration (Both, fixed sigma)'); % open/relabel the waitbar
progressQueue = parallel.pool.DataQueue; % fresh progress channel for this block
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'RL + Est. Sensitivity subjective PE, no integration (Both, fixed sigma)'));
parfor n = 1:numSubjs % fit every subject independently and in parallel
    subj = preprocess_fitSlider(dataBoth, uniqueID(n)); % this subject's trial-level data
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast); % recode rewards into the model's fixed frame
    sigma_n = fixedSigma(n); % this subject's fixed sigma
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_subjectivePE( ... % point-observation subjective-PE version of RLsigma's objective
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff); % params(1)=alpha, params(2)=kappa, sigma fixed in
    options = optimset('Display', 'off'); % keep fmincon quiet

    best_nll = Inf; % best NLL seen so far across restarts
    best_params = NaN(1,2); % [alpha, kappa] that achieved best_nll
    for sp = 1:n_startingPoints % try each random starting point
        start_params = [initAlpha(n,sp), initKappa(n,sp)]; % this restart's starting [alpha, kappa]
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options); % local optimization within [lb, ub]
        if nll < best_nll % keep this restart if it's the best so far
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]); % report progress
    end

    alphaParameter(n) = best_params(1); % store this subject's best-fit alpha
    kappaParameter(n) = best_params(2); % store this subject's best-fit kappa
    sigmaParameter(n) = sigma_n; % store this subject's fixed sigma
    nll_RLsigma_subjectivePE_noIntegration(n) = best_nll; % store this subject's best NLL
end
params_RLsigma_subjectivePE_noIntegration.alpha = alphaParameter; % bundle alphas into a struct
params_RLsigma_subjectivePE_noIntegration.sigma = sigmaParameter; % ...and sigmas
params_RLsigma_subjectivePE_noIntegration.kappa = kappaParameter; % ...and kappas
safe_saveall(fullfile(save_dir, 'params_RLsigma_subjectivePE_noIntegration_fixedSigma_Both_multiSP.mat'), params_RLsigma_subjectivePE_noIntegration); % save fitted parameters
safe_saveall(fullfile(save_dir, 'nll_RLsigma_subjectivePE_noIntegration_fixedSigma_Both_multiSP.mat'), nll_RLsigma_subjectivePE_noIntegration); % save per-subject NLLs
end

%% ----------------- Perceptual condition: fitting -------------------------
%% --- Basic RL model, subjective PE, no integration (Perceptual) ---
if run_fitting
alphaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit alpha
kappaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit kappa
sigmaParameter = NaN(numSubjs, 1); % will hold each subject's fixed sigma
nll_basicRL_subjectivePE_noIntegration = NaN(numSubjs, 1); % will hold each subject's best NLL
lb = [0, 1]; % lower bounds for [alpha, kappa]
ub = [1, 100]; % upper bounds for [alpha, kappa]
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]); % random alpha starting points
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]); % random kappa starting points
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Basic RL subjective PE, no integration (Perceptual, fixed sigma)'); % open/relabel the waitbar
progressQueue = parallel.pool.DataQueue; % fresh progress channel for this block
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'Basic RL subjective PE, no integration (Perceptual, fixed sigma)'));
parfor n = 1:numSubjs % fit every subject independently and in parallel
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n)); % this subject's trial-level data, Perceptual condition
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast); % recode rewards into the model's fixed frame
    sigma_n = fixedSigma(n); % this subject's fixed sigma
    nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_pointObs_subjectivePE( ... % point-observation subjective-PE version of basicRL's objective
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff); % params(1)=alpha, params(2)=kappa, sigma fixed in
    options = optimset('Display', 'off'); % keep fmincon quiet

    best_nll = Inf; % best NLL seen so far across restarts
    best_params = NaN(1,2); % [alpha, kappa] that achieved best_nll
    for sp = 1:n_startingPoints % try each random starting point
        start_params = [initAlpha(n,sp), initKappa(n,sp)]; % this restart's starting [alpha, kappa]
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options); % local optimization within [lb, ub]
        if nll < best_nll % keep this restart if it's the best so far
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]); % report progress
    end

    alphaParameter(n) = best_params(1); % store this subject's best-fit alpha
    kappaParameter(n) = best_params(2); % store this subject's best-fit kappa
    sigmaParameter(n) = sigma_n; % store this subject's fixed sigma
    nll_basicRL_subjectivePE_noIntegration(n) = best_nll; % store this subject's best NLL
end
params_basicRL_subjectivePE_noIntegration.alpha = alphaParameter; % bundle alphas into a struct
params_basicRL_subjectivePE_noIntegration.kappa = kappaParameter; % ...and kappas
params_basicRL_subjectivePE_noIntegration.sigma = sigmaParameter; % ...and sigmas
safe_saveall(fullfile(save_dir, 'params_basicRL_subjectivePE_noIntegration_fixedSigma_Perceptual_multiSP.mat'), params_basicRL_subjectivePE_noIntegration); % save fitted parameters
safe_saveall(fullfile(save_dir, 'nll_basicRL_subjectivePE_noIntegration_fixedSigma_Perceptual_multiSP.mat'), nll_basicRL_subjectivePE_noIntegration); % save per-subject NLLs
end

%% --- RL + Est. Sensitivity model, subjective PE, no integration (Perceptual) ---
if run_fitting
alphaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit alpha
kappaParameter = NaN(numSubjs, 1); % will hold each subject's best-fit kappa
sigmaParameter = NaN(numSubjs, 1); % will hold each subject's fixed sigma
nll_RLsigma_subjectivePE_noIntegration = NaN(numSubjs, 1); % will hold each subject's best NLL
lb = [0, 1]; % lower bounds for [alpha, kappa]
ub = [1, 100]; % upper bounds for [alpha, kappa]
initAlpha = unifrnd(lb(1), ub(1), [numSubjs, n_startingPoints]); % random alpha starting points
initKappa = unifrnd(lb(2), ub(2), [numSubjs, n_startingPoints]); % random kappa starting points
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'RL + Est. Sensitivity subjective PE, no integration (Perceptual, fixed sigma)'); % open/relabel the waitbar
progressQueue = parallel.pool.DataQueue; % fresh progress channel for this block
afterEach(progressQueue, @(data) fitSlider_ALLmodels.progress_bar('update', data, numSubjs, n_startingPoints, 'RL + Est. Sensitivity subjective PE, no integration (Perceptual, fixed sigma)'));
parfor n = 1:numSubjs % fit every subject independently and in parallel
    subj = preprocess_fitSlider(dataPerceptual, uniqueID(n)); % this subject's trial-level data, Perceptual condition
    rewards = fitSlider_ALLmodels.recode_rewards(subj.recoded_rewards, subj.contrast); % recode rewards into the model's fixed frame
    sigma_n = fixedSigma(n); % this subject's fixed sigma
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_subjectivePE( ... % point-observation subjective-PE version of RLsigma's objective
        [params(1), params(2), sigma_n], subj.mu_hat, subj.blocks, rewards, subj.condiff); % params(1)=alpha, params(2)=kappa, sigma fixed in
    options = optimset('Display', 'off'); % keep fmincon quiet

    best_nll = Inf; % best NLL seen so far across restarts
    best_params = NaN(1,2); % [alpha, kappa] that achieved best_nll
    for sp = 1:n_startingPoints % try each random starting point
        start_params = [initAlpha(n,sp), initKappa(n,sp)]; % this restart's starting [alpha, kappa]
        [params, nll] = fmincon(nll_fun, start_params, [], [], [], [], lb, ub, [], options); % local optimization within [lb, ub]
        if nll < best_nll % keep this restart if it's the best so far
            best_nll = nll;
            best_params = params;
        end
        send(progressQueue, [n, sp]); % report progress
    end

    alphaParameter(n) = best_params(1); % store this subject's best-fit alpha
    kappaParameter(n) = best_params(2); % store this subject's best-fit kappa
    sigmaParameter(n) = sigma_n; % store this subject's fixed sigma
    nll_RLsigma_subjectivePE_noIntegration(n) = best_nll; % store this subject's best NLL
end
params_RLsigma_subjectivePE_noIntegration.alpha = alphaParameter; % bundle alphas into a struct
params_RLsigma_subjectivePE_noIntegration.sigma = sigmaParameter; % ...and sigmas
params_RLsigma_subjectivePE_noIntegration.kappa = kappaParameter; % ...and kappas
safe_saveall(fullfile(save_dir, 'params_RLsigma_subjectivePE_noIntegration_fixedSigma_Perceptual_multiSP.mat'), params_RLsigma_subjectivePE_noIntegration); % save fitted parameters
safe_saveall(fullfile(save_dir, 'nll_RLsigma_subjectivePE_noIntegration_fixedSigma_Perceptual_multiSP.mat'), nll_RLsigma_subjectivePE_noIntegration); % save per-subject NLLs
end

fitSlider_ALLmodels.progress_bar('close'); % done fitting -- close the waitbar window (safe no-op if run_fitting was false or nothing above ran)

%% ----------------- Model comparison ---------------------------------------
%% --- Both condition ---
nll_basicRL_subjectivePE_noIntegration = importdata(fullfile(save_dir, "nll_basicRL_subjectivePE_noIntegration_fixedSigma_Both_multiSP.mat"));
nll_RLsigma_subjectivePE_noIntegration = importdata(fullfile(save_dir, "nll_RLsigma_subjectivePE_noIntegration_fixedSigma_Both_multiSP.mat"));
params_basicRL_subjectivePE_noIntegration = importdata(fullfile(save_dir, "params_basicRL_subjectivePE_noIntegration_fixedSigma_Both_multiSP.mat"));
params_RLsigma_subjectivePE_noIntegration = importdata(fullfile(save_dir, "params_RLsigma_subjectivePE_noIntegration_fixedSigma_Both_multiSP.mat"));

disp('Computing AIC and BIC for subjective PE, no-integration models...');
num_trials_both = NaN(numSubjs, 1);
for n = 1:numSubjs
    num_trials_both(n) = length(preprocess_fitSlider(dataBoth, uniqueID(n)).mu_hat);
end
num_params_subjectivePE_noIntegration = [2, 2];
model_names_subjectivePE_noIntegration = {'basicRL_subjectivePE_noIntegration','RLsigma_subjectivePE_noIntegration'};
[AICBIC_subjectivePE_noIntegration_table, bestModel_subjectivePE_noIntegration_table, BMS_subjectivePE_noIntegration_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_subjectivePE_noIntegration, nll_RLsigma_subjectivePE_noIntegration], model_names_subjectivePE_noIntegration, num_params_subjectivePE_noIntegration, num_trials_both, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_subjectivePE_noIntegration_fixedSigma_Both_reducedMS_multiSP.mat'), AICBIC_subjectivePE_noIntegration_table);
safe_saveall(fullfile(save_dir, 'bestModel_subjectivePE_noIntegration_fixedSigma_Both_multiSP.mat'), bestModel_subjectivePE_noIntegration_table);
disp('Subjective PE, no-integration model AIC/BIC estimated and saved.');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_subjectivePE_noIntegration_table, bestModel_subjectivePE_noIntegration_table, BMS_subjectivePE_noIntegration_table, model_names_subjectivePE_noIntegration, 'Both condition, Subjective PE, No Integration, Sigma Fixed');

bic_ax_both_subj_noInt = fitSlider_ALLmodels.plot_fitted_params({params_basicRL_subjectivePE_noIntegration, params_RLsigma_subjectivePE_noIntegration}, model_names_subjectivePE_noIntegration, AICBIC_subjectivePE_noIntegration_table, 'Both condition, Subjective PE, No Integration, Sigma Fixed', 'sigma (fixed, from perceptual-choice fit)');

%% --- Perceptual condition ---
nll_basicRL_subjectivePE_noIntegration = importdata(fullfile(save_dir, "nll_basicRL_subjectivePE_noIntegration_fixedSigma_Perceptual_multiSP.mat"));
nll_RLsigma_subjectivePE_noIntegration = importdata(fullfile(save_dir, "nll_RLsigma_subjectivePE_noIntegration_fixedSigma_Perceptual_multiSP.mat"));
params_basicRL_subjectivePE_noIntegration = importdata(fullfile(save_dir, "params_basicRL_subjectivePE_noIntegration_fixedSigma_Perceptual_multiSP.mat"));
params_RLsigma_subjectivePE_noIntegration = importdata(fullfile(save_dir, "params_RLsigma_subjectivePE_noIntegration_fixedSigma_Perceptual_multiSP.mat"));

disp('Computing AIC and BIC for subjective PE, no-integration models (Perceptual)...');
num_trials_perceptual = NaN(numSubjs, 1);
for n = 1:numSubjs
    num_trials_perceptual(n) = length(preprocess_fitSlider(dataPerceptual, uniqueID(n)).mu_hat);
end
num_params_subjectivePE_noIntegration = [2, 2];
model_names_subjectivePE_noIntegration = {'basicRL_subjectivePE_noIntegration','RLsigma_subjectivePE_noIntegration'};
[AICBIC_subjectivePE_noIntegration_table, bestModel_subjectivePE_noIntegration_table, BMS_subjectivePE_noIntegration_table] = fitSlider_ALLmodels.compute_model_comparison( ...
    [nll_basicRL_subjectivePE_noIntegration, nll_RLsigma_subjectivePE_noIntegration], model_names_subjectivePE_noIntegration, num_params_subjectivePE_noIntegration, num_trials_perceptual, uniqueID);
safe_saveall(fullfile(save_dir, 'AICBIC_subjectivePE_noIntegration_fixedSigma_Perceptual_reducedMS_multiSP.mat'), AICBIC_subjectivePE_noIntegration_table);
safe_saveall(fullfile(save_dir, 'bestModel_subjectivePE_noIntegration_fixedSigma_Perceptual_multiSP.mat'), bestModel_subjectivePE_noIntegration_table);
disp('Subjective PE, no-integration model AIC/BIC estimated and saved (Perceptual).');

fitSlider_ALLmodels.plot_model_comparison(AICBIC_subjectivePE_noIntegration_table, bestModel_subjectivePE_noIntegration_table, BMS_subjectivePE_noIntegration_table, model_names_subjectivePE_noIntegration, 'Perceptual condition, Subjective PE, No Integration, Sigma Fixed');

bic_ax_perc_subj_noInt = fitSlider_ALLmodels.plot_fitted_params({params_basicRL_subjectivePE_noIntegration, params_RLsigma_subjectivePE_noIntegration}, model_names_subjectivePE_noIntegration, AICBIC_subjectivePE_noIntegration_table, 'Perceptual condition, Subjective PE, No Integration, Sigma Fixed', 'sigma (fixed, from perceptual-choice fit)');

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
%  other params_* reload in this script: these variable names get
%  overwritten by the other condition's fitting blocks elsewhere in the
%  script.
%  ========================================================================
conditions_pe_compare = {'Both', 'Perceptual'};
model_names_pe_compare = {'basicRL', 'RLsigma'};
param_names_pe_compare = {'alpha', 'kappa'};

for c = 1:length(conditions_pe_compare)
    cond_label = conditions_pe_compare{c};

    params_obj = { ...
        importdata(fullfile(save_dir, sprintf('params_basicRL_fixedSigma_%s_multiSP.mat', cond_label))), ...
        importdata(fullfile(save_dir, sprintf('params_RLsigma_fixedSigma_%s_multiSP.mat', cond_label)))};
    params_subj = { ...
        importdata(fullfile(save_dir, sprintf('params_basicRL_subjectivePEv2_fixedSigma_%s_multiSP.mat', cond_label))), ...
        importdata(fullfile(save_dir, sprintf('params_RLsigma_subjectivePEv2_fixedSigma_%s_multiSP.mat', cond_label)))};

    figure('Position', [100, 100, 1200, 700]);
    sgtitle(sprintf('Objective vs Subjective PE: Fitted Parameters (%s condition, Sigma Fixed)', cond_label));

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
        importdata(fullfile(save_dir, sprintf('params_basicRL_fixedSigma_%s_multiSP.mat', cond_label))), ...
        importdata(fullfile(save_dir, sprintf('params_RLsigma_fixedSigma_%s_multiSP.mat', cond_label)))};
    params_noInt = { ...
        importdata(fullfile(save_dir, sprintf('params_basicRL_noIntegration_fixedSigma_%s_multiSP.mat', cond_label))), ...
        importdata(fullfile(save_dir, sprintf('params_RLsigma_noIntegration_fixedSigma_%s_multiSP.mat', cond_label)))};

    figure('Position', [100, 100, 1200, 700]);
    sgtitle(sprintf('Integrated vs No-Integration: Fitted Parameters, Objective PE (%s condition, Sigma Fixed)', cond_label));

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
        importdata(fullfile(save_dir, sprintf('params_basicRL_subjectivePEv2_fixedSigma_%s_multiSP.mat', cond_label))), ...
        importdata(fullfile(save_dir, sprintf('params_RLsigma_subjectivePEv2_fixedSigma_%s_multiSP.mat', cond_label)))};
    params_noInt_subj = { ...
        importdata(fullfile(save_dir, sprintf('params_basicRL_subjectivePE_noIntegration_fixedSigma_%s_multiSP.mat', cond_label))), ...
        importdata(fullfile(save_dir, sprintf('params_RLsigma_subjectivePE_noIntegration_fixedSigma_%s_multiSP.mat', cond_label)))};

    figure('Position', [100, 100, 1200, 700]);
    sgtitle(sprintf('Integrated vs No-Integration: Fitted Parameters, Subjective PE (%s condition, Sigma Fixed)', cond_label));

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
