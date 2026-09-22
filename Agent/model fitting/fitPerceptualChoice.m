%% ========================================================================
%  Script: Fit Perceptual Sensitivity (sigma) to Perceptual-Condition Choices
%  ------------------------------------------------------------------------
%  Fits a single sigma parameter per subject to their binary choices in
%  the perceptual condition, using a Bayesian ideal-observer agent
%  (Agent class) as the choice-generating model. Uses 15 random starting
%  points per subject and keeps the lowest-NLL fit.
%  ========================================================================
clc                 % Clear command window
clearvars           % Clear all variables from the workspace
rng(123); % for reproducability -- same seed as fitReducedModelSpace.m /
% fitReducedModelSpace_fixedSigma.m, so re-running produces the same
% random starting points (and hence the same fmincon result) each time,
% instead of safe_saveall flagging harmless numerical noise as "different"

% PATH STUFF -- anchor the save location to Agent/model fitting/ so output
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

% Load the preprocessed dataset, already restricted to perceptual-condition
% trials with warm-up trials dropped and choice recoded into a fixed
% reference frame (see fitSlider_ALLmodels.load_fitting_data()).
[~, ~, uniqueID, numSubjs, data] = fitSlider_ALLmodels.load_fitting_data();

% Preallocate per-subject outputs
sigmaParameter = NaN(numSubjs, 1);    % Best-fitting sigma for each subject
nll_bayesianAgent = NaN(numSubjs, 1); % Negative log-likelihood at that fit
matchRate = NaN(numSubjs, 1);         % Posterior predictive check: per-subject
                                       % fraction of trials where the simulated
                                       % choice matches the observed choice

% fmincon settings: single free parameter (sigma) with bounds
lb = 0.001;
ub = 0.1;

% Multi-start optimization: fmincon is a local optimizer, so a single
% starting point risks converging to a local rather than global minimum.
% Draw multiple random starting points per subject within [lb, ub] and
% keep whichever converges to the lowest NLL.
n_startingPoints = 15;
initSigma = unifrnd(lb, ub, [numSubjs, n_startingPoints]);

% Graphical progress bar (fitSlider_ALLmodels.progress_bar, shared with
% fitReducedModelSpace.m; no DataQueue/afterEach needed here since this
% loop is serial, not parfor, so waitbar can be called directly).
fitSlider_ALLmodels.progress_bar('reset', numSubjs, n_startingPoints, 'Perceptual sigma');

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
    for sp = 1:n_startingPoints
        [params, nll] = fmincon(nll_fun, initSigma(n, sp), [], [], [], [], lb, ub, [], options);
        if nll < best_nll
            best_nll = nll;
            best_sigma = params(1);
        end
        fitSlider_ALLmodels.progress_bar('update', [n, sp], numSubjs, n_startingPoints, 'Perceptual sigma');
    end

    sigmaParameter(n) = best_sigma;   % Store this subject's best-fitting sigma
    nll_bayesianAgent(n) = best_nll;  % Store NLL at the best fit

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

    % 3. Plot likelihood landscape for one subject
    % sigma_range = linspace(0.01, 0.5, 50);
    % nll_values = arrayfun(@(s) fitSlider_ALLmodels.nll_perceptualChoice(s, subj.dataTable, ...
    %     length(unique(subj.blocks)), 20, unique(subj.blocks)), sigma_range);
    % hold on; plot(sigma_range, nll_values);
    % xlabel('Sigma'); ylabel('Negative Log-Likelihood');
end
fitSlider_ALLmodels.progress_bar('close');

% Package fitted sigmas into a struct (named to match the convention used
% elsewhere, e.g. plotSigma.m, even though this isn't a full learning-model fit)
params_bayesianAgent.sigma = sigmaParameter;

% Persist results to disk (consumed by fitReducedModelSpace_fixedSigma.m,
% which fixes sigma to these fitted values instead of fitting it jointly)
safe_saveall(fullfile(save_dir, 'sigma_perceptualChoice.mat'), params_bayesianAgent);
safe_saveall(fullfile(save_dir, 'nll_perceptualChoice.mat'), nll_bayesianAgent);

%% Plot fitted sigma parameter - will remove it later. just here to see quick results
% Mean and SEM of the fitted sigma across subjects
mean_sigma = nanmean(sigmaParameter);
SEM_sigma = nanstd(sigmaParameter) ./ sqrt(sum(~isnan(sigmaParameter)));

figure('Position', [100, 100, 200, 200]);
% Bar plot with per-subject scatter dots, mean marker, and SEM error bar
bar_plots_pval(sigmaParameter, mean_sigma, SEM_sigma, numSubjs, 1, 1, ...
    {'Perceptual choice model'}, 1, {'sigma'}, 'Fitted perceptual sigma', ...
    '', 'sigma', 0, 1, 20, 1, 12, 1, 'Arial', 1, lines(1));

%% POSTERIOR PREDICTIVE CHECK -- single replicate
% Compares the psychometric curve (P(choice = 1) vs. contrast difference)
% between observed choices and one simulated replicate per subject,
% pooled across all subjects' trials. Each replicate was generated above
% using that subject's own fitted sigma and real trial sequence, so this
% checks whether the fitted model reproduces the real data's psychometric
% pattern -- a single replicate is a first pass; a proper check would draw
% many replicates per subject and compare against the resulting predictive
% distribution rather than one simulated curve.
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

%% POSTERIOR PREDICTIVE CHECK -- group-level summary
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

%% PARAMETER RECOVERY -- validate the fit above by simulating synthetic
% choice data at known sigma values, re-fitting sigma from that data, and
% comparing recovered vs. true sigma (fitSlider_ALLmodels.m owns the
% simulation/fitting/plotting logic; this just chooses the study parameters).
n_parameters = 20;     % number of synthetic subjects/sigma values to test
n_recoveryStartingPoints = 20;  % random fmincon starting points per subject
n_recoveryTrials = 1000;        % simulated trials per subject

[sigmaRange, best_recovered_sigmas, best_nlls] = fitSlider_ALLmodels.recover_perceptualChoiceSigma(n_parameters, n_recoveryStartingPoints, n_recoveryTrials);
fitSlider_ALLmodels.plot_perceptualChoiceRecovery(sigmaRange, best_recovered_sigmas);
