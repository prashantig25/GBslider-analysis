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

% fmincon settings: single free parameter (sigma) with bounds
lb = 0.001;
ub = 0.1;

% Multi-start optimization: fmincon is a local optimizer, so a single
% starting point risks converging to a local rather than global minimum.
% Draw multiple random starting points per subject within [lb, ub] and
% keep whichever converges to the lowest NLL.
n_startingPoints = 15;
initSigma = unifrnd(lb, ub, [numSubjs, n_startingPoints]);

% Graphical progress bar (mirrors the progress_bar helper in
% fitReducedModelSpace.m; no DataQueue/afterEach needed here since this
% loop is serial, not parfor, so waitbar can be called directly).
progress_bar('reset', numSubjs, n_startingPoints, 'Perceptual sigma');

for n = 1:numSubjs
    % Extract and preprocess this subject's trials (pupil flag unused here, set to 0)
    subj = preprocess_fitSlider(data, uniqueID(n), 0);

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
        progress_bar('update', [n, sp], numSubjs, n_startingPoints, 'Perceptual sigma');
    end

    sigmaParameter(n) = best_sigma;   % Store this subject's best-fitting sigma
    nll_bayesianAgent(n) = best_nll;  % Store NLL at the best fit

    % 3. Plot likelihood landscape for one subject
    % sigma_range = linspace(0.01, 0.5, 50);
    % nll_values = arrayfun(@(s) fitSlider_ALLmodels.nll_perceptualChoice(s, subj.dataTable, ...
    %     length(unique(subj.blocks)), 20, unique(subj.blocks)), sigma_range);
    % hold on; plot(sigma_range, nll_values);
    % xlabel('Sigma'); ylabel('Negative Log-Likelihood');
end
progress_bar('close');

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

%%

% Graphical progress bar, shared pattern with fitReducedModelSpace.m.
%   'reset'  - (numSubjs, n_startingPoints, label): open/reset the bar
%   'update' - ([n, sp], numSubjs, n_startingPoints, label): advance it
%   'close'  - (): close the bar window
function progress_bar(mode, varargin)
persistent h count total
switch mode
    case 'reset'
        [numSubjs, n_startingPoints, label] = varargin{:};
        count = 0;
        total = numSubjs * n_startingPoints;
        if isempty(h) || ~isvalid(h)
            h = waitbar(0, '', 'Name', 'Model fitting progress');
        end
        waitbar(0, h, sprintf('%s: subject 0/%d, start 0/%d', label, numSubjs, n_startingPoints));
    case 'update'
        [data, numSubjs, n_startingPoints, label] = varargin{:};
        count = count + 1;
        n = data(1); sp = data(2);
        waitbar(min(count / total, 1), h, ...
            sprintf('%s: subject %d/%d, start %d/%d', label, n, numSubjs, sp, n_startingPoints));
    case 'close'
        if ~isempty(h) && isvalid(h)
            close(h);
        end
end
end
