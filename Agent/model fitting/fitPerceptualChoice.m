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

% Load the full preprocessed dataset (all subjects, all conditions)
data = importdata("preprocessed_dataFitting.mat");
uniqueID = unique(data.ID);        % List of unique subject IDs
numSubjs = length(uniqueID);       % Number of subjects to fit

% Relative contrast difference between left and right stimulus (the
% perceptual evidence available to the agent on each trial)
data.condiff_relative = (data.contrast_left - data.contrast_right) ./ 2;

% Keep only perceptual-condition trials (condition == 2); drop everything else
data(data.condition ~= 2,:) = [];

% Drop the first 5 trials of each block (warm-up trials)
data(data.trials <=5,:) = [];

% Recode choice so it is always relative to the same reference stimulus:
% when contrast == 1 (the "flipped" contrast condition), invert the
% recorded choice (0 <-> 1) so choices are comparable across contrast conditions
for h = 1:height(data)
    if data.contrast(h) == 1
        data.choice(h) = 1-data.choice(h);
    end
end

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

for n = 1:numSubjs
    % Extract and preprocess this subject's trials (pupil flag unused here, set to 0)
    subj = preprocess_fitSlider(data, uniqueID(n), 0);

    % Objective function: NLL of this subject's perceptual choices given sigma
    nll_fun = @(params) nll_perceptualChoice(params, subj.dataTable, ...
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
    end

    sigmaParameter(n) = best_sigma;   % Store this subject's best-fitting sigma
    nll_bayesianAgent(n) = best_nll;  % Store NLL at the best fit

    fprintf('Subject number: %d\n', n);

    % 3. Plot likelihood landscape for one subject
    % sigma_range = linspace(0.01, 0.5, 50);
    % nll_values = arrayfun(@(s) nll_perceptualChoice(s, subj.dataTable, ...
    %     length(unique(subj.blocks)), 20, unique(subj.blocks)), sigma_range);
    % hold on; plot(sigma_range, nll_values);
    % xlabel('Sigma'); ylabel('Negative Log-Likelihood');
end

% Package fitted sigmas into a struct (named to match the convention used
% elsewhere, e.g. plotSigma.m, even though this isn't a full learning-model fit)
params_bayesianAgent.sigma = sigmaParameter;

% Saving is currently disabled -- uncomment to persist results to disk
% safe_saveall('sigma_perceptualChoice.mat', params_bayesianAgent);
% safe_saveall('nll_perceptualChoice.mat', nll_bayesianAgent);

%% Plot fitted sigma parameter
% Mean and SEM of the fitted sigma across subjects
mean_sigma = nanmean(sigmaParameter);
SEM_sigma = nanstd(sigmaParameter) ./ sqrt(sum(~isnan(sigmaParameter)));

figure('Position', [100, 100, 200, 200]);
% Bar plot with per-subject scatter dots, mean marker, and SEM error bar
bar_plots_pval(sigmaParameter, mean_sigma, SEM_sigma, numSubjs, 1, 1, ...
    {'Perceptual choice model'}, 1, {'sigma'}, 'Fitted perceptual sigma', ...
    '', 'sigma', 0, 1, 20, 1, 12, 1, 'Arial', 1, lines(1));

%%

% Negative log-likelihood of a subject's perceptual choices given sigma.
%   params  - [sigma], perceptual sensitivity (observation noise) parameter
%   data    - subject's trial table (must include blocks, choice, condiff_relative)
%   nBlocks - number of blocks
%   nTrials - number of trials per block
%   blocks  - block index for each trial
function nll = nll_perceptualChoice(params, data, nBlocks, nTrials, blocks)
sigma = params(1);
nll_trial = NaN(nTrials,nBlocks);   % Per-trial log-likelihood, laid out [trial x block]
uniqueBlocks = unique(blocks);

for bl = 1:nBlocks
    % Fresh Bayesian agent for each block, evaluated at the candidate sigma
    agent = Agent();
    agent.task_agent_analysis = 1;   % Restrict agent to perceptual-choice mode
    agent.confirmation_bias = 0;     % No confirmation bias in this model
    agent.sigma = sigma;

    % This block's trials
    dataBlocks = data(data.blocks == uniqueBlocks(bl),:);
    choices = dataBlocks.choice;
    condiff = dataBlocks.condiff_relative;

   for t = 1:height(dataBlocks)

        % Bayesian agent inference steps
        agent.o_t = condiff(t);      % Set this trial's perceptual observation
        agent.p_s_giv_o(agent.o_t);  % Compute posterior over states given the observation
        agent.decide_p();            % Compute the agent's perceptual choice probabilities

        % Log-likelihood for this trial
        % nll_trial(t,bl) = log(agent.p_d_t(choices(t) + 1));

        % Probability the agent assigns to the choice actually made,
        % clipped away from 0/1 to avoid -Inf from log()
        p = max(min(agent.p_d_t(choices(t) + 1), 1 - 1e-10), 1e-10);
        nll_trial(t,bl) = log(p);
    end
end
% Sum log-likelihoods across all trials/blocks and negate -> total NLL
nll = -nansum(nll_trial,"all");
end
