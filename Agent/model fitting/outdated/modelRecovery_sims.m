clc
clearvars
rng(123)

% Number of parameters
n_parameters = 30;

% FIXED parameters (not estimated)
% fixed_beta = unifrnd(0.01, 2, [n_parameters, 1]); % Now beta is fixed

% PARAMETERS TO ESTIMATE: alpha, sigma, kappa
alphaRange = unifrnd(0, 0.3, [n_parameters, 1]);
sigmaRange = unifrnd(0, 0.05, [n_parameters, 1]);
kappaRange = unifrnd(0, 25, [n_parameters, 1]);
confirmBiasRange = unifrnd(0.5,1,[n_parameters,1]);

n_simulations = 1;
% Number of trials in each simulation
n_trials = 1000;

% create synthetic data
state = randi([0 1], n_trials, 1);
condiff = NaN(n_trials, 1);

% Generate condiff values depending on state
condiff(state == 0) = -0.08 + (0 - (-0.08)) .* rand(sum(state == 0), 1); % uniform in [-0.08, 0]
condiff(state == 1) = 0 + (0.08 - 0) .* rand(sum(state == 1), 1);        % uniform in [0, 0.08]

block_size = 25; % trials per block
nBlocks = n_trials / block_size;

% Create block labels
blocks = repelem(1:nBlocks, block_size)';

dataAll = [];
for p = 1:n_parameters

    % Generate predicted data using current parameter ranges + FIXED beta
    [predicted_updates, mu_hat, rewards, choice] = predict_allModels.predict_bayesianAgent(...
        [25, 0.1], ...
        blocks, condiff, 1,'mean', state);

    % [predicted_updates, mu_hatConfirm, rewards, choice] = predict_allModels.predict_bayesianAgent_confirmBias(...
    %     [kappaRange(p), sigmaRange(p), confirmBiasRange(p)], ...
    %     blocks, condiff, 1,'sample', state);

    % Generate predicted data using current parameter ranges + FIXED beta
    % [predictedUp, mu_hat, rewards, choice] = predict_allModels.predict_RLsigma(...
    %     [alphaRange(p), kappaRange(p), sigmaRange(p)], ...
    %     blocks, state, condiff, 'sample');

    dataTable = table();
    dataTable.state = state; 
    dataTable.condiff_relative = condiff;
    dataTable.blocks = blocks;
    dataTable.rewards = rewards;
    dataTable.choice = choice;
    dataTable.ID = repelem(p,n_trials,1);
    dataTable.mu_hat = mu_hat;

    dataAll = [dataAll;dataTable];
end

safe_saveall("genModel_AgentFixedParams_mean0.1Sigma.mat",dataAll);