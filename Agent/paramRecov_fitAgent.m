
% Parameter Recovery Script
clc
clearvars

% Number of parameters
n_parameters = 50;

% Number of starting points
n_startingPoints = 10;

% Define true parameters
true_sigma = unifrnd(0,0.1,[n_parameters,1]);
true_beta = unifrnd(1,20,[n_parameters,1]); % We'll keep this fixed for simplicity

options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'interior-point');

% Initial parameter guesses
initSigma = unifrnd(0,0.1,[n_parameters,n_startingPoints]); % [alpha, beta]
initBeta = unifrnd(0,20,[n_parameters,n_startingPoints]); % [alpha, beta]
lb = [0, 0]; % Lower bounds (alpha, beta must be >= 0)
ub = [20, 0.1]; % Upper bounds (alpha <= 1, beta reasonable range)

% Arrays to store best recovered parameters
best_recovered_sigmas = NaN(n_parameters,1);
best_recovered_betas = NaN(n_parameters,1);
best_nlls = Inf(n_parameters,1); % Start with Inf to track the minimum

% Arrays to store recovered parameters
recovered_sigmas = NaN(n_startingPoints, n_parameters);
recovered_betas = NaN(n_startingPoints, n_parameters);
nllAll = NaN(n_startingPoints,n_parameters);

agent = 1; % agent type normative (1) or categorical (2)

% INITIALIZE VARS
num_sims = 1; % number of simulations
num_blocks = 40; % number of blocks per simulation
numTrials = 25; % number of trials in one block

for p = 1:n_parameters

    fprintf('\n--- Parameter set %d/%d ---\n', p, n_parameters);  % New line

    for sp = 1:n_startingPoints

    simulations_condition1 = []; % store the simulations
    simulations_condition2 = []; % store the simulation

    % RUN SIMULATIONS FOR NORMATIVE AGENT
    condition = 1; % task condition
    beta = true_beta(p);
    sigma = true_sigma(p);
    [simulations_condition1] = run_simulations(num_sims,num_blocks,simulations_condition1,condition,agent,sigma,1,beta); % use run simulations function -- task_agent_analysis=1 inserted to preserve this script's prior (hardcoded) behavior; sigma/beta occupy the rangeMu/varargin{1} slots exactly as before
    condition = 2; % task condition
    agent = 1; % agent type normative (1) or categorical (2)
    [simulations_condition2] = run_simulations(num_sims,num_blocks,simulations_condition2,condition,agent,sigma,1,beta); % use run simulations function -- task_agent_analysis=1 inserted to preserve this script's prior (hardcoded) behavior; sigma/beta occupy the rangeMu/varargin{1} slots exactly as before
    simulated_data = [simulations_condition1; simulations_condition2];
    blocks_all = [];
    for n = 1:num_blocks*2
        blocks = repelem(n,numTrials,1);
        blocks_all = [blocks_all; blocks];
    end
    simulated_data.blocks = blocks_all;
    choices = simulated_data.action;
    reward = simulated_data.reward;
    states = simulated_data.state;
    condiff = simulated_data.contrast_diff;

    % Optimize using fmincon
    %         options = optimset('Display', 'iter');
    options = optimoptions('fmincon', ...
        'Display', 'off', ...
        'Algorithm', 'active-set', ...
        'FiniteDifferenceType', 'central', ...
        'MaxIterations', 1000, ...
        'FunctionTolerance', 1e-6);
    [recovered_params,nll] = fmincon(@(params) NLL_fitSigmaBeta_Agent([params(1), params(2)], simulated_data, condiff, reward, choices), ...
        [initBeta(p,sp), initSigma(p,sp)], [], [], [], [], lb, ub, [], options);

    % Store recovered parameters
    recovered_sigmas(sp,p) = recovered_params(2);
    recovered_betas(sp,p) = recovered_params(1);
    nllAll(sp,p) = nll;
    end

    % Check if this is the best solution so far
    if nll < best_nlls(p)
        best_nlls(p) = nll;
        best_recovered_sigmas(p) = recovered_params(2);
        best_recovered_betas(p) = recovered_params(1);
    end
end

%% Plot results
figure
hold on
scatter(true_sigma, best_recovered_sigmas, "filled", "o", "MarkerEdgeColor", 'k', 'MarkerFaceColor', 'b', 'MarkerFaceAlpha', 0.4)
lsline
xlabel('True alpha')
ylabel('Best recovered sigma')
title('Best Parameter Recovery')

figure
hold on
scatter(true_beta, best_recovered_betas)
lsline
xlabel('True beta')
ylabel('Best recovered beta')
title('Best Parameter Recovery')

% Print summary statistics
fprintf('Mean True Alpha: %.2f, Mean Best Recovered Alpha: %.2f\n', mean(true_alpha), mean(best_recovered_alphas));
fprintf('Mean True Beta: %.2f, Mean Best Recovered Beta: %.2f\n', mean(true_beta), mean(best_recovered_betas));
