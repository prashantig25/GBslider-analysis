clc
clearvars

% Number of parameters
n_parameters = 15;
% Number of starting points
n_startingPoints = 1;

% FIXED parameters (not estimated)
% fixed_beta = unifrnd(0.01, 2, [n_parameters, 1]); % Now beta is fixed

% PARAMETERS TO ESTIMATE: alpha, sigma, kappa
sigmaRange = unifrnd(0, 0.05, [n_parameters, 1]);
kappaRange = unifrnd(1, 15, [n_parameters, 1]);

n_simulations = 1;
% Number of trials in each simulation
n_trials = 1000;

% Initial parameter guesses for ALPHA, KAPPA, SIGMA ONLY
initKappa = unifrnd(1, 15, [n_parameters, n_startingPoints]);
initSigma = unifrnd(0, 0.05, [n_parameters, n_startingPoints]);

% Bounds for ALPHA, KAPPA, SIGMA
lb = [1, 0];     % Lower bounds for [alpha, kappa, sigma]
ub = [15, 0.05];   % Upper bounds for [alpha, kappa, sigma]

% Arrays to store best recovered parameters
best_recovered_kappas = NaN(n_parameters, 1);
best_recovered_sigmas = NaN(n_parameters, 1);
best_nlls = Inf(n_parameters, 1); % Start with Inf to track the minimum

% create synthetic data
state = randi([0 1], n_trials, 1);
condiff = NaN(n_trials, 1);

% Generate condiff values depending on state
condiff(state == 0) = -0.08 + (0 - (-0.08)) .* rand(sum(state == 0), 1); % uniform in [-0.08, 0]
condiff(state == 1) = 0 + (0.08 - 0) .* rand(sum(state == 1), 1);        % uniform in [0, 0.08]

block_size = 25;     % trials per block
nBlocks = n_trials / block_size;

% Create block labels
blocks = repelem(1:nBlocks, block_size)';

parfor p = 1:n_parameters

    % Generate predicted data using current parameter ranges + FIXED beta
    [predicted_updates, mu_hatAll, rewards, choices] = predict_allModels.predict_bayesianAgent(...
        [kappaRange(p), sigmaRange(p)], ...
        blocks, condiff, 1, 'sample', state);
    dataTable = table();
    dataTable.state = state; 
    dataTable.condiff_relative = condiff;
    dataTable.blocks = blocks;
    dataTable.rewards = rewards;
    dataTable.choice = choices;
    
    for sp = 1:n_startingPoints

        % Optimization options
        options = optimoptions('fmincon', ...
            'Display', 'off', ...
            'Algorithm', 'interior-point', ...
            'FiniteDifferenceType', 'central', ...
            'MaxIterations', 1000, ...
            'FunctionTolerance', 1e-8);
        
        % Define objective function that estimates alpha, kappa, sigma
        % The function receives [alpha, kappa, sigma] and uses fixed beta
        nll_fun = @(params) fitSlider_ALLmodels.nll_bayesianAgent(...
            [params(1), params(2)], ...
            mu_hatAll, dataTable, ...
        length(unique(blocks)), 25, unique(blocks), rewards);
        initial_guess = [initKappa(p, sp), initSigma(p, sp)];

        % Optimize alpha, kappa, sigma parameters
        [recovered_params, nll] = fmincon(nll_fun, initial_guess, ...
            [], [], [], [], lb, ub, [], options);
        
        % Check if this is the best solution so far
        if nll < best_nlls(p)
            best_nlls(p) = nll;
            best_recovered_kappas(p) = recovered_params(1);
            best_recovered_sigmas(p) = recovered_params(2);
        end
    end
    fprintf('Currently processing: Parameter %d/%d, Starting Point %d/%d\n', ...
        p, n_parameters, sp, n_startingPoints);
end

%% Plotting results
color = copper(5);

% Kappa recovery plot
figure
scatter(kappaRange, best_recovered_kappas, 'filled', 'o', 'MarkerFaceColor', ...
    color(3,:), 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.3,'SizeData',50)
lsline
xlabel('Actual Kappa Parameter')
ylabel('Best Recovered Kappa Parameter')
title('Kappa Recovery')
axis equal
grid on

% Sigma recovery plotbubuqqqqqq
figure
scatter(sigmaRange, best_recovered_sigmas, 'filled', 'o', 'MarkerFaceColor', ...
    color(3,:), 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.3,'SizeData',50)
lsline
xlabel('Actual Sigma Parameter')
ylabel('Best Recovered Sigma Parameter')
title('Sigma Recovery')
axis equal
grid on