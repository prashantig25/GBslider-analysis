clc
clearvars

rng(123)

% Number of parameters
n_parameters = 30;
% Number of starting points
n_startingPoints = 10;

% FIXED parameters (not estimated)
% fixed_beta = unifrnd(0.01, 2, [n_parameters, 1]); % Now beta is fixed

% PARAMETERS TO ESTIMATE: alpha, sigma, kappa
alphaRange = unifrnd(0, 0.3, [n_parameters, 1]);
sigmaRange = unifrnd(0, 0.05, [n_parameters, 1]);
kappaRange = unifrnd(0, 25, [n_parameters, 1]);

n_simulations = 1;
% Number of trials in each simulation
n_trials = 1000;

% Initial parameter guesses for ALPHA, KAPPA, SIGMA ONLY
initAlpha = unifrnd(0, 0.3, [n_parameters, n_startingPoints]);
initKappa = unifrnd(0, 25, [n_parameters, n_startingPoints]);
initSigma = unifrnd(0, 0.05, [n_parameters, n_startingPoints]);

% Bounds for ALPHA, KAPPA, SIGMA
lb = [0, 0, 0];     % Lower bounds for [alpha, kappa, sigma]
ub = [0.3, 25, 0.05];   % Upper bounds for [alpha, kappa, sigma]

% Arrays to store best recovered parameters
best_recovered_alphas = NaN(n_parameters, 1);
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

% Test sensitivity
sigmas = [0.001, 0.002, 0.01, 0.05];
alphas = [0.1,0.3,0.5,0.7];
kappas = [5,10,20,40];
for s = 1:length(sigmas)
    rng(123)
    params_test = [alphas(s), kappas(s), 0.03];
    [~, mu_test(:,s)] = predict_allModels.predict_RLsigma_updated(params_test, blocks, state, condiff, 'mean');
end
corr(mu_test)  % High correlations = poor identifiability

parfor p = 1:n_parameters

    % Generate predicted data using current parameter ranges + FIXED beta
    [predictedUp, mu_hat, rewards, choice] = predict_allModels.predict_RLsigma_updated(...
        [alphaRange(p), kappaRange(p), sigmaRange(p)], ...
        blocks, state, condiff, 'sample');

    for sp = 1:n_startingPoints
        
        % Optimization options
        options = optimoptions('fmincon', ...
            'Display', 'off', ...
            'Algorithm', 'interior-point', ...
            'FiniteDifferenceType', 'central', ...
            'MaxIterations', 1000, ...
            'FunctionTolerance', 1e-6);
        
        % Define objective function that estimates alpha, kappa, sigma
        % The function receives [alpha, kappa, sigma] and uses fixed beta
        nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI(...
            [params(1), params(2), params(3)], ...
            mu_hat, blocks, rewards, condiff);
        initial_guess = [initAlpha(p, sp), initKappa(p, sp), initSigma(p, sp)];

        % Optimize alpha, kappa, sigma parameters
        [recovered_params, nll] = fmincon(nll_fun, initial_guess, ...
            [], [], [], [], lb, ub, [], options);
        
        % Check if this is the best solution so far
        if nll < best_nlls(p)
            best_nlls(p) = nll;
            best_recovered_alphas(p) = recovered_params(1);
            best_recovered_kappas(p) = recovered_params(2);
            best_recovered_sigmas(p) = recovered_params(3);
        end
    end
    fprintf('Currently processing: Parameter %d/%d, Starting Point %d/%d\n', ...
        p, n_parameters, sp, n_startingPoints);
end

%% Plotting results
color = copper(5);

% Alpha recovery plot
figure
scatter(alphaRange, best_recovered_alphas, 'filled', 'o', 'MarkerFaceColor', ...
    color(3,:), 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.3,'SizeData',50)
lsline
xlabel('Actual Alpha Parameter')
ylabel('Best Recovered Alpha Parameter')
title('Alpha Recovery')
axis equal
grid on

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

% Sigma recovery plot
figure
scatter(sigmaRange, best_recovered_sigmas, 'filled', 'o', 'MarkerFaceColor', ...
    color(3,:), 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.3,'SizeData',50)
lsline
xlabel('Actual Sigma Parameter')
ylabel('Best Recovered Sigma Parameter')
title('Sigma Recovery')
axis equal
grid on