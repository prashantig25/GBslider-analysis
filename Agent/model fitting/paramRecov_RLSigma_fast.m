clc
clearvars
tic;

% Number of parameters
n_parameters = 100;

% Number of starting points
n_startingPoints = 30;

% set range of parameters
alphaRange = unifrnd(0,0.5,[n_parameters,1]);
betaRange = unifrnd(0,2,[n_parameters,1]);
sigmaRange = unifrnd(0,0.1,[n_parameters,1]);
kappaRange = unifrnd(1,30,[n_parameters,1]);

n_simulations = 1;

% Number of trials in each simulation
n_trials = 200;

% Pre-generate all initial parameter guesses outside the loop
initAlpha = unifrnd(0,0.3,[n_parameters,n_startingPoints]);
initKappa = unifrnd(0,20,[n_parameters,n_startingPoints]);
initBeta = unifrnd(0,2,[n_parameters,n_startingPoints]);
initSigma = unifrnd(0,0.1,[n_parameters,n_startingPoints]);

lb = [0, 1, 0, 0]; % Lower bounds
ub = [0.5, 100, 0.1, 2]; % Upper bounds

% Optimized optimizer options
options = optimoptions('fmincon', ...
    'Display', 'off', ...
    'Algorithm', 'interior-point', ...  % Generally faster than active-set
    'FiniteDifferenceType', 'forward', ... % Faster than central
    'MaxIterations', 500, ...  % Reduced from 1000
    'FunctionTolerance', 1e-4, ... % Relaxed from 1e-6
    'OptimalityTolerance', 1e-4, ...
    'StepTolerance', 1e-8, ...
    'UseParallel', false); % Set to true if you have Parallel Computing Toolbox

% Pre-generate shared data that doesn't change across parameters
% Generate block structure once
blocks_all = [];
for nb = 1:n_trials./25
    blocks_all = [repelem(nb,25), blocks_all];
end

% Arrays to store best recovered parameters
best_recovered_alphas = NaN(n_parameters,1);
best_recovered_kappas = NaN(n_parameters,1);
best_recovered_betas = NaN(n_parameters,1);
best_recovered_sigma = NaN(n_parameters,1);
best_nlls = Inf(n_parameters,1);

% Pre-allocate cell arrays for parallel processing
param_sets = cell(n_parameters, 1);
reward_sets = cell(n_parameters, 1);
state_sets = cell(n_parameters, 1);
condiff_sets = cell(n_parameters, 1);
mu_hat_sets = cell(n_parameters, 1);

% Pre-generate all data and predictions outside the parallel loop
fprintf('Pre-generating data and predictions...\n');
for p = 1:n_parameters
    % Generate simulated data once per parameter set
    rewards = [rand(1,n_trials./2) < 0.7; rand(1,n_trials./2) < 0.9]; 
    state = rand(1,n_trials) < 0.5;
    
    condiff = state.';
    for s = 1:length(state)
        if state(s) == 0
            condiff(s) = unifrnd(-0.08,0,1);
        else
            condiff(s) = unifrnd(0,0.08,1);
        end
    end
    
    % Store the generated data
    param_sets{p} = [alphaRange(p), kappaRange(p), sigmaRange(p), betaRange(p)];
    reward_sets{p} = rewards;
    state_sets{p} = state;
    condiff_sets{p} = condiff;
    
    % Pre-compute predictions
    [~, mu_hat] = predict_allModels.predict_RLsigma(...
        param_sets{p}, blocks_all, rewards, state, condiff, 'mean');
    mu_hat_sets{p} = mu_hat;
end

fprintf('Starting parallel parameter recovery...\n');

% Main parallel loop
parfor p = 1:n_parameters
    
    % Get pre-computed data for this parameter set
    mu_hat = mu_hat_sets{p};
    rewards = reward_sets{p};
    condiff = condiff_sets{p};
    
    % Create function handle once per parameter
    nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI(params, mu_hat, blocks_all, rewards, condiff);
    
    best_nll_local = Inf;
    best_params_local = NaN(1,4);
    
    % Try multiple starting points for this parameter
    for sp = 1:n_startingPoints
        
        initial_guess = [initAlpha(p,sp), initKappa(p,sp), initSigma(p,sp), initBeta(p,sp)];
        
        try
            [recovered_params, nll] = fmincon(nll_fun, initial_guess, [], [], [], [], lb, ub, [], options);
            
            % Check if this is the best solution so far for this parameter
            if nll < best_nll_local
                best_nll_local = nll;
                best_params_local = recovered_params;
            end
            
        catch ME
            % Continue if optimization fails for this starting point
            warning('Optimization failed for parameter %d, starting point %d: %s', p, sp, ME.message);
            continue;
        end
    end
    
    % Store the best results for this parameter
    best_nlls(p) = best_nll_local;
    best_recovered_alphas(p) = best_params_local(1);
    best_recovered_kappas(p) = best_params_local(2);
    best_recovered_sigma(p) = best_params_local(3);
    best_recovered_betas(p) = best_params_local(4);
    
    % Progress reporting (note: may appear out of order due to parallel execution)
    if mod(p, 10) == 0
        fprintf('Completed parameter %d/%d\n', p, n_parameters);
    end
end

fprintf('Parameter recovery complete!\n');
time = toc;
disp(toc)

%% Plotting results
figure('Position', [100, 100, 800, 600]);

subplot(2,2,1)
hold on
scatter(alphaRange, best_recovered_alphas, 50, 'filled')
lsline
xlabel('Actual parameter')
ylabel('Best recovered parameter')
title('Alpha Recovery')
grid on

subplot(2,2,2)
hold on
scatter(kappaRange, best_recovered_kappas, 50, 'filled')
lsline
xlabel('Actual parameter')
ylabel('Best recovered parameter')
title('Kappa Recovery')
grid on

subplot(2,2,3)
hold on
scatter(betaRange, best_recovered_betas, 50, 'filled')
lsline
xlabel('Actual parameter')
ylabel('Best recovered parameter')
title('Beta Recovery')
grid on

subplot(2,2,4)
hold on
scatter(sigmaRange, best_recovered_sigma, 50, 'filled')
lsline
xlabel('Actual parameter')
ylabel('Best recovered parameter')
title('Sigma Recovery')
grid on

sgtitle('Parameter Recovery Results', 'FontSize', 14, 'FontWeight', 'bold');

% Calculate and display recovery statistics
fprintf('\nRecovery Statistics:\n');
fprintf('Alpha - R²: %.3f, RMSE: %.4f\n', corr(alphaRange, best_recovered_alphas)^2, sqrt(mean((alphaRange - best_recovered_alphas).^2)));
fprintf('Kappa - R²: %.3f, RMSE: %.4f\n', corr(kappaRange, best_recovered_kappas)^2, sqrt(mean((kappaRange - best_recovered_kappas).^2)));
fprintf('Beta - R²: %.3f, RMSE: %.4f\n', corr(betaRange, best_recovered_betas)^2, sqrt(mean((betaRange - best_recovered_betas).^2)));
fprintf('Sigma - R²: %.3f, RMSE: %.4f\n', corr(sigmaRange, best_recovered_sigma)^2, sqrt(mean((sigmaRange - best_recovered_sigma).^2)));