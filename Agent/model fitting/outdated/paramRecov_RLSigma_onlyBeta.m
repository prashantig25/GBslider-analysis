clc
clearvars

% Number of parameters
n_parameters = 60;
% Number of starting points
n_startingPoints = 15;

% FIXED parameters (not estimated)
fixed_alpha = unifrnd(0, 0.3, [n_parameters, 1]); % repelem(0.2, n_parameters);
fixed_sigma = unifrnd(0.05, 0.1, [n_parameters, 1]);  %repelem(0.07, n_parameters).';  
fixed_kappa = unifrnd(0, 25, [n_parameters, 1]); % repelem(15, n_parameters).';

% PARAMETER TO ESTIMATE: beta
betaRange = unifrnd(0.01, 2, [n_parameters, 1]); %repelem(1, n_parameters).';

n_simulations = 1;
% Number of trials in each simulation
n_trials = 200;

% Initial parameter guesses for BETA ONLY
initBeta = unifrnd(0.01, 2, [n_parameters, n_startingPoints]);
initAlpha = unifrnd(0, 0.3, [n_parameters, n_startingPoints]);
initKappa = unifrnd(0, 20, [n_parameters, n_startingPoints]);
initSigma = unifrnd(0.05, 0.1, [n_parameters, n_startingPoints]);

% Bounds for BETA ONLY
lb = [0,0,0.05,0];     % Lower bound for beta
ub = [0.3,20,0.1,2];     % Upper bound for beta

% Arrays to store best recovered parameters
best_recovered_betas = NaN(n_parameters, 1);
best_recovered_alphas = NaN(n_parameters, 1);
best_recovered_kappa = NaN(n_parameters, 1);
best_nlls = Inf(n_parameters, 1); % Start with Inf to track the minimum

parfor p = 1:n_parameters
    for sp = 1:n_startingPoints
        % Generate simulated data
        rewards = [rand(1, n_trials./2) < 0.7; rand(1, n_trials./2) < 0.9];
        state = rand(1, n_trials) < 0.5;
        blocks_all = [];
        for nb = 1:n_trials./25
            blocks_all = [repelem(nb, 25), blocks_all];
        end
        condiff = state.';
        for s = 1:length(state)
            if state(s) == 0
                condiff(s) = unifrnd(-0.08, 0, 1);
            else
                condiff(s) = unifrnd(0, 0.08, 1);
            end
        end
        
        % Generate predicted data using FIXED parameters + current beta range
        [predictedUp, mu_hat] = predict_allModels.predict_RLsigma(...
            [fixed_alpha(p), fixed_kappa(p), fixed_sigma(p), betaRange(p)], ...
            blocks_all, rewards, state, condiff, 'mean');
        
        % Optimization options
        options = optimoptions('fmincon', ...
            'Display', 'off', ...
            'Algorithm', 'interior-point', ...
            'FiniteDifferenceType', 'central', ...
            'MaxIterations', 1000, ...
            'FunctionTolerance', 1e-6);
        
        % Define objective function that only estimates beta
        % The function receives only beta, but passes fixed values for other params
        nll_fun = @(params) fitSlider_ALLmodels.nll_RLsigma_VOI(...
            [params(1), params(2), params(3), params(4)], ...
            mu_hat, blocks_all, rewards, condiff);
        initial_guess = [initAlpha(p, sp), initKappa(p, sp), initSigma(p, sp), initBeta(p, sp)];

        % Optimize only beta parameter
        [recovered_beta, nll] = fmincon(nll_fun, initial_guess, ...
            [], [], [], [], lb, ub, [], options);
        
        % Check if this is the best solution so far
        if nll < best_nlls(p)
            best_nlls(p) = nll;
            best_recovered_betas(p) = recovered_beta(4);
            best_recovered_alphas(p) = recovered_beta(1);
            best_recovered_kappa(p) = recovered_beta(2);
            best_recovered_sigma(p) = recovered_beta(3);
        end
    end
    fprintf('Currently processing: Parameter %d/%d, Starting Point %d/%d\n', ...
        p, n_parameters, sp, n_startingPoints);
end

% Display results
fprintf('\n=== RESULTS ===\n');
fprintf('Fixed Parameters:\n');
fprintf('Alpha: %.3f\n', fixed_alpha(1));
fprintf('Kappa: %.3f\n', fixed_kappa(1));
fprintf('Sigma: %.3f\n', fixed_sigma(1));
fprintf('\nEstimated Beta Parameters:\n');
for p = 1:n_parameters
    fprintf('Parameter %d - True Beta: %.3f, Recovered Beta: %.3f, NLL: %.3f\n', ...
        p, betaRange(p), best_recovered_betas(p), best_nlls(p));
end

%%
color = copper(5);

figure
scatter(fixed_alpha,best_recovered_alphas,'filled','o','MarkerFaceColor', ...
    color(3,:),'MarkerEdgeColor','k','MarkerFaceAlpha',0.3)
lsline
xlabel('Actual parameter')
ylabel('Best recovered parameter')
title('alpha (all free - but high range sigma)')
lsline

figure
scatter(fixed_kappa,best_recovered_kappa,'filled','o','MarkerFaceColor', ...
    color(3,:),'MarkerEdgeColor','k','MarkerFaceAlpha',0.3)
lsline
xlabel('Actual parameter')
ylabel('Best recovered parameter')
title('kappa (all free - but high range sigma)')
lsline

figure
scatter(betaRange,best_recovered_betas,'filled','o','MarkerFaceColor', ...
    color(3,:),'MarkerEdgeColor','k','MarkerFaceAlpha',0.3)
lsline
xlabel('Actual parameter')
ylabel('Best recovered parameter')
title('beta (all free - but high range sigma)')
lsline

figure
scatter(fixed_sigma,best_recovered_sigma,'filled','o','MarkerFaceColor', ...
    color(3,:),'MarkerEdgeColor','k','MarkerFaceAlpha',0.3)
lsline
xlabel('Actual parameter')
ylabel('Best recovered parameter')
title('sigma (all free - but high range sigma)')
lsline
