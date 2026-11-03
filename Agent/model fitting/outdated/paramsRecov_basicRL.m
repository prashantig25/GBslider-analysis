clc
clearvars

% Number of parameters
n_parameters = 60;

% Number of starting points
n_startingPoints = 15;

% set range of parameters
alphaRange = unifrnd(0,0.5,[n_parameters,1]);
betaRange = unifrnd(0,2,[n_parameters,1]);
kappaRange = unifrnd(1,30,[n_parameters,1]);

n_simulations = 1;

% Number of trials in each simulation
n_trials = 200;
options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'interior-point');

% Initial parameter guesses
initAlpha = unifrnd(0,0.3,[n_parameters,n_startingPoints]); % [alpha, beta]
initKappa = unifrnd(0,20,[n_parameters,n_startingPoints]); % [alpha, beta]
initBeta = unifrnd(0,2,[n_parameters,n_startingPoints]); % [alpha, beta]
lb = [0, 1, 0]; % Lower bounds (alpha, beta must be >= 0)
ub = [0.5, 100, 2]; % Upper bounds (alpha <= 1, beta reasonable range)

% Arrays to store best recovered parameters
best_recovered_alphas = NaN(n_parameters,1);
best_recovered_kappas = NaN(n_parameters,1);
best_recovered_betas = NaN(n_parameters,1);
best_recovered_sigma = NaN(n_parameters,1);
best_nlls = Inf(n_parameters,1); % Start with Inf to track the minimum

parfor p = 1:n_parameters

    for sp = 1:n_startingPoints

        % Generate simulated data
        rewards = [rand(1,n_trials./2) < 0.7;rand(1,n_trials./2) < 0.9]; 
        state = rand(1,n_trials) < 0.5;
        blocks_all = [];
        for nb = 1:n_trials./25
            blocks_all = [repelem(nb,25), blocks_all];
        end
        condiff = state.';
        for s = 1:length(state)
            if state(s) == 0
                condiff(s) = unifrnd(-0.08,0,1);
            else
                condiff(s) = unifrnd(0,0.08,1);
            end
        end
        [predictedUp, mu_hat] = predict_allModels.predict_basicRL(...
            [alphaRange(p), kappaRange(p), betaRange(p)],blocks_all,rewards,state,'sample');
        % options = optimoptions('fmincon', ...
        %     'Display', 'off', ...
        %     'Algorithm', 'active-set', ...
        %     'FiniteDifferenceType', 'central', ...
        %     'MaxIterations', 1000, ...
        %     'FunctionTolerance', 1e-6);
        % [recovered_params,nll] = fmincon(@(params) fitSlider_ALLmodels.nll_basicRL_integrated([alphaRange(p), kappaRange(p), betaRange(p)],mu_hat,blocks_all,rewards,condiff), ...
        %     [initAlpha(p,sp), initKappa(p,sp), initBeta(p,sp)], [], [], [], [], lb, ub, [], options);

        nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated(params, mu_hat,blocks_all,rewards,condiff);
        options = optimset('Display', 'off');
        [recovered_params, nll] = fmincon(nll_fun, [initAlpha(p,sp), initKappa(p,sp), initBeta(p,sp)], [], [], [], [], lb, ub, [], options);


        % Store recovered parameters
        % Check if this is the best solution so far
        if nll < best_nlls(p)
            best_nlls(p) = nll;
            best_recovered_alphas(p) = recovered_params(1);
            best_recovered_kappas(p) = recovered_params(2);
            % best_recovered_sigma(p) = recovered_params(3);
            best_recovered_betas(p) = recovered_params(3);
        end
    end

    fprintf('Currently processing: Parameter %d/%d, Starting Point %d/%d\n', ...
                p, n_parameters, sp, n_startingPoints);

end

figure
subplot(1,3,1)
hold on
scatter(alphaRange,best_recovered_alphas)
lsline


subplot(1,3,2)
scatter(kappaRange,best_recovered_kappas)
lsline


subplot(1,3,3)
scatter(betaRange,best_recovered_betas)
lsline


% subplot(2,2,4)
% scatter(sigmaRange,best_recovered_sigma)
% lsline
% 


%% ONLY BETA 

clc
clearvars
% Number of parameters
n_parameters = 60;
% Number of starting points
n_startingPoints = 15;
% set range of parameters
alpha_fixed = 0.25; % Fixed alpha value
kappa_fixed = 15; % Fixed kappa value
betaRange = unifrnd(0,2,[n_parameters,1]); % Only beta varies
n_simulations = 1;
% Number of trials in each simulation
n_trials = 200;
options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'interior-point');
% Initial parameter guesses - only for beta
initBeta = unifrnd(0,2,[n_parameters,n_startingPoints]);
lb = [0]; % Lower bounds for beta only
ub = [2]; % Upper bounds for beta only
% Arrays to store best recovered parameters
best_recovered_betas = NaN(n_parameters,1);
best_nlls = Inf(n_parameters,1); % Start with Inf to track the minimum

parfor p = 1:n_parameters
    for sp = 1:n_startingPoints
        % Generate simulated data
        rewards = [rand(1,n_trials./2) < 0.7;rand(1,n_trials./2) < 0.9];
        state = rand(1,n_trials) < 0.5;
        blocks_all = [];
        for nb = 1:n_trials./25
            blocks_all = [repelem(nb,25), blocks_all];
        end
        condiff = state.';
        for s = 1:length(state)
            if state(s) == 0
                condiff(s) = unifrnd(-0.08,0,1);
            else
                condiff(s) = unifrnd(0,0.08,1);
            end
        end
        
        [predictedUp, mu_hat] = predict_allModels.predict_basicRL(...
            [alpha_fixed, kappa_fixed, betaRange(p)],blocks_all,rewards,state,'sample');
        
        % Objective function - only estimate beta, alpha and kappa are fixed
        nll_fun = @(beta_param) fitSlider_ALLmodels.nll_basicRL_integrated(...
            [alpha_fixed, kappa_fixed, beta_param], mu_hat,blocks_all,rewards,condiff);
        
        options = optimset('Display', 'off');
        [recovered_beta, nll] = fmincon(nll_fun, initBeta(p,sp), [], [], [], [], lb, ub, [], options);
        
        % Check if this is the best solution so far
        if nll < best_nlls(p)
            best_nlls(p) = nll;
            best_recovered_betas(p) = recovered_beta;
        end
    end
    fprintf('Currently processing: Parameter %d/%d\n', p, n_parameters);
end

figure
scatter(betaRange,best_recovered_betas)
lsline
xlabel('True Beta')
ylabel('Recovered Beta')
title('Parameter Recovery: Beta Only')

%% recover alpha and kappa

clc
clearvars
% Number of parameters
n_parameters = 60;
% Number of starting points
n_startingPoints = 15;
% set range of parameters
alphaRange = unifrnd(0,0.3,[n_parameters,1]);
beta_fixed = 0.8; % Fixed beta value
kappaRange = unifrnd(1,25,[n_parameters,1]);
n_simulations = 1;
% Number of trials in each simulation
n_trials = 200;
options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'interior-point');
% Initial parameter guesses - for alpha and kappa
initAlpha = unifrnd(0,0.3,[n_parameters,n_startingPoints]);
initKappa = unifrnd(0,20,[n_parameters,n_startingPoints]);
lb = [0, 1]; % Lower bounds [alpha, kappa]
ub = [0.5, 100]; % Upper bounds [alpha, kappa]
% Arrays to store best recovered parameters
best_recovered_alphas = NaN(n_parameters,1);
best_recovered_kappas = NaN(n_parameters,1);
best_nlls = Inf(n_parameters,1); % Start with Inf to track the minimum

parfor p = 1:n_parameters
    for sp = 1:n_startingPoints
        % Generate simulated data
        rewards = [rand(1,n_trials./2) < 0.7;rand(1,n_trials./2) < 0.9];
        state = rand(1,n_trials) < 0.5;
        blocks_all = [];
        for nb = 1:n_trials./25
            blocks_all = [repelem(nb,25), blocks_all];
        end
        condiff = state.';
        for s = 1:length(state)
            if state(s) == 0
                condiff(s) = unifrnd(-0.08,0,1);
            else
                condiff(s) = unifrnd(0,0.08,1);
            end
        end
        
        [predictedUp, mu_hat] = predict_allModels.predict_basicRL(...
            [alphaRange(p), kappaRange(p), beta_fixed],blocks_all,rewards,state,'sample');
        
        % Objective function - estimate alpha and kappa, beta is fixed
        nll_fun = @(params) fitSlider_ALLmodels.nll_basicRL_integrated(...
            [params(1), params(2), beta_fixed], mu_hat,blocks_all,rewards,condiff);
        
        options = optimset('Display', 'off');
        [recovered_params, nll] = fmincon(nll_fun, [initAlpha(p,sp), initKappa(p,sp)], [], [], [], [], lb, ub, [], options);
        
        % Check if this is the best solution so far
        if nll < best_nlls(p)
            best_nlls(p) = nll;
            best_recovered_alphas(p) = recovered_params(1);
            best_recovered_kappas(p) = recovered_params(2);
        end
    end
    fprintf('Currently processing: Parameter %d/%d\n', p, n_parameters);
end

figure
subplot(1,2,1)
hold on
scatter(alphaRange,best_recovered_alphas)
lsline
xlabel('True Alpha')
ylabel('Recovered Alpha')
title('Parameter Recovery: Alpha (Beta Fixed)')

subplot(1,2,2)
scatter(kappaRange,best_recovered_kappas)
lsline
xlabel('True Kappa')
ylabel('Recovered Kappa')
title('Parameter Recovery: Kappa (Beta Fixed)')