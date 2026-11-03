% Parameter Recovery Script
clc
clearvars

% Number of parameters
n_parameters = 100;

% Number of starting points
n_startingPoints = 30;

% Define true parameters
true_kappa = unifrnd(1,50,[n_parameters,1]); % We'll keep this fixed for simplicity
true_sigma = unifrnd(0,0.1,[n_parameters,1]); % unifrnd(0,0.05,[n_parameters,1]);
true_alpha = unifrnd(0,0.3,[n_parameters,1]); 

% Number of simulations
n_simulations = 1;

% Number of trials in each simulation
n_trials = 500;

options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'interior-point');

% Initial parameter guesses
initKappa = unifrnd(0,30,[n_parameters,n_startingPoints]); % [alpha, beta]
initSigma = unifrnd(0,0.1,[n_parameters,n_startingPoints]);
initAlpha = unifrnd(0,1,[n_parameters,n_startingPoints]);
lb = [0, 1, 0]; % Lower bounds (alpha, beta must be >= 0)
ub = [1, 100, 0.1]; % Upper bounds (alpha <= 1, beta reasonable range)

% Arrays to store best recovered parameters
best_recovered_kappas = NaN(n_parameters,1);
best_recovered_sigmas = NaN(n_parameters,1);
best_recovered_alphas = NaN(n_parameters,1);
best_nlls = Inf(n_parameters,1); % Start with Inf to track the minimum

% Arrays to store recovered parameters
recovered_kappas = NaN(n_startingPoints, n_parameters);
recovered_sigmas = NaN(n_startingPoints, n_parameters);
recovered_alphas = NaN(n_startingPoints, n_parameters);
nllAll = NaN(n_startingPoints,n_parameters);

data = importdata("preprocessed_dataFitting.mat");
data = data(data.choice_cond ~= 3,:);
uniqueID = unique(data.ID);
numSubjs = length(uniqueID);

% Precompute contrast difference
data.condiff_relative = (data.contrast_left - data.contrast_right) ./ 2;
subj = preprocess_fitSlider(data, uniqueID(1));


parfor p = 1:n_parameters-2

    subj = preprocess_fitSlider(data, uniqueID(p));
    params = [true_alpha(p), true_kappa(p), true_sigma(p)];
    [predictedUp, mu_hat] = predict_allModels.predict_PWRL(...
        params, subj.blocks, subj.recoded_rewards, subj.condiff, subj.choices, ...
        subj.recoded_rewards, subj.state, 'mean',subj.contrast);

    for sp = 1:n_startingPoints


        % Optimize using fmincon
        options = optimoptions('fmincon', ...
            'Display', 'off', ...
            'Algorithm', 'active-set', ...
            'FiniteDifferenceType', 'central', ...
            'MaxIterations', 500, ...
            'FunctionTolerance', 1e-4);
        nll_fun = @(params) fitSlider_ALLmodels.nll_PWRL(params, mu_hat, subj.blocks, subj.recoded_rewards, ...
               subj.condiff, subj.choices, subj.recoded_rewards, subj.contrast);

        options = optimset('Display', 'off');
        [recovered_params, nll] = fmincon(nll_fun, [initAlpha(p,sp), initKappa(p,sp), initSigma(p,sp)], [], [], [], [], lb, ub, [], options);

        % Store recovered parameters
        recovered_sigmas(sp,p) = recovered_params(3);
        recovered_kappas(sp,p) = recovered_params(2);
        recovered_alphas(sp,p) = recovered_params(1);

        nllAll(sp,p) = nll;

        % Check if this is the best solution so far
        if nll < best_nlls(p)
            best_nlls(p) = nll;
            best_recovered_kappas(p) = recovered_params(2);
            best_recovered_sigmas(p) = recovered_params(3);
            best_recovered_alphas(p) = recovered_params(1);
        end

        fprintf('Currently processing: Parameter %d/%d, Starting Point %d/%d\n', ...
                p, n_parameters, sp, n_startingPoints);
    end

end

% Plot results
color = lines(4);
color = color(3,:);
figure
hold on
scatter(true_kappa, best_recovered_kappas, "filled", "o", "MarkerEdgeColor", ...
    color, 'MarkerFaceColor', color, 'MarkerFaceAlpha', 0.4, SizeData=50)
lsline
xlabel('True kappa')
ylabel('Best recovered kappa')
title('Posterior-weighted RL')

figure
hold on
scatter(true_sigma, best_recovered_sigmas, "filled", "o", "MarkerEdgeColor", ...
    color, 'MarkerFaceColor',color, 'MarkerFaceAlpha', 0.4, SizeData=50)
lsline
xlabel('True sigma')
ylabel('Best recovered sigma')
title('Posterior-weighted RL')

figure
hold on
scatter(true_alpha, best_recovered_alphas, "filled", "o", "MarkerEdgeColor", ...
    color, 'MarkerFaceColor', color, 'MarkerFaceAlpha', 0.4, SizeData=50)
lsline
xlabel('True alpha')
ylabel('Best recovered alpha')
title('Posterior-weighted RL')

% Print summary statistics
% fprintf('Mean True Alpha: %.2f, Mean Best Recovered Alpha: %.2f\n', mean(true_alpha), mean(best_recovered_alphas));
% fprintf('Mean True Beta: %.2f, Mean Best Recovered Beta: %.2f\n', mean(true_beta), mean(best_recovered_kappas));

