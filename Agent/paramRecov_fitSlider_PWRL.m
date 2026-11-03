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

for p = 1:n_parameters

        % GENERATE SIMULATED DATA
        % rewards = [double(rand(1,n_trials./2) < 0.7), double(rand(1,n_trials./2) < 0.9)].';
        state = double(rand(1,n_trials) < 0.5);
        condiff = state.';
        for s = 1:length(state)
            if state(s) == 0
                condiff(s) = unifrnd(-0.08,0,1);
            else
                condiff(s) = unifrnd(0,0.08,1);
            end
        end

        blocks_all = [];
        for nb = 1:n_trials./25
            blocks_all = [repelem(nb,25), blocks_all];
        end
        condition = [repelem(1,n_trials./2),repelem(2,n_trials./2)];
        [mu_hat_sim,data] = simulate_slider_PWRL([true_alpha(p), true_kappa(p), true_sigma(p)], blocks_all, rewards, condiff, [], 1, [], state, condition);


    for sp = 1:n_startingPoints

        % Optimize using fmincon
        %         options = optimset('Display', 'iter');
        options = optimoptions('fmincon', ...
            'Display', 'off', ...
            'Algorithm', 'active-set', ...
            'FiniteDifferenceType', 'central', ...
            'MaxIterations', 500, ...
            'FunctionTolerance', 1e-4);
        % (params, mu_hat, blocks, rewards, condiff, choices, recoded_rewards)
        [recovered_params,nll] = fmincon(@(params) nll_fitSlider_PWRL([params(1), params(2), params(3)],mu_hat_sim,blocks_all,data.rewards,data.condiff_relative,data.choice,data.recoded_rewards), ...
            [initAlpha(p,sp), initKappa(p,sp), initSigma(p,sp)], [], [], [], [], lb, ub, [], options);

        % Store recovered parameters
        recovered_kappas(sp,p) = recovered_params(3);
        recovered_sigmas(sp,p) = recovered_params(2);
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

