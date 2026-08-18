clc
clearvars

% Number of parameters
n_parameters = 15;

% Number of starting points
n_startingPoints = 20;

% PARAMETERS TO ESTIMATE: alpha, sigma, kappa
sigmaRange = unifrnd(0, 0.07, [n_parameters, 1]);

n_simulations = 1;
% Number of trials in each simulation
n_trials = 1000;

% Initial parameter guesses for ALPHA, KAPPA, SIGMA ONLY
initSigma = unifrnd(0, 0.05, [n_parameters, n_startingPoints]);

% Bounds for ALPHA, KAPPA, SIGMA
lb = [0];     % Lower bounds for [alpha, kappa, sigma]
ub = [0.07];   % Upper bounds for [alpha, kappa, sigma]

% Arrays to store best recovered parameters
best_recovered_sigmas = NaN(n_parameters, 1);
best_nlls = Inf(n_parameters, 1); % Start with Inf to track the minimum

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
parfor p = 1:n_parameters

    % Generate perceptual choice data
    [choices] = simulate_perceptualChoice(sigmaRange(p), condiff, blocks, nBlocks, block_size);
    data = table();
    data.choice = choices.';
    data.condiff_relative = condiff;
    data.blocks = blocks;

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
        nll_fun = @(params) nll_perceptualChoice(params, data, nBlocks, block_size, blocks);

        % Optimize alpha, kappa, sigma parameters
        [recovered_params, nll] = fmincon(nll_fun, initSigma(p,sp), ...
            [], [], [], [], lb, ub, [], options);
        
        % Check if this is the best solution so far
        if nll < best_nlls(p)
            best_nlls(p) = nll;
            best_recovered_sigmas(p) = recovered_params(1);
        end
    end
    fprintf('Currently processing: Parameter %d/%d, Starting Point %d/%d\n', ...
        p, n_parameters, sp, n_startingPoints);
end

% Plotting results
color = copper(5);

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

function nll = nll_perceptualChoice(params, data, nBlocks, nTrials, blocks)
sigma = params(1);
nll_trial = NaN(nTrials,nBlocks);
uniqueBlocks = unique(blocks);

for bl = 1:nBlocks
    agent = Agent();
    agent.task_agent_analysis = 1;
    agent.confirmation_bias = 0;
    agent.sigma = sigma;
    dataBlocks = data(data.blocks == uniqueBlocks(bl),:);
    choices = dataBlocks.choice;
    condiff = dataBlocks.condiff_relative;
   for t = 1:height(dataBlocks)

        % Bayesian agent inference and learning steps
        agent.o_t = condiff(t);
        agent.p_s_giv_o(agent.o_t);
        agent.decide_p();

        % Log-likelihood for this trial
        % nll_trial(t,bl) = log(agent.p_d_t(choices(t) + 1));

        p = max(min(agent.p_d_t(choices(t) + 1), 1 - 1e-10), 1e-10);
        nll_trial(t,bl) = log(p);
    end
end
nll = -nansum(nll_trial,"all");
end

function [choicesAll] = simulate_perceptualChoice(params, condiff, blocks, nBlocks, trials)
sigma = params(1);
choicesAll = [];
agent = Agent();
for bl = 1:nBlocks
    agent.task_agent_analysis = 1;
    agent.confirmation_bias = 0;
    agent.sigma = sigma;
    choices = [];
    block_indices = (bl-1)*trials + (1:trials);
   for t = 1:trials

        % Bayesian agent inference and learning steps
        agent.o_t = condiff(block_indices(t));
        agent.p_s_giv_o(agent.o_t);
        agent.decide_p();
        choices(t) = agent.d_t;
   end
   choicesAll = [choicesAll, choices];
end
end