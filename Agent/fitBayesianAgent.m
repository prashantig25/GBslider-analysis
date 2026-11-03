function params = fitBayesianAgent(data, sigma)
    % Fit a normative bayesian agent to participant choice data and
    % estimated sensitivity parameter
    % Inputs:
    %   choices - vector of choices (1 or 2)
    %   rewards - vector of rewards (0 or 1)
    % Output:
    %   params - fitted parameters [beta]

    % Initial parameter guesses
    init_params = [2,0.02]; % [alpha, beta]
    lb = [0,0]; % Lower bounds (alpha, beta must be >= 0)
    ub = [10,0.1]; % Upper bounds (alpha <= 1, beta reasonable range)
    
    % Optimize using fmincon
    options = optimset('Display', 'off');
    params = fmincon(@(p) NLL_fitSigmaBeta_Agent(p, data), init_params, [], [], [], [], lb, ub, [], options);
end