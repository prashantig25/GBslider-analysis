function [Q, p] = Q_model(params, choices, outcomes)
    % Q_model: Calculates Q-values for a basic reinforcement learning model
    %
    % Inputs:
    %   params: Vector of model parameters [learning rate, inverse temperature]
    %   choices: Vector of subject's choices (e.g., 1 or 2 for two-option task)
    %   outcomes: Vector of outcomes (typically rewards, e.g., 0 or 1)
    %
    % Outputs:
    %   Q: Matrix of Q-values for each trial and option
    %   p: Vector of choice probabilities for the chosen option on each trial

    alpha = params(1);  % learning rate
    beta = params(2);   % inverse temperature (for softmax)

    nTrials = length(choices);
    nOptions = max(choices);

    % Initialize Q-values
    Q = zeros(nTrials + 1, nOptions);
    p = zeros(nTrials, 1);

    for t = 1:nTrials
        % Calculate choice probabilities using softmax
        expQ = exp(beta * Q(t, :));
        p(t) = expQ(choices(t)) / sum(expQ);

        % Update Q-value for the chosen option
        delta = outcomes(t) - Q(t, choices(t));
        Q(t+1, :) = Q(t, :);
        Q(t+1, choices(t)) = Q(t, choices(t)) + alpha * delta;
    end
end
