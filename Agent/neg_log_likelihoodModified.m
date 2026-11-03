function nll = neg_log_likelihoodModified(params, choices, rewards, state, ~)

    % Negative log likelihood function for Q-learning model
    alpha = params(1); % Learning rate
    beta = params(2);  % Inverse temperature
    
    % Initialize Q-values
    q_0_0 = 0.5;
    n_trials = length(choices);
    nll = 0;

    for t = 1:n_trials

        % Compute choice probabilities using softmax
        q_0_1 = 1-q_0_0;
        q_1_0 = 1-q_0_0;
        q_1_1 = q_0_0;  

        Qvalues = [q_0_0, q_0_1];
        p_a_t = exp(Qvalues.*beta) / sum(exp(Qvalues.*beta));
        
        % Avoid log(0) by setting lower bounds
        p1 = max(p_a_t(1), 1e-10);
        p2 = max(p_a_t(2), 1e-10);
        
        % Compute negative log-likelihood
        if state(t) == 0
            if choices(t) == 0
                nll = nll - log(p1);
            else
                nll = nll - log(p2);
            end
        else
            if choices(t) == 1
                nll = nll - log(p1);
            else
                nll = nll - log(p2);
            end
        end
        
        % Update Q-values based on reward
        if state(t) == 0
            q_0_0 = q_0_0 + alpha .* (rewards(t) - q_0_0);
        else            
            q_0_0 = q_0_0 + alpha .* ((1-rewards(t)) - q_0_0);
        end

    end
end
