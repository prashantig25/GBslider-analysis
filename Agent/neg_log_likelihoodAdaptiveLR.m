function nll = neg_log_likelihoodAdaptiveLR(params, choices, rewards, state, condiff)

    % Extract parameters
    base_alpha = params(1); % Base learning rate
    beta = params(2);  % Inverse temperature

    % Initialize Q-values
    q_0_0 = 0.5;
    n_trials = length(choices);
    nll = 0;

    for t = 1:n_trials

        % Compute choice probabilities using softmax
        q_0_1 = 1 - q_0_0;
        q_1_0 = 1 - q_0_0;
        q_1_1 = q_0_0;  

        if state(t) == 0
            Qvalues = [q_0_0, q_0_1];
        else
            Qvalues = [q_0_1, q_0_0];
        end
        p_a_t = exp(Qvalues .* beta) / sum(exp(Qvalues .* beta));
        
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
                nll = nll - log(p2);
            else
                nll = nll - log(p1);
            end
        end
        
        % Adjust learning rate based on contrast difference
        alpha_t = base_alpha * (1 + condiff(t));  % Example: scaling alpha with condiff
        alpha_t = min(max(alpha_t, 0), 1); % Ensure alpha remains within [0,1]

        % Update Q-values using adaptive learning rate
        if state(t) == 0
            q_0_0 = q_0_0 + alpha_t * (rewards(t) - q_0_0);
        else            
            q_0_0 = q_0_0 + alpha_t * ((1 - rewards(t)) - q_0_0);
        end
    end
end
