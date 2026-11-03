function nll = neg_log_likelihoodEstSensitivity(params, choices, rewards, state, condiff, contrast)

    % Negative log likelihood function for Q-learning model
    alpha = params(1); % Learning rate
    beta = params(2);  % Inverse temperature
    sigma = params(3); % Perceptual sensitivity
    
    % Initialize Q-values
    q_0_0 = 0.5;
    n_trials = length(choices);
    nll = 0;

    for t = 1:n_trials

%         if mod(t,25) == 0
%             q_0_0 = 0.5;
%         end

        % Compute belief about perceptual state based on σ
        belief_state = normcdf(condiff(t), 0, sigma);

        if condiff(t) < 0
            pi_0 = 1-belief_state;
            pi_1 = belief_state;
        else
            pi_1 = belief_state;
            pi_0 = 1-belief_state;
        end

        % Compute choice probabilities using softmax
        q_0_1 = 1-q_0_0;
        q_1_0 = 1-q_0_0;
        q_1_1 = q_0_0;  

%         if belief_state(1) > 0.5
%             Qvalues = [q_0_0, q_0_1];
%         else
%             Qvalues = [q_0_1, q_0_0];
%         end

        capital_q_0 = q_0_0 * pi_0 + q_1_0 * pi_1;
        capital_q_1 = q_0_1 * pi_0 + q_1_1 * pi_1;

        capital_q_a = [capital_q_0, capital_q_1];
%         # Eq. 25
%         capital_q_0 = q_0_0 * pi_0 + q_1_0 * pi_1
%         capital_q_1 = q_0_1 * pi_0 + q_1_1 * pi_1
% 
%         capital_q_a = [capital_q_0, capital_q_1]

        % Compute action probabilities using softmax
        p_a_t = exp(capital_q_a.*beta) / sum(exp(capital_q_a.*beta));

%         self.p_a_t = self.softmax(self.v_a_t)
% 
%         # This could be improved such that sampling only takes place during simulations
%         self.a_t = np.random.binomial(1, self.p_a_t[1])
        
        % Avoid log(0) by setting lower bounds
        p1 = max(p_a_t(1), 1e-10);
        p2 = max(p_a_t(2), 1e-10);
        
        % Compute negative log-likelihood
        if contrast(t) == 0
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
        else
            if state(t) == 0
                if choices(t) == 1
                    nll = nll - log(p1);
                else
                    nll = nll - log(p2);
                end
            else
                if choices(t) == 0
                    nll = nll - log(p2);
                else
                    nll = nll - log(p1);
                end
            end
        end

%         if state(t) == 0
%             q_0_0 = q_0_0 + alpha .* (rewards(t) - q_0_0);
%         else            
%             q_0_0 = q_0_0 + alpha .* ((1-rewards(t)) - q_0_0);
%         end

        % Update Q-values based on reward
        if pi_0 >= pi_1
            q_0_0 = q_0_0 + pi_0 * alpha * (rewards(t) - q_0_0);
        else
            q_0_0 = q_0_0 + pi_1 * alpha * ((1 - rewards(t)) - q_0_0);
        end

    end
end
