function nll = neg_log_likelihoodPWRL(params, choices, rewards, state)

    % set the seed
    rng('default');
    
    alpha = params(1);
    T = params(2);
    rho = params(3);
    
    % Initialize Q-values
    q_0_0 = 0.5;
    q_0_1 = 1 - q_0_0;
    q_1_0 = 1 - q_0_0;
    q_1_1 = q_0_0;
    Q_fit = [q_0_0, q_0_1; q_1_0, q_1_1]; % initialisation
    n_trials = length(choices);
    nll = 0;
    
    % sampled action from WTA given previous actions up to n-1 to use to estimate the likelihoods of selected actions
    An_predicted = NaN(n_trials,1);
    
    
    for n = 1:n_trials % looping over number of trials

        % Compute choice probabilities using softmax
    
        % Update the distribution of actions given a state s
        P_action_state = Update_action_distribution(Q_fit,state(n),T,rho);
    
        % select a motor action based on the calculated distribution
        % An_predicted(n,:) = sample_discrete(P_action_state)-1;

        % Compute negative log-likelihood
        if state(n) == 0
            if choices(n) == 0
                nll = nll - log(P_action_state(1));
            else
                nll = nll - log(P_action_state(2));
            end
        else
            if choices(n) == 1
                nll = nll - log(P_action_state(2));
            else
                nll = nll - log(P_action_state(1));
            end
        end
    
        % update Q
        Q_fit = update_state_action_valuePG(Q_fit,state(n),choices(n),rewards(n),alpha,rho);

%         if choices(n) == 0
%             Q_fit(2,2) = Q_fit(1,1);
%             Q_fit(1,2) = Q_fit(2,1);
%         else
%             Q_fit(1,1) = Q_fit(2,2);
%             Q_fit(2,1) = Q_fit(1,2);
%         end
    
    end
    
    % Compute -loglik
%     B = (An_predicted == choices);
%     NeglogML = -sum(log(mean(B, 2)));
    
    % Equivalent to this for loop:
    %     logML = 0;
    %     for n = 1:N
    %         logML = logML + log(mean(An_predicted(n,:)==A(n))); % updating the log likelihood
    %     end
    %     NeglogML = -logML;

end
