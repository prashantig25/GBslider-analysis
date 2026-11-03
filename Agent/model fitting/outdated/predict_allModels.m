classdef predict_allModels
    % RLModelPredictor
    %
    % This class provides static methods for predicting updates and beliefs
    % (mu_hat) in various reinforcement learning and Bayesian models.
    %
    % All methods are static and can be called without creating an object.
    % Each method returns both the predicted updates and the mu_hat vector.
    %
    % Example usage:
    %   [updates, mu_hat] = RLModelPredictor.predict_basicRL(params, blocks, rewards, state);

    methods (Static)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % BASIC RL
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % function [predicted_updates, mu_hat, pe] = predict_basicRL(params, blocks, rewards, state, condiff, mode)
        %     % predict_basicRL - Basic Rescorla-Wagner RL model
        %     %
        %     % Inputs:
        %     %   params  - [alpha, kappa] learning rate and beta prior strength
        %     %   blocks  - vector of block indices for each trial
        %     %   rewards - vector of reward outcomes (0/1)
        %     %   state   - vector of state indicators (0 or 1)
        %     %   mode    - 'mean' (default) or 'sample' for beta distribution
        %     %
        %     % Outputs:
        %     %   predicted_updates - trial-to-trial change in mu_hat
        %     %   mu_hat            - belief estimate for each trial
        % 
        %     if nargin < 5, mode = 'mean'; end
        %     alpha = params(1); kappa = params(2); sigma = params(3);
        %     nTrials = length(blocks);
        %     q_0_0 = 0.5;
        %     mu_hat = zeros(nTrials,1);
        %     pe = NaN(nTrials,1);
        % 
        %     for n = 1:nTrials
        % 
        %         % Compute belief state from perceptual sensitivity
        %         belief_state = normcdf(condiff(n), 0, sigma);
        %         if condiff(n) < 0
        %             pi_0 = 1 - belief_state;
        %             pi_1 = belief_state;
        %         else
        %             pi_1 = belief_state;
        %             pi_0 = 1 - belief_state;
        %         end
        % 
        %         % Reset Q-value at block transitions
        %         if n > 1 && blocks(n) ~= blocks(n-1)
        %             q_0_0 = 0.5;
        %         end
        %         % Q-value update depending on state
        %         if pi_0 >= pi_1
        %             pe(n) = rewards(n) - q_0_0;
        %             q_0_0 = q_0_0 + alpha * (rewards(n) - q_0_0);
        %         else
        %             pe(n) = rewards(n) - q_0_0;
        %             q_0_0 = q_0_0 + alpha * ((1 - rewards(n)) - q_0_0);
        %         end % recoded for action = 0
        %         % q_transformed = anti_sigmoidPG(q_0_0,beta);
        %         q_transformed = q_0_0;
        %         % Compute posterior mean or sample from beta
        %         a = q_transformed * kappa;
        %         b_beta = (1 - q_transformed) * kappa;
        %         switch lower(mode)
        %             case 'mean'
        %                 mu_hat(n) = a / (a + b_beta);
        %             case 'sample'
        %                 mu_hat(n) = betarnd(a, b_beta);
        %             otherwise
        %                 error('Unknown mode. Use ''mean'' or ''sample''.');
        %         end
        %     end
        %     predicted_updates = [NaN; diff(mu_hat)];
        % end

        function [predicted_updates, mu_hat, pe, choice, reward] = predict_basicRL(params, blocks, state, condiff, mode)
            % simulate_basicRL — Basic RL model that also generates simulated choices & rewards
            %
            % Inputs:
            %   params  - [alpha, kappa, sigma]
            %   blocks  - vector of block indices (defines resets)
            %   state   - vector of true states (0 or 1)
            %   condiff - perceptual evidence per trial (affects belief via sigma)
            %   mode    - 'mean' (default) or 'sample' for beta distribution
            %
            % Outputs:
            %   predicted_updates - trial-to-trial change in mu_hat
            %   mu_hat            - posterior belief per trial
            %   pe                - prediction error per trial
            %   choice            - simulated choice per trial (0 or 1)
            %   reward            - simulated reward per trial (0 or 1)

            if nargin < 5, mode = 'mean'; end

            alpha = params(1);
            kappa = params(2);
            sigma = params(3);

            nTrials = length(blocks);

            % Initialise variables
            q_0_0 = 0.5;           % initial Q-value
            mu_hat = zeros(nTrials,1);
            pe = NaN(nTrials,1);
            choice = NaN(nTrials,1);
            reward = NaN(nTrials,1);

            for n = 1:nTrials
                % Reset Q at block transitions
                if n > 1 && blocks(n) ~= blocks(n-1)
                    q_0_0 = 0.5;
                end

                % Compute belief from perceptual sensitivity (sigmoid of condiff)
                belief_state = normcdf(condiff(n), 0, sigma);
                if condiff(n) < 0
                    pi_0 = 1 - belief_state;
                    pi_1 = belief_state;
                else
                    pi_1 = belief_state;
                    pi_0 = 1 - belief_state;
                end

                % === 1️⃣ SIMULATE CHOICE ===
                % Treat mu_hat(n) as probability of choosing action = 1
                if n > 1 && blocks(n) ~= blocks(n-1)
                    if pi_0 >= pi_1
                        choice(n) = rand < 0.5;  % Bernoulli draw
                    else
                        choice(n) = 1-(rand < 0.5);
                    end
                elseif n == 1
                    if pi_0 >= pi_1
                        choice(n) = rand < 0.5;  % Bernoulli draw
                    else
                        choice(n) = 1-(rand < 0.5);
                    end
                else
                    if pi_0 >= pi_1
                        [~,pos] = max([mu_hat(n-1), 1-mu_hat(n-1)]);  % Bernoulli draw
                        choice(n) = pos - 1;
                    else
                        [~,pos] = max([1-mu_hat(n-1), mu_hat(n-1)]);
                        choice(n) = pos - 1;
                    end
                end

                % === 2️⃣ GENERATE REWARD ===
                % Define reward as 1 if choice matches state, 0 otherwise
                % (Optionally add noise, e.g., reward = rand < 0.8 if correct)
                if choice(n) == state(n)
                    reward(n) = rand < 0.9;
                else
                    reward(n) = rand < 0.1;
                end

                if choice(n) == 1
                    reward(n) = 1-reward(n);
                end


                % === 3️⃣ UPDATE Q-VALUE USING REWARD ===
                if pi_0 >= pi_1
                    pe(n) = reward(n) - q_0_0;
                    q_0_0 = q_0_0 + alpha * (reward(n) - q_0_0);
                else
                    pe(n) = reward(n) - q_0_0;
                    q_0_0 = q_0_0 + alpha * ((1-reward(n)) - q_0_0);
                end

                % Compute current posterior parameters for mu_hat
                q_transformed = q_0_0;
                a = q_transformed * kappa;
                b_beta = (1 - q_transformed) * kappa;

                switch lower(mode)
                    case 'mean'
                        mu_hat(n) = a / (a + b_beta);
                    case 'sample'
                        mu_hat(n) = betarnd(a, b_beta);
                    otherwise
                        error('Unknown mode. Use ''mean'' or ''sample''.');
                end
            end

            predicted_updates = [NaN; diff(mu_hat)];
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % RL + ESTIMATED SENSITIVITY (SIGMA)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [predicted_updates, mu_hat, rewards, choice] = predict_RLsigma(params, blocks, state, condiff, mode)
            % predict_RLsigma - RL model with perceptual sensitivity
            %
            % Inputs:
            %   params  - [alpha, kappa, sigma]
            %   blocks  - vector of block indices
            %   rewards - vector of reward outcomes (0/1)
            %   state   - vector of state indicators (0 or 1)
            %   condiff - vector of contrast differences
            %   mode    - 'mean' (default) or 'sample'
            %
            % Outputs:
            %   predicted_updates - trial-to-trial change in mu_hat
            %   mu_hat            - belief estimate for each trial

            if nargin < 5, mode = 'mean'; end
            alpha = params(1); kappa = params(2); sigma = params(3); % beta = params(4);
            nTrials = length(blocks);
            q_0_0 = 0.5;
            mu_hat = zeros(nTrials,1);

            for n = 1:nTrials
                % rng(123)
                % Compute belief state from perceptual sensitivity
                belief_state = normcdf(condiff(n), 0, sigma);
                % if condiff(n) < 0
                pi_0 = 1 - belief_state;
                pi_1 = belief_state;
                % else
                %     pi_1 = belief_state;
                %     pi_0 = 1 - belief_state;
                % end
                % Reset Q-value at block transitions
                if n > 1 && blocks(n) ~= blocks(n-1)
                    q_0_0 = 0.5;
                end

                % === 1️⃣ SIMULATE CHOICE ===
                % Treat mu_hat(n) as probability of choosing action = 1
                if n > 1 && blocks(n) ~= blocks(n-1)
                    if pi_0 >= pi_1
                        choice(n) = rand < 0.5;  % Bernoulli draw
                    else
                        choice(n) = 1-(rand < 0.5);
                    end
                elseif n == 1
                    if pi_0 >= pi_1
                        choice(n) = rand < 0.5;  % Bernoulli draw
                    else
                        choice(n) = 1-(rand < 0.5);
                    end
                else
                    if pi_0 >= pi_1
                        [~,pos] = max([mu_hat(n-1), 1-mu_hat(n-1)]);  % Bernoulli draw
                        choice(n) = pos - 1;
                    else
                        [~,pos] = max([1-mu_hat(n-1), mu_hat(n-1)]);
                        choice(n) = pos - 1;
                    end
                end

                % === 2️⃣ GENERATE REWARD ===
                % Define reward as 1 if choice matches state, 0 otherwise
                % (Optionally add noise, e.g., reward = rand < 0.8 if correct)
                if choice(n) ~= state(n)
                    rewards(n) = rand < 0.2;   % always rewarded if correct
                else
                    rewards(n) = rand < 0.8;   % always no reward if wrong
                end

                if choice(n) == 1
                    rewards(n) = 1-rewards(n);
                end

                % Belief-weighted Q-update
                if pi_0 >= pi_1
                    q_0_0 = q_0_0 + pi_0 * alpha * (rewards(n) - q_0_0);
                else
                    q_0_0 = q_0_0 + pi_1 * alpha * ((1-rewards(n)) - q_0_0);
                end
                % if beta < 0
                %     disp('Negative beta')
                % end
                % q_transformed = anti_sigmoidPG(q_0_0,beta);
                q_transformed = q_0_0;
                % Compute posterior mean or sample from beta
                a = q_transformed * kappa;
                b_beta = (1 - q_transformed) * kappa;
                switch lower(mode)
                    case 'mean'
                        mu_hat(n) = a / (a + b_beta);
                    case 'sample'
                        mu_hat(n) = betarnd(a, b_beta);
                    otherwise
                        error('Unknown mode. Use ''mean'' or ''sample''.');
                end
            end

            predicted_updates = [NaN; diff(mu_hat)];
        end

        function [predicted_updates, mu_hat, rewards, choice] = predict_RLsigma_updated(params, blocks, state, condiff, mode)
    if nargin < 5, mode = 'mean'; end
    
    alpha = params(1); 
    kappa = params(2); 
    sigma = params(3);
    
    nTrials = length(blocks);
    
    % Track Q-values for both state-action pairs
    q_0_0 = 0.5;  % Q(s=0, a=0)
    q_0_1 = 0.5;  % Q(s=0, a=1)
    
    mu_hat = zeros(nTrials, 1);
    choice = zeros(nTrials, 1);
    rewards = zeros(nTrials, 1);
    
    for n = 1:nTrials
        % rng(123)
        % Compute belief state from perceptual sensitivity
        belief_state = normcdf(condiff(n), 0, sigma);
        pi_0 = 1 - belief_state;  % belief in state 0
        pi_1 = belief_state;      % belief in state 1
        
        % Reset Q-values at block transitions
        if n > 1 && blocks(n) ~= blocks(n-1)
            q_0_0 = 0.5;
            q_0_1 = 0.5;
        end
        
        % === COMPUTE EXPECTED VALUES ===
        % Expected value of each action, weighted by belief about state
        % Since task is symmetric, Q(s=1,a=1) ≈ Q(s=0,a=0) and Q(s=1,a=0) ≈ Q(s=0,a=1)
        v_action_0 = pi_0 * q_0_0 + pi_1 * q_0_1;  % EV of choosing action 0
        v_action_1 = pi_0 * q_0_1 + pi_1 * q_0_0;  % EV of choosing action 1 (symmetric)
        
        % === SIMULATE CHOICE ===
        if v_action_1 > v_action_0
            choice(n) = 1;
        elseif v_action_0 > v_action_1
            choice(n) = 0;
        else
            choice(n) = rand < 0.5;  % Random choice if tied
        end
        
        % === GENERATE REWARD ===
        % 80% reward probability when choice matches state
        if choice(n) == state(n)
            rewards(n) = rand < 0.9;  % 80% chance of reward when correct
        else
            rewards(n) = rand < 0.1;  % 20% chance of reward when incorrect (optional)
        end
        
        % === BELIEF-WEIGHTED Q-UPDATE ===
        % Update Q-value for the chosen action
        if choice(n) == 0
            % Chose action 0: update Q(s=0,a=0) weighted by belief in state 0
            % and update Q(s=0,a=1) weighted by belief in state 1
            q_0_0 = q_0_0 + pi_0 * alpha * (rewards(n) - q_0_0);
            q_0_1 = q_0_1 + pi_1 * alpha * (rewards(n) - q_0_1);
        else
            % Chose action 1: update Q(s=0,a=1) weighted by belief in state 0
            % and update Q(s=0,a=0) weighted by belief in state 1
            q_0_1 = q_0_1 + pi_0 * alpha * (rewards(n) - q_0_1);
            q_0_0 = q_0_0 + pi_1 * alpha * (rewards(n) - q_0_0);
        end
        
        % === COMPUTE POSTERIOR BELIEF ===
        % mu_hat represents reported reward probability
        % This should reflect the expected reward for the better action
        q_transformed = q_0_0; %max(q_0_0, q_0_1);  % Take the better action's value
        
        a = q_transformed * kappa;
        b_beta = (1 - q_transformed) * kappa;
        
        switch lower(mode)
            case 'mean'
                mu_hat(n) = a / (a + b_beta);
            case 'sample'
                mu_hat(n) = betarnd(a, b_beta);
            otherwise
                error('Unknown mode. Use ''mean'' or ''sample''.');
        end
    end
    
    predicted_updates = [NaN; diff(mu_hat)];
end

        function [predicted_updates, mu_hat, rewards, choice] = predict_RLsigma_deterministic(params, blocks, state, condiff, mode)
    % predict_RLsigma - RL model with perceptual sensitivity (deterministic reward version)
    %
    % Inputs:
    %   params  - [alpha, kappa, sigma]
    %   blocks  - vector of block indices
    %   state   - vector of true states (0/1)
    %   condiff - vector of contrast differences
    %   mode    - 'mean' (default) or 'sample'
    %
    % Outputs:
    %   predicted_updates - trial-to-trial change in mu_hat
    %   mu_hat            - belief estimate for each trial
    %   rewards           - deterministic reward vector (0/1)
    %   choice            - chosen action (0/1)

    if nargin < 5, mode = 'mean'; end

    % === Unpack parameters ===
    alpha = params(1);
    kappa = params(2);
    sigma = params(3);

    nTrials = length(blocks);
    mu_hat = zeros(nTrials, 1);
    choice = zeros(nTrials, 1);
    rewards = zeros(nTrials, 1);

    % Initialize Q-value
    q_0_0 = 0.5;

    % === 1️⃣ SIMULATE CHOICES (deterministically based on previous μ) ===
    for n = 1:nTrials
        % Compute perceptual belief from condiff
        belief_state = normcdf(condiff(n), 0, sigma);
        pi_0 = 1 - belief_state;
        pi_1 = belief_state;

        % Reset Q at block transitions
        if n > 1 && blocks(n) ~= blocks(n-1)
            q_0_0 = 0.5;
        end

        % --- Deterministic choice policy ---
        if n == 1 || (n > 1 && blocks(n) ~= blocks(n-1))
            % Random first trial per block for variability
            choice(n) = double(pi_1 > pi_0);
        else
            % Deterministic based on previous μ
            if pi_1 >= pi_0
                choice(n) = double(mu_hat(n-1) >= 0.5);
            else
                choice(n) = double(mu_hat(n-1) < 0.5);
            end
        end

        % === 2️⃣ PREDEFINE REWARDS DETERMINISTICALLY ===
        % Create reward schedule exactly matching 80%/20%
        % (computed only once per state-choice pattern)
        if n == 1
            rng(123); % fixed seed for reproducibility
        end
    end

    % Now that all choices are made, assign deterministic rewards
    idx_incorrect = find(choice ~= state);
    idx_correct   = find(choice == state);

    % Predefine exactly 80% rewards for incorrect, 20% for correct
    n_reward_incorrect = round(0.8 * numel(idx_incorrect));
    n_reward_correct   = round(0.2 * numel(idx_correct));

    % Deterministic but pseudo-random selection for which trials get rewarded
    rng(123);  % ensures reproducible subset selection
    if ~isempty(idx_incorrect)
        rewards(randsample(idx_incorrect, n_reward_incorrect)) = 1;
    end
    if ~isempty(idx_correct)
        rewards(randsample(idx_correct, n_reward_correct)) = 1;
    end

    % === 3️⃣ RUN LEARNING UPDATES ===
    for n = 1:nTrials
        belief_state = normcdf(condiff(n), 0, sigma);
        pi_0 = 1 - belief_state;
        pi_1 = belief_state;

        % Reset Q at block transitions
        if n > 1 && blocks(n) ~= blocks(n-1)
            q_0_0 = 0.5;
        end

        % Belief-weighted Q-update (no randomness)
        if pi_0 >= pi_1
            q_0_0 = q_0_0 + pi_0 * alpha * (rewards(n) - q_0_0);
        else
            q_0_0 = q_0_0 + pi_1 * alpha * ((1 - rewards(n)) - q_0_0);
        end

        % Compute posterior mean or sample
        a = q_0_0 * kappa;
        b_beta = (1 - q_0_0) * kappa;

        switch lower(mode)
            case 'mean'
                mu_hat(n) = a / (a + b_beta);
            case 'sample'
                mu_hat(n) = betarnd(a, b_beta); %#ok<*UNRCH> % only if stochastic mode requested
            otherwise
                error('Unknown mode. Use ''mean'' or ''sample''.');
        end
    end

    predicted_updates = [NaN; diff(mu_hat)];
end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % RL + ESTIMATED SENSITIVITY + confirmBias(SIGMA)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [predicted_updates, mu_hat] = predict_RLsigma_confirmBias(params, blocks, rewards, state, condiff, confirmRew, mode)
            % predict_RLsigma - RL model with perceptual sensitivity
            %
            % Inputs:
            %   params  - [alpha, kappa, sigma]
            %   blocks  - vector of block indices
            %   rewards - vector of reward outcomes (0/1)
            %   state   - vector of state indicators (0 or 1)
            %   condiff - vector of contrast differences
            %   mode    - 'mean' (default) or 'sample'
            %
            % Outputs:
            %   predicted_updates - trial-to-trial change in mu_hat
            %   mu_hat            - belief estimate for each trial

            if nargin < 6, mode = 'mean'; end
            kappa = params(1); sigma = params(2);confirmBias = params(3);noconfirmBias = params(4); % beta = params(5);
            nTrials = length(blocks);
            q_0_0 = 0.5;
            mu_hat = zeros(nTrials,1);

            for n = 1:nTrials
                % Compute belief state from perceptual sensitivity
                belief_state = normcdf(condiff(n), 0, sigma);
                if condiff(n) < 0
                    pi_0 = 1 - belief_state;
                    pi_1 = belief_state;
                else
                    pi_1 = belief_state;
                    pi_0 = 1 - belief_state;
                end
                % Reset Q-value at block transitions
                if n > 1 && blocks(n) ~= blocks(n-1)
                    q_0_0 = 0.5;
                end
                % Belief-weighted Q-update
                % if pi_0 >= pi_1
                %     q_0_0 = q_0_0 + pi_0 * alpha * (rewards(n) - q_0_0) + confirmBias * confirmRew(n) * (rewards(n) - q_0_0);
                % else
                %     q_0_0 = q_0_0 + pi_1 * alpha * ((1 - rewards(n)) - q_0_0) + confirmBias * confirmRew(n) * ((1 - rewards(n)) - q_0_0);
                % end

                if pi_0 >= pi_1
                    if confirmRew(n) == 1
                        q_0_0 = q_0_0 + pi_0 * confirmBias * (rewards(n) - q_0_0); % + confirmBias * (rewards(n) - q_sim);
                    else
                        q_0_0 = q_0_0 + pi_0 * noconfirmBias * (rewards(n) - q_0_0); % + noconfirmBias * (rewards(n) - q_sim);
                    end
                else
                    if confirmRew(n) == 1
                        q_0_0 = q_0_0 + pi_1 * confirmBias * ((1 - rewards(n)) - q_0_0);% + confirmBias  * ((1 - rewards(n)) - q_sim);
                    else
                        q_0_0 = q_0_0 + pi_1 * noconfirmBias * ((1 - rewards(n)) - q_0_0);% + noconfirmBias  * ((1 - rewards(n)) - q_sim);
                    end
                end
                % Compute posterior mean or sample from beta
                % q_transformed = anti_sigmoidPG(q_0_0,beta);
                q_transformed = q_0_0;
                a = q_transformed * kappa;
                b_beta = (1 - q_transformed) * kappa;
                switch lower(mode)
                    case 'mean'
                        mu_hat(n) = a / (a + b_beta);
                    case 'sample'
                        mu_hat(n) = betarnd(a, b_beta);
                    otherwise
                        error('Unknown mode. Use ''mean'' or ''sample''.');
                end
            end
            predicted_updates = [NaN; diff(mu_hat)];
        end


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % BAYESIAN AGENT
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [predicted_updates, mu_hatAll, rewards, choices] = predict_bayesianAgent(params, blocks, condiff, condition, mode, state)
            % predict_bayesianAgent - Bayesian agent model
            %
            % Inputs:
            %   params    - [kappa, sigma]
            %   blocks    - vector of block indices
            %   rewards   - vector of reward outcomes (0/1)
            %   condiff   - vector of contrast differences
            %   choices   - vector of choices
            %   condition - vector of condition indices (one per block)
            %   mode      - 'mean' (default) or 'sample'
            %
            % Outputs:
            %   predicted_updates - trial-to-trial change in mu_hat
            %   mu_hatAll         - belief estimate for each trial (concatenated across blocks)

            kappa = params(1);
            sigma = params(2);
            % beta = params(3);
            uniqueBlocks = unique(blocks);
            mu_hatAll = [];
            rewards = [];
            choices = [];
            for bl = 1:length(uniqueBlocks)
                agent = Agent(); % Assumes Agent class is defined elsewhere
                task = Task();
                task.taskMu(0);
                agent.task_agent_analysis = 0;
                agent.sigma = sigma;
                agent.eval_ana = 1; % analytical solution here
                agent.condition = condition;
                condiffBlock = condiff(blocks == uniqueBlocks(bl));
                stateBlock = state(blocks == uniqueBlocks(bl));
                for t = 1:length(condiffBlock)
                    agent.o_t = condiffBlock(t);
                    agent.p_s_giv_o(agent.o_t);
                    agent.decide_e(agent.o_t);
                    task.s_t = stateBlock(t);
                    task.reward_sample(agent.a_t);% economic choice based reward
                    agent.learn(task.r_t);
                    q_0_0 = agent.G;
                    a = q_0_0 * kappa; b_beta = (1 - q_0_0) * kappa;
                    switch lower(mode)
                        case 'mean',   mu_hat = a / (a + b_beta);
                        case 'sample', mu_hat = betarnd(a, b_beta);
                        otherwise, error('Unknown mode. Use ''mean'' or ''sample''.');
                    end
                    rewards = [rewards; task.r_t];
                    choices = [choices; agent.a_t];
                    mu_hatAll = [mu_hatAll; mu_hat];
                end
            end
            predicted_updates = [NaN; diff(mu_hatAll)];
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % BAYESIAN AGENT + CONFIRMATION BIAS (have to add it yet - update
        % this line when done !!)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [predicted_updates, mu_hatAll, rewards, choices, confirmRew] = predict_bayesianAgent_confirmBias(params, blocks, ...
                condiff, condition, mode, state)
            % predict_bayesianAgent - Bayesian agent model
            %
            % Inputs:
            %   params    - [kappa, sigma]
            %   blocks    - vector of block indices
            %   rewards   - vector of reward outcomes (0/1)
            %   condiff   - vector of contrast differences
            %   choices   - vector of choices
            %   condition - vector of condition indices (one per block)
            %   mode      - 'mean' (default) or 'sample'
            %
            % Outputs:
            %   predicted_updates - trial-to-trial change in mu_hat
            %   mu_hatAll         - belief estimate for each trial (concatenated across blocks)

            kappa = params(1);
            sigma = params(2);
            confirmBias = params(3);
            uniqueBlocks = unique(blocks);
            mu_hatAll = [];
            rewards = [];
            choices = [];
            confirmRew = [];
            for bl = 1:length(uniqueBlocks)

                agent = Agent(); % Assumes Agent class is defined elsewhere
                task = Task();
                task.taskMu(0);
                agent.task_agent_analysis = 0;
                agent.sigma = sigma;
                agent.condition = condition;
                agent.confirmBias = confirmBias;
                agent.eval_ana = 1; % analytical solution here
                condiffBlock = condiff(blocks == uniqueBlocks(bl));
                stateBlock = state(blocks == uniqueBlocks(bl));

                for t = 1:length(stateBlock)
                    agent.confirmation_bias = 1;
                    agent.o_t = condiffBlock(t);
                    agent.p_s_giv_o(agent.o_t);
                    agent.decide_e(agent.o_t);
                    task.s_t = stateBlock(t);
                    task.reward_sample(agent.a_t);% economic choice based reward
                    % Case: match (state == action) → chosen pair is better → use reward directly
                    if task.s_t == agent.a_t
                        agent.confirmation_bias = task.r_t;
                        confirm = task.r_t;
                    else
                        agent.confirmation_bias = 1- task.r_t;
                        confirm = 1 - task.r_t;
                    end
                    agent.learn(task.r_t);
                    q_0_0 = agent.G;
                    a = q_0_0 * kappa; b_beta = (1 - q_0_0) * kappa;
                    switch lower(mode)
                        case 'mean',   mu_hat = a / (a + b_beta);
                        case 'sample', mu_hat = betarnd(a, b_beta);
                        otherwise, error('Unknown mode. Use ''mean'' or ''sample''.');
                    end
                    rewards = [rewards; task.r_t];
                    choices = [choices; agent.a_t];
                    mu_hatAll = [mu_hatAll; mu_hat];
                    confirmRew = [confirmRew; confirm];
                end
            end
            predicted_updates = [NaN; diff(mu_hatAll)];
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % POSTERIOR-WEIGHTED RL (PWRL)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [predicted_updates, mu_hat] = predict_PWRL(params, blocks, rewards, condiff, choices, recoded_rewards, state, mode, contrast)
            % predict_PWRL - Posterior-weighted RL model
            %
            % Inputs:
            %   params          - [alpha, kappa, sigma]
            %   blocks          - vector of block indices
            %   rewards         - vector of reward outcomes (0/1)
            %   condiff         - vector of contrast differences
            %   choices         - vector of choices
            %   recoded_rewards - vector of recoded rewards
            %   state           - vector of state indicators (0 or 1)
            %   mode            - 'mean' (default) or 'sample'
            %
            % Outputs:
            %   predicted_updates - trial-to-trial change in mu_hat
            %   mu_hat            - belief estimate for each trial

            if nargin < 8, mode = 'mean'; end
            alpha = params(1);
            kappa = params(2);
            sigma = params(3);
            nTrials = length(blocks);

            q_0_0 = 0.5;
            q_1_0 = 1 - q_0_0;
            q_1_1 = q_0_0;
            q_0_1 = 1 - q_1_1;
            mu_hat = zeros(nTrials,1);

            for n = 1:nTrials
                % Reset Q-values at block transitions
                if n > 1 && blocks(n) ~= blocks(n-1)
                    q_0_0 = 0.5; q_1_0 = 0.5; q_1_1 = 0.5; q_0_1 = 0.5;
                end

                % Calculate belief state
                belief_state = normcdf(condiff(n), 0, sigma);
                if condiff(n) < 0
                    pi_0 = 1 - belief_state;
                    pi_1 = belief_state;
                else
                    pi_1 = belief_state;
                    pi_0 = 1 - belief_state;
                end

                % Posterior-weighted state inference
                % if choices(n) == 0
                %     pw_pi0_a0 = (q_0_0*pi_0)./((q_0_0*pi_0) + (q_1_0*pi_1));
                %     pw_pi1_a1 = (q_1_0*pi_1)./((q_1_0*pi_1) + (q_0_0*pi_0));
                % else
                %     pw_pi0_a0 = (q_0_1*pi_0)./((q_0_1*pi_0) + (q_1_1*pi_1));
                %     pw_pi1_a1 = (q_1_1*pi_1)./((q_1_1*pi_1) + (q_0_1*pi_0));
                % end

                if choices(n) == 0
                    pw_pi0_a0 = (q_0_0*pi_0)./((q_0_0*pi_0) + (q_1_0*pi_1)); % reward(correct) == 1
                    pw_pi1_a1 = (q_1_0*pi_1)./((q_1_0*pi_1) + (q_0_0*pi_0)); % reward(correct) == 0
                else
                    % if choices is 1, you should believe you are in state
                    % 1. hence belief state 1, should increase with the
                    % contingency parameter
                    % pw_pi0_a0 = (q_0_1*pi_0)./((q_0_1*pi_0) + (q_1_1*pi_1)); % reward(correct) == 1
                    % pw_pi1_a1 = (q_1_1*pi_1)./((q_1_1*pi_1) + (q_0_1*pi_0)); % reward(correct) == 0

                    pw_pi1_a1 = (q_0_1*pi_0)./((q_0_1*pi_0) + (q_1_1*pi_1)); % reward(correct) == 0
                    pw_pi0_a0 = (q_1_1*pi_1)./((q_1_1*pi_1) + (q_0_1*pi_0)); % reward(correct) == 1

                end

                if contrast(n) == 1
                    pw_pi1_a1 = 1 - pw_pi1_a1;
                    pw_pi0_a0 = 1 - pw_pi0_a0;
                end

                if rewards(n) == 1
                    pw_pi = pw_pi0_a0;
                else
                    pw_pi = pw_pi1_a1;
                end

                if q_0_0 == q_1_0
                    if pi_0 >= pi_1
                        pw_pi = pi_0;
                    else
                        pw_pi = pi_1;
                    end
                end

                % Q-value update weighted by posterior belief
                if contrast(n) == 0
                    if pi_0 >= pi_1
                        q_0_0 = q_0_0 + pw_pi * alpha * (recoded_rewards(n) - q_0_0);
                    else
                        q_0_0 = q_0_0 + pw_pi * alpha * ((1 - recoded_rewards(n)) - q_0_0);
                    end
                else
                    q_0_0 = q_0_0 + pw_pi * alpha * (recoded_rewards(n) - q_0_0);
                end

                % Compute posterior mean or sample from beta
                a = q_0_0 * kappa;
                b_beta = (1 - q_0_0) * kappa;
                q_1_1 = q_0_0;
                q_1_0 = 1-q_0_0;
                q_0_1 = q_1_0;
                switch lower(mode)
                    case 'mean',   mu_hat(n) = a / (a + b_beta);
                    case 'sample', mu_hat(n) = betarnd(a, b_beta);
                    otherwise, error('Unknown mode. Use ''mean'' or ''sample''.');
                end
            end
            predicted_updates = [NaN; diff(mu_hat)];
        end

    end
end
