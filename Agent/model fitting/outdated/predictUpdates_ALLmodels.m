classdef predictUpdates_ALLmodels
    % predictUpdates_ALLmodels
    % --------------------------------------------------------------------
    % A helper class with static methods for computing *model-predicted*
    % trial-by-trial internal belief updates (e.g., Q-values or G-values)
    % based on a participant’s estimated parameters and trial data.
    %
    % This can be used to:
    %   - visualize model-implied learning curves
    %   - compare model predictions to observed behavior
    %   - compute model fit diagnostics
    % --------------------------------------------------------------------
    
    methods (Static)

        %% BASIC RL MODEL: Q-learning with one learning rate
        function q_vals = predict_basicRL(params, blocks, state, rewards)
            alpha = params(1);                 % Learning rate
            q_vals = NaN(length(blocks), 1);   % Store Q-value at each trial
            q = 0.5;                           % Initialize Q-value

            for n = 1:length(blocks)
                q_vals(n) = q;                 % Save current Q-value

                % Reset Q at the start of a new block
                if n < length(blocks) && blocks(n+1) ~= blocks(n)
                    q = 0.5;
                    continue;
                end

                % Update rule depends on latent state
                if state(n) == 0
                    q = q + alpha * (rewards(n) - q);
                else
                    q = q + alpha * ((1 - rewards(n)) - q);
                end
            end
        end

        %% RL SIGMA MODEL: Belief-weighted update using perceptual uncertainty
        function q_vals = predict_RLsigma(params, blocks, rewards, condiff)
            alpha = params(1);
            sigma = params(3);                 % Perceptual noise
            q_vals = NaN(length(blocks), 1);   % Q-value prediction
            q = 0.5;                           % Initial Q-value

            for n = 1:length(blocks)
                q_vals(n) = q;

                % Reset at block start
                if n < length(blocks) && blocks(n+1) ~= blocks(n)
                    q = 0.5; continue;
                end

                % Compute belief from sensory evidence
                belief = normcdf(condiff(n), 0, sigma);
                pi_0 = 1 - belief;
                pi_1 = belief;

                % Belief-weighted Q update
                if pi_0 >= pi_1
                    q = q + pi_0 * alpha * (rewards(n) - q);
                else
                    q = q + pi_1 * alpha * ((1 - rewards(n)) - q);
                end
            end
        end

        %% POSTERIOR-WEIGHTED RL MODEL
        function q_vals = predict_PWRL(params, blocks, rewards, condiff, choices, recoded_rewards)
            alpha = params(1);
            sigma = params(3);
            q_vals = NaN(length(blocks), 1);
            q = 0.5; % Initialize q-values

            % Other Q-values used for computing posteriors
            q_1_0 = 1 - q;
            q_0_1 = 1 - q;
            q_1_1 = q;

            for n = 1:length(blocks)
                q_vals(n) = q;

                % Reset at block change
                if n < length(blocks) && blocks(n+1) ~= blocks(n)
                    q = 0.5; 
                    q_1_0 = 1 - q; 
                    q_0_1 = 1 - q; 
                    q_1_1 = q;
                    continue;
                end

                % Compute perceptual belief
                belief = normcdf(condiff(n), 0, sigma);
                pi_0 = 1 - belief;
                pi_1 = belief;

                % Posterior-weighted inference depends on action taken
                if choices(n) == 0
                    pw_pi0_a0 = (q * pi_0) / ((q * pi_0) + (q_1_0 * pi_1));
                    pw_pi1_a1 = (q_1_0 * pi_1) / ((q_1_0 * pi_1) + (q * pi_0));
                else
                    pw_pi0_a0 = (q_0_1 * pi_0) / ((q_0_1 * pi_0) + (q_1_1 * pi_1));
                    pw_pi1_a1 = (q_1_1 * pi_1) / ((q_1_1 * pi_1) + (q_0_1 * pi_0));
                end

                % Choose posterior-weighted update depending on reward
                if rewards(n) == 1
                    pw_pi = pw_pi0_a0;
                else
                    pw_pi = pw_pi1_a1;
                end

                % Fallback if posterior values collapse
                if q == q_1_0
                    pw_pi = max(pi_0, pi_1);
                end

                % Perform update
                if pi_0 >= pi_1
                    q = q + pw_pi * alpha * (recoded_rewards(n) - q);
                else
                    q = q + pw_pi * alpha * ((1 - recoded_rewards(n)) - q);
                end
            end
        end

        %% BAYESIAN AGENT MODEL
        function G_vals = predict_bayesianAgent(params, data, nBlocks, blocks, rewards)
            sigma = params(2);           % Perceptual noise
            G_vals = [];                 % Posterior belief trajectory
            uniqueBlocks = unique(blocks);

            for bl = 1:nBlocks
                agent = Agent();         % Instantiate agent for each block
                agent.sigma = sigma;

                % Filter trial data for this block
                blockData = data(data.blocks == uniqueBlocks(bl), :);
                condiff = blockData.condiff_relative;
                choices = blockData.choice;
                rewardsBlock = rewards(data.blocks == uniqueBlocks(bl));

                for t = 1:height(blockData)
                    % Agent inference and learning
                    agent.o_t = condiff(t);      % sensory input
                    agent.p_s_giv_o(agent.o_t); % belief update
                    agent.compute_valence();    % compute expected reward
                    agent.softmax();            % action policy
                    agent.a_t = choices(t);     % observed choice
                    agent.learn(rewardsBlock(t)); % update internal belief

                    G_vals = [G_vals; agent.G];  % Save agent’s belief
                end
            end
        end
    end
end
