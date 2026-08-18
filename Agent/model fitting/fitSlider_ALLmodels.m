classdef fitSlider_ALLmodels
    % fitSlider
    % A class containing static methods for computing negative log-likelihoods (NLL)
    % for different reinforcement learning (RL) models fitted to participant data.
    %
    % Each method implements a specific RL model's likelihood function.
    % Use these methods for model fitting and comparison.

    properties
        % No instance properties needed; all methods are static
    end

    methods (Static)

        %% ===================================================================
        %  BASIC RL MODEL
        %  -------------------------------------------------------------------
        %  Standard Q-learning update with a single learning rate (alpha) and
        %  a kappa concentration parameter for the Beta likelihood.
        %  Inputs:
        %    params  - [alpha, kappa]
        %    mu_hat  - observed values (vector)
        %    blocks  - block indices (vector)
        %    state   - state indicator (vector)
        %    rewards - reward outcomes (vector)
        % ====================================================================
        function nll = nll_basicRL(params, mu_hat, blocks, state, rewards)
            alpha = params(1);
            kappa = params(2);
            nll_trial = NaN(length(mu_hat),1);
            eps = 1e-9; % Prevent log(0)
            mu_hat(mu_hat <= 0) = eps;
            mu_hat(mu_hat >= 1) = 1 - eps;
            q_0_0 = 0.5; % Initial Q-value

            for n = 1:length(mu_hat)-1

                % Reset Q-value at block transitions
                if blocks(n+1) - blocks(n) == 1
                    q_0_0 = 0.5;
                end

                % Q-value update based on state and reward
                if state(n) == 0
                    q_0_0 = q_0_0 + alpha * (rewards(n) - q_0_0);
                else
                    q_0_0 = q_0_0 + alpha * ((1 - rewards(n)) - q_0_0);
                end

                % Beta distribution parameters
                a = q_0_0 * kappa;
                b = (1 - q_0_0) * kappa;
                if any(a <= 0 | b <= 0)
                    disp('Invalid a or b detected:');
                    disp([a(:), b(:)]);
                end

                % Log-likelihood for this trial
                p = betapdf(mu_hat(n), a, b);
                nll_trial(n) = log(p);
            end

            % Sum negative log-likelihood over trials
            nll = -nansum(nll_trial,"all");
        end

        function nll = nll_basicRL_integrated(params, mu_hat, blocks, rewards, condiff)
            % Basic RL model with integration over observations (state uncertainty)
            % This creates state confusion on some trials, unlike the original version
            % that had access to the true state

            alpha = params(1);
            kappa = params(2);
            sigma = params(3);

            nll_trial = NaN(length(mu_hat),1);
            eps = 1e-9;
            mu_hat(mu_hat <= 0) = eps;
            mu_hat(mu_hat >= 1) = 1 - eps;

            q_0_0 = 0.5;

            % Discretize observation space (like obj.set_o)
            set_o = linspace(-0.1, 0.1, 20); % adjust resolution if needed

            for n = 1:length(mu_hat)-1
                % Reset Q at block transitions
                if blocks(n+1) - blocks(n) == 1
                    q_0_0 = 0.5;
                end

                % Observation likelihood under current internal estimate
                p_o_given_u = normpdf(set_o, condiff(n), sigma);
                p_o_given_u = p_o_given_u / sum(p_o_given_u); % normalize

                % Initialize matrix to hold hypothetical Q-values
                voi_matrix = NaN(length(set_o), 2);

                for i = 1:length(set_o)
                    % Belief update based on simulated observation

                    % MY ORIGINAL METHOD
                    % belief_state = normcdf(set_o(i), 0, sigma);
                    % pi_0 = 1 - belief_state;
                    % pi_1 = belief_state;

                    % HOW ITS DONE IN RASMUS'S PREPRINT
                    u = normcdf(0, set_o(i), sigma);     % P(X <= 0)
                    v = normcdf(-0.1, set_o(i), sigma); % P(X <= -kappa)
                    w = normcdf(0.1, set_o(i), sigma);  % P(X <= kappa)
                    normalization_constant = w - v;
                    pi_0 = (u - v) / normalization_constant;
                    pi_1 = (w - u) / normalization_constant;

                    % Simulate Q-value update under this hypothetical observation
                    q_sim = q_0_0;

                    % Basic RL update (without learning rate modulation like sigmaRL)
                    if pi_0 >= pi_1
                        q_sim = q_sim + alpha * (rewards(n) - q_sim);
                    else
                        q_sim = q_sim + alpha * ((1 - rewards(n)) - q_sim);
                    end

                    voi_matrix(i, :) = [q_sim, 1 - q_sim];
                end

                % Integrate over all possible observations
                q_0_0 = sum(voi_matrix(:,1) .* p_o_given_u');
                % q_transformed = anti_sigmoidPG(q_0_0,beta);
                q_transformed = q_0_0;

                % Beta parameters for likelihood
                a = q_transformed * kappa;
                b = (1 - q_transformed) * kappa;
                % q_0_0 = q_transformed;

                % Avoid invalid beta params
                if a <= 0 || b <= 0
                    disp("Invalid beta params");
                    a = max(eps, a);
                    b = max(eps, b);
                end

                % Log-likelihood
                p = betapdf(mu_hat(n), a, b);
                nll_trial(n) = log(p + eps);
            end

            nll = -nansum(nll_trial,"all");
        end


        function nll = nll_basicRL_integrated_fast(params, mu_hat, blocks, rewards, condiff)
            % Optimized version of nll_basicRL_integrated
            alpha = params(1);
            kappa = params(2);
            sigma = params(3);

            eps_val = 1e-9;
            mu_hat(mu_hat <= 0) = eps_val;
            mu_hat(mu_hat >= 1) = 1 - eps_val;

            nTrials = length(mu_hat);
            nll_trial = zeros(nTrials-1,1);

            q_0_0 = 0.5;

            % Discretize observation space
            set_o = linspace(-0.1, 0.1, 20);  % can reduce to 10-15 for speed

            % Precompute belief states for all set_o values
            belief_grid = normcdf(set_o, 0, sigma);
            pi_0_grid = 1 - belief_grid;
            pi_1_grid = belief_grid;

            for n = 1:nTrials-1
                % Reset Q at block transitions
                if blocks(n+1) - blocks(n) == 1
                    q_0_0 = 0.5;
                end

                % Observation likelihood vectorized
                p_o_given_u = normpdf(set_o, condiff(n), sigma);
                p_o_given_u = p_o_given_u / sum(p_o_given_u);

                % Vectorized Q-value updates for all set_o
                q_sim = q_0_0 + alpha * ((pi_1_grid >= pi_0_grid) .* (rewards(n) - q_0_0) + ...
                    (pi_0_grid > pi_1_grid) .* ((1 - rewards(n)) - q_0_0));

                % Integrate over all observations
                q_0_0 = sum(q_sim .* p_o_given_u);

                % Beta parameters for likelihood
                a = q_0_0 * kappa;
                b = (1 - q_0_0) * kappa;
                if a <= 0 || b <= 0
                    a = max(eps_val, a);
                    b = max(eps_val, b);
                end

                % Log-likelihood
                p = betapdf(mu_hat(n), a, b);
                nll_trial(n) = log(p + eps_val);
            end

            nll = -sum(nll_trial);
        end



        function nll = nll_mixture_integrated(params, mu_hat, blocks, rewards, condiff)
            % Basic RL model with integration over observations (state uncertainty)
            % This creates state confusion on some trials, unlike the original version
            % that had access to the true state

            alpha = params(1);
            kappa = params(2);
            sigma = params(3);
            lambda = params(4);

            nll_trial = NaN(length(mu_hat),1);
            eps = 1e-9;
            mu_hat(mu_hat <= 0) = eps;
            mu_hat(mu_hat >= 1) = 1 - eps;

            q_0_0 = 0.5;

            % Discretize observation space (like obj.set_o)
            set_o = linspace(-0.1, 0.1, 20); % adjust resolution if needed

            for n = 1:length(mu_hat)-1
                % Reset Q at block transitions
                if blocks(n+1) - blocks(n) == 1
                    q_0_0 = 0.5;
                end

                % Observation likelihood under current internal estimate (computed once)
                p_o_given_u = normpdf(set_o, condiff(n), sigma);
                p_o_given_u = p_o_given_u / sum(p_o_given_u); % normalize

                % Initialize matrices to hold hypothetical Q-values for both models
                voi_matrix_basicRL = NaN(length(set_o), 2);
                voi_matrix_RLSigma = NaN(length(set_o), 2);

                for i = 1:length(set_o)
                    % Belief update based on simulated observation (computed once)
                    belief_state = normcdf(set_o(i), 0, sigma);
                    pi_0 = 1 - belief_state;
                    pi_1 = belief_state;

                    % Simulate Q-value update for Basic RL model
                    q_sim_basic = q_0_0;
                    if pi_0 >= pi_1
                        q_sim_basic = q_sim_basic + alpha * (rewards(n) - q_sim_basic);
                    else
                        q_sim_basic = q_sim_basic + alpha * ((1 - rewards(n)) - q_sim_basic);
                    end
                    voi_matrix_basicRL(i, :) = [q_sim_basic, 1 - q_sim_basic];

                    % Simulate Q-value update for RL-Sigma model
                    q_sim_sigma = q_0_0;
                    if pi_0 >= pi_1
                        q_sim_sigma = q_sim_sigma + pi_0 * alpha * (rewards(n) - q_sim_sigma);
                    else
                        q_sim_sigma = q_sim_sigma + pi_1 * alpha * ((1 - rewards(n)) - q_sim_sigma);
                    end
                    voi_matrix_RLSigma(i, :) = [q_sim_sigma, 1 - q_sim_sigma];
                end

                % Integrate over all possible observations for both models
                q_basicRL = sum(voi_matrix_basicRL(:,1) .* p_o_given_u');
                q_RLSigma = sum(voi_matrix_RLSigma(:,1) .* p_o_given_u');

                % Mixture of the two models
                q_transformed = q_RLSigma * lambda + q_basicRL * (1-lambda);

                % Beta parameters for likelihood
                a = q_transformed * kappa;
                b = (1 - q_transformed) * kappa;

                % Avoid invalid beta params
                if a <= 0 || b <= 0
                    disp("Invalid beta params");
                    a = max(eps, a);
                    b = max(eps, b);
                end

                % Log-likelihood
                p = betapdf(mu_hat(n), a, b);
                nll_trial(n) = log(p + eps);
            end

            nll = -nansum(nll_trial,"all");
        end
        %% ===================================================================
        %  RL SIGMA MODEL
        %  -------------------------------------------------------------------
        %  Q-learning with perceptual uncertainty (sigma) affecting belief state.
        %  Inputs:
        %    params   - [alpha, kappa, sigma]
        %    mu_hat   - observed values (vector)
        %    blocks   - block indices (vector)
        %    rewards  - reward outcomes (vector)
        %    condiff  - contrast difference (vector)
        % ====================================================================

        function nll = nll_RLsigma_VOI(params, mu_hat, blocks, rewards, condiff)

            alpha = params(1);
            kappa = params(2);
            sigma = params(3);
            % beta = params(4);
            nll_trial = NaN(length(mu_hat),1);
            eps = 1e-6;
            mu_hat(mu_hat <= 0) = eps;
            mu_hat(mu_hat >= 1) = 1 - eps;

            q_0_0 = 0.5;

            % Discretize observation space (like obj.set_o)
            set_o = linspace(-0.1, 0.1, 20);  % adjust resolution if needed

            for n = 1:length(mu_hat)-1

                % Reset Q at block transitions
                if blocks(n+1) - blocks(n) == 1
                    q_0_0 = 0.5;
                end

                % Observation likelihood under current internal estimate
                p_o_given_u = normpdf(set_o, condiff(n), sigma);
                p_o_given_u = p_o_given_u / sum(p_o_given_u); % normalize


                % Initialize matrix to hold hypothetical Q-values
                voi_matrix = NaN(length(set_o), 2);

                for i = 1:length(set_o)
                    % Belief update based on simulated observation
                    % MY ORIGINAL METHOD
                    % belief_state = normcdf(set_o(i), 0, sigma);
                    % pi_0 = 1 - belief_state;
                    % pi_1 = belief_state;

                    % HOW ITS DONE IN RASMUS'S PREPRINT
                    u = normcdf(0, set_o(i), sigma);     % P(X <= 0)
                    v = normcdf(-0.1, set_o(i), sigma); % P(X <= -kappa)
                    w = normcdf(0.1, set_o(i), sigma);  % P(X <= kappa)
                    normalization_constant = w - v;
                    pi_0 = (u - v) / normalization_constant;
                    pi_1 = (w - u) / normalization_constant;

                  

                    % Simulate Q-value update under this hypothetical observation
                    q_sim = q_0_0;
                    if pi_0 >= pi_1
                        q_sim = q_sim + pi_0 * alpha * (rewards(n) - q_sim);
                    else
                        q_sim = q_sim + pi_1 * alpha * ((1 - rewards(n)) - q_sim);
                    end
                    voi_matrix(i, :) = [q_sim, 1 - q_sim];
                end

                % Integrate over all possible observations
                q_0_0 = sum(voi_matrix(:,1) .* p_o_given_u');
                
                if any(voi_matrix(:,1) < 0)
                    fprintf('Negative Q-values detected!');
                    fprintf('Min Q-value: %.6f\n', min(voi_matrix(:,1)));
                end
                if q_0_0 < 0
                    disp("Invalid q_0_0 params");
                end
                q_transformed = q_0_0;
                a = q_transformed * kappa;
                b = (1 - q_transformed) * kappa;

                % Avoid invalid beta params
                if a <= 0 || b <= 0
                    disp("Invalid beta params");
                    a = max(eps, a);
                    b = max(eps, b);
                end

                p = betapdf(mu_hat(n), a, b);
                nll_trial(n) = log(p + eps);
            end

            nll = -nansum(nll_trial,"all");
        end

        %% ===================================================================
        %  RL SIGMA MODEL
        %  -------------------------------------------------------------------
        %  Q-learning with perceptual uncertainty (sigma) affecting belief state.
        %  Inputs:
        %    params   - [alpha, kappa, sigma]
        %    mu_hat   - observed values (vector)
        %    blocks   - block indices (vector)
        %    rewards  - reward outcomes (vector)
        %    condiff  - contrast difference (vector)
        % ====================================================================

        function nll = nll_RLsigma_confirmBias_VOI(params, mu_hat, blocks, rewards, condiff, confirmRew)

            % alpha = params(1);
            kappa = params(1);
            sigma = params(2);
            confirmBias = params(3);
            noconfirmBias = params(4);
            % beta = params(5);

            nll_trial = NaN(length(mu_hat),1);
            eps = 1e-6;
            mu_hat(mu_hat <= 0) = eps;
            mu_hat(mu_hat >= 1) = 1 - eps;

            q_0_0 = 0.5;

            % Discretize observation space (like obj.set_o)
            set_o = linspace(-0.1, 0.1, 20);  % adjust resolution if needed

            for n = 1:length(mu_hat)-1

                % Reset Q at block transitions
                if blocks(n+1) - blocks(n) == 1
                    q_0_0 = 0.5;
                end

                % Observation likelihood under current internal estimate
                p_o_given_u = normpdf(set_o, condiff(n), sigma);
                p_o_given_u = p_o_given_u / sum(p_o_given_u); % normalize

                % Initialize matrix to hold hypothetical Q-values
                voi_matrix = NaN(length(set_o), 2);

                for i = 1:length(set_o)

                    % MY ORIGINAL METHOD
                    % belief_state = normcdf(set_o(i), 0, sigma);
                    % pi_0 = 1 - belief_state;
                    % pi_1 = belief_state;

                    % HOW ITS DONE IN RASMUS'S PREPRINT
                    u = normcdf(0, set_o(i), sigma);     % P(X <= 0)
                    v = normcdf(-0.1, set_o(i), sigma); % P(X <= -kappa)
                    w = normcdf(0.1, set_o(i), sigma);  % P(X <= kappa)
                    normalization_constant = w - v;
                    pi_0 = (u - v) / normalization_constant;
                    pi_1 = (w - u) / normalization_constant;

                    % Simulate Q-value update under this hypothetical observation
                    q_sim = q_0_0;
                    if pi_0 >= pi_1
                        if confirmRew(n) == 1
                            q_sim = q_sim + pi_0 * confirmBias * (rewards(n) - q_sim); % + confirmBias * (rewards(n) - q_sim);
                        else
                            q_sim = q_sim + pi_0 * noconfirmBias * (rewards(n) - q_sim); % + noconfirmBias * (rewards(n) - q_sim);
                        end
                    else
                        if confirmRew(n) == 1
                            q_sim = q_sim + pi_1 * confirmBias * ((1 - rewards(n)) - q_sim);% + confirmBias  * ((1 - rewards(n)) - q_sim);
                        else
                            q_sim = q_sim + pi_1 * noconfirmBias * ((1 - rewards(n)) - q_sim);% + noconfirmBias  * ((1 - rewards(n)) - q_sim);
                        end
                    end
                    voi_matrix(i, :) = [q_sim, 1 - q_sim];
                end

                % Integrate over all possible observations
                q_0_0 = sum(voi_matrix(:,1) .* p_o_given_u');
                % Add before the q_0_0 calculation:
                if any(voi_matrix(:,1) < 0)
                    fprintf('Negative Q-values detected! alpha=%.3f, confirmBias=%.3f, noconfirmBias=%.3f\n', ...
                        confirmBias, noconfirmBias);
                    fprintf('Min Q-value: %.6f\n', min(voi_matrix(:,1)));
                end
                if q_0_0 < 0
                    disp("Invalid q_0_0 params");
                end
                q_transformed = q_0_0;
                % q_transformed = anti_sigmoidPG(q_0_0,beta);
                % q_transformed = real(q_transformed);  % Force real
                % q_transformed = max(eps, min(1-eps, q_transformed));  % Ensure valid probability
                %
                % a = real(q_transformed * kappa);
                % b = real((1 - q_transformed) * kappa);
                % a = max(eps, a);
                % b = max(eps, b);
                % Beta parameters for likelihood
                a = q_transformed * kappa;
                b = (1 - q_transformed) * kappa;
                % q_0_0 = q_transformed;

                % Avoid invalid beta params
                if a <= 0 || b <= 0
                    disp("Invalid beta params");
                    a = max(eps, a);
                    b = max(eps, b);
                end
                % Add this right before line 312 in your nll_RLsigma_confirmBias_VOI function
                % Log-likelihood
                p = betapdf(mu_hat(n), a, b);
                nll_trial(n) = log(p + eps);
            end

            nll = -nansum(nll_trial,"all");
        end

        function nll = nll_basicRL_confirmBias_VOI(params, mu_hat, blocks, rewards, condiff, confirmRew)

            % alpha = params(1);
            kappa = params(1);
            sigma = params(2);
            confirmBias = params(3);
            noconfirmBias = params(4);
            % beta = params(5);

            nll_trial = NaN(length(mu_hat),1);
            eps = 1e-6;
            mu_hat(mu_hat <= 0) = eps;
            mu_hat(mu_hat >= 1) = 1 - eps;

            q_0_0 = 0.5;

            % Discretize observation space (like obj.set_o)
            set_o = linspace(-0.1, 0.1, 20);  % adjust resolution if needed

            for n = 1:length(mu_hat)-1

                % Reset Q at block transitions
                if blocks(n+1) - blocks(n) == 1
                    q_0_0 = 0.5;
                end

                % Observation likelihood under current internal estimate
                p_o_given_u = normpdf(set_o, condiff(n), sigma);
                p_o_given_u = p_o_given_u / sum(p_o_given_u); % normalize

                % Initialize matrix to hold hypothetical Q-values
                voi_matrix = NaN(length(set_o), 2);

                for i = 1:length(set_o)

                    % MY ORIGINAL METHOD
                    % belief_state = normcdf(set_o(i), 0, sigma);
                    % pi_0 = 1 - belief_state;
                    % pi_1 = belief_state;

                    % HOW ITS DONE IN RASMUS'S PREPRINT
                    u = normcdf(0, set_o(i), sigma);     % P(X <= 0)
                    v = normcdf(-0.1, set_o(i), sigma); % P(X <= -kappa)
                    w = normcdf(0.1, set_o(i), sigma);  % P(X <= kappa)
                    normalization_constant = w - v;
                    pi_0 = (u - v) / normalization_constant;
                    pi_1 = (w - u) / normalization_constant;
                    
                    % Simulate Q-value update under this hypothetical observation
                    q_sim = q_0_0;
                    if pi_0 >= pi_1
                        if confirmRew(n) == 1
                            q_sim = q_sim + confirmBias * (rewards(n) - q_sim); % + confirmBias * (rewards(n) - q_sim);
                        else
                            q_sim = q_sim + noconfirmBias * (rewards(n) - q_sim); % + noconfirmBias * (rewards(n) - q_sim);
                        end
                    else
                        if confirmRew(n) == 1
                            q_sim = q_sim + confirmBias * ((1 - rewards(n)) - q_sim);% + confirmBias  * ((1 - rewards(n)) - q_sim);
                        else
                            q_sim = q_sim + noconfirmBias * ((1 - rewards(n)) - q_sim);% + noconfirmBias  * ((1 - rewards(n)) - q_sim);
                        end
                    end
                    voi_matrix(i, :) = [q_sim, 1 - q_sim];
                end

                % Integrate over all possible observations
                q_0_0 = sum(voi_matrix(:,1) .* p_o_given_u');
                % Add before the q_0_0 calculation:
                if any(voi_matrix(:,1) < 0)
                    fprintf('Negative Q-values detected! alpha=%.3f, confirmBias=%.3f, noconfirmBias=%.3f\n', ...
                        confirmBias, noconfirmBias);
                    fprintf('Min Q-value: %.6f\n', min(voi_matrix(:,1)));
                end
                if q_0_0 < 0
                    disp("Invalid q_0_0 params");
                end
                q_transformed = q_0_0;
                a = q_transformed * kappa;
                b = (1 - q_transformed) * kappa;

                % Avoid invalid beta params
                if a <= 0 || b <= 0
                    disp("Invalid beta params");
                    a = max(eps, a);
                    b = max(eps, b);
                end

                % Log-likelihood
                p = betapdf(mu_hat(n), a, b);
                nll_trial(n) = log(p + eps);
            end

            nll = -nansum(nll_trial,"all");
        end

        function nll = nll_RLsigma_confirmBias(params, mu_hat, blocks, rewards, condiff, confirmRew)

            kappa = params(1);
            sigma = params(2);
            confirmBias = params(3);
            noconfirmBias = params(4);
            % beta = params(5);

            nll_trial = NaN(length(mu_hat),1);
            eps = 1e-6;
            mu_hat(mu_hat <= 0) = eps;
            mu_hat(mu_hat >= 1) = 1 - eps;

            q_0_0 = 0.5;

            for n = 1:length(mu_hat)-1

                % Reset Q at block transitions
                if blocks(n+1) - blocks(n) == 1
                    q_0_0 = 0.5;
                end

                % Compute belief state from perceptual uncertainty
                belief_state = normcdf(condiff(n), 0, sigma);
                if condiff(n) < 0
                    pi_0 = 1-belief_state;
                    pi_1 = belief_state;
                else
                    pi_1 = belief_state;
                    pi_0 = 1-belief_state;
                end


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

                if q_0_0 < 0
                    disp("Invalid q_0_0 params");
                end
                q_transformed = q_0_0;
                % q_transformed = anti_sigmoidPG(q_0_0,beta);
                % q_transformed = real(q_transformed);  % Force real
                % q_transformed = max(eps, min(1-eps, q_transformed));  % Ensure valid probability
                %
                % a = real(q_transformed * kappa);
                % b = real((1 - q_transformed) * kappa);
                % a = max(eps, a);
                % b = max(eps, b);
                % Beta parameters for likelihood
                a = q_transformed * kappa;
                b = (1 - q_transformed) * kappa;
                % q_0_0 = q_transformed;

                % % Avoid invalid beta params
                % if a <= 0 || b <= 0
                %     disp("Invalid beta params");
                %     a = max(eps, a);
                %     b = max(eps, b);
                % end
                % % Add this right before line 312 in your nll_RLsigma_confirmBias_VOI function
                % % Also check a and b parameters
                % fprintf('a = %f, b = %f\n', a, b);
                % if a <= 0 || b <= 0
                %     error('Beta parameters must be positive!');
                % end

                % Log-likelihood
                p = betapdf(mu_hat(n), a, b);
                nll_trial(n) = log(p + eps);
            end

            nll = -nansum(nll_trial,"all");
        end


        function nll = nll_RLsigma(params, mu_hat, blocks, rewards, condiff)
            alpha = params(1);
            kappa = params(2);
            sigma = params(3);
            nll_trial = NaN(length(mu_hat),1);
            eps = 1e-9;
            mu_hat(mu_hat <= 0) = eps;
            mu_hat(mu_hat >= 1) = 1 - eps;
            q_0_0 = 0.5;

            for n = 1:length(mu_hat)-1

                % Reset Q-value at block transitions
                if blocks(n+1) - blocks(n) == 1
                    q_0_0 = 0.5;
                end

                % Compute belief state from perceptual uncertainty
                belief_state = normcdf(condiff(n), 0, sigma);
                if condiff(n) < 0
                    pi_0 = 1-belief_state;
                    pi_1 = belief_state;
                else
                    pi_1 = belief_state;
                    pi_0 = 1-belief_state;
                end

                % Q-value update weighted by belief
                if pi_0 >= pi_1
                    q_0_0 = q_0_0 + pi_0 * alpha * (rewards(n) - q_0_0);
                else
                    q_0_0 = q_0_0 + pi_1 * alpha * ((1 - rewards(n)) - q_0_0);
                end

                % Beta distribution parameters
                a = q_0_0 * kappa;
                b = (1 - q_0_0) * kappa;
                if any(a <= 0 | b <= 0)
                    disp('Invalid a or b detected:');
                    disp([a(:), b(:)]);
                end

                % Log-likelihood for this trial
                p = betapdf(mu_hat(n), a, b);
                nll_trial(n) = log(p);
            end
            nll = -nansum(nll_trial,"all");
        end

        %% ===================================================================
        %  POSTERIOR-WEIGHTED RL MODEL
        %  -------------------------------------------------------------------
        %  RL model with power-weighted state inference and perceptual uncertainty.
        %  Inputs:
        %    params          - [alpha, kappa, sigma]
        %    mu_hat          - observed values (vector)
        %    blocks          - block indices (vector)
        %    rewards         - reward outcomes (vector)
        %    condiff         - contrast difference (vector)
        %    choices         - action choices (vector)
        %    recoded_rewards - rewards recoded for chosen action (vector)
        % ====================================================================
        function nll = nll_PWRL(params, mu_hat, blocks, rewards, condiff, choices, recoded_rewards, contrast)
            alpha = params(1);
            kappa = params(2);
            sigma = params(3);
            nll_trial = NaN(length(mu_hat),1);
            eps = 1e-9;
            mu_hat(mu_hat <= 0) = eps;
            mu_hat(mu_hat >= 1) = 1 - eps;
            q_0_0 = 0.5;
            q_1_0 = 1-q_0_0;
            q_1_1 = q_0_0;
            q_0_1 = 1 - q_0_0;

            for n = 1:length(mu_hat)-1

                % Reset Q-values at block transitions
                if blocks(n+1) - blocks(n) == 1
                    q_0_0 = 0.5;
                    q_1_0 = 1-q_0_0;
                    q_1_1 = q_0_0;
                    q_0_1 = 1 - q_0_0;
                end

                % Compute belief state from perceptual uncertainty
                belief_state = normcdf(condiff(n), 0, sigma);
                if condiff(n) < 0
                    pi_0 = 1-belief_state;
                    pi_1 = belief_state;
                else
                    pi_1 = belief_state;
                    pi_0 = 1-belief_state;
                end
                % s = 1, a = 1, r = 0
                % s = 1, a = 0, r = 1
                % Posterior-weighted state inference
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

                % here rewards mean correct i.e. reward(correct) i.e.
                % task-generated reward that the participant has seen
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
                        q_0_0 = q_0_0 + pw_pi * alpha * ((1-recoded_rewards(n)) - q_0_0);
                    end
                else
                    q_0_0 = q_0_0 + pw_pi * alpha * (recoded_rewards(n) - q_0_0);
                end
                q_1_0 = 1-q_0_0;
                q_1_1 = q_0_0;
                q_0_1 = 1 - q_0_0;

                % Beta distribution parameters
                a = q_0_0 * kappa;
                b = (1 - q_0_0) * kappa;
                if any(a <= 0 | b <= 0)
                    disp('Invalid a or b detected:');
                    disp([a(:), b(:)]);
                end

                % Log-likelihood for this trial
                p = betapdf(mu_hat(n), a, b);
                nll_trial(n) = log(p);
            end
            nll = -nansum(nll_trial,"all");
        end

        %% ===================================================================
        %  BAYESIAN AGENT MODEL
        %  -------------------------------------------------------------------
        %  Bayesian agent model using an external Agent class.
        %  Inputs:
        %    params   - [kappa, sigma]
        %    mu_hat   - observed values (vector)
        %    data     - table with trial data (must include blocks, choice, condiff_relative)
        %    nBlocks  - number of blocks (scalar)
        %    nTrials  - number of trials per block (scalar)
        %    blocks   - block indices (vector)
        %    rewards  - reward outcomes (vector)
        % ====================================================================
        function nll = nll_bayesianAgent(params, mu_hat, data, nBlocks, nTrials, blocks, rewards)
            kappa = params(1);
            sigma = params(2);
            % beta = params(3);
            nll_trial = NaN(nTrials,nBlocks);
            eps = 1e-9;
            mu_hat(mu_hat <= 0) = eps;
            mu_hat(mu_hat >= 1) = 1 - eps;
            mu = [];
            uniqueBlocks = unique(blocks);
            rewards = rewards;

            for bl = 1:nBlocks
                agent = Agent();
                agent.task_agent_analysis = 1;
                agent.confirmation_bias = 0;
                agent.sigma = sigma;
                dataBlocks = data(data.blocks == uniqueBlocks(bl),:);
                choices = dataBlocks.choice;
                condiff = dataBlocks.condiff_relative;
                rewardsBlocks = rewards(data.blocks == uniqueBlocks(bl),:);
                mu_hatBlocks = mu_hat(data.blocks == uniqueBlocks(bl),:);
                for t = 1:height(dataBlocks)

                    % Bayesian agent inference and learning steps
                    agent.o_t = condiff(t);
                    agent.p_s_giv_o(agent.o_t);
                    agent.compute_valence();
                    agent.softmax();
                    agent.a_t = choices(t);
                    agent.learn(rewardsBlocks(t));

                    % Beta distribution parameters
                    % q_transformed = anti_sigmoidPG(agent.G,beta);
                    q_transformed = agent.G;
                    a = q_transformed * kappa;
                    b = (1 - q_transformed) * kappa;
                    if any(a <= 0 | b <= 0)
                        disp('Invalid a or b detected:');
                        disp([a(:), b(:)]);
                    end

                    % Log-likelihood for this trial
                    p = betapdf(mu_hatBlocks(t), a, b);
                    nll_trial(t,bl) = log(p);
                    mu = [mu;agent.G];
                end
            end
            nll = -nansum(nll_trial,"all");
        end

        function nll = nll_bayesianAgent_confirmBias(params, mu_hat, data, nBlocks, nTrials, blocks, rewards)
            kappa = params(1);
            sigma = params(2);
            confirmBias = params(3);
            % beta = params(4);
            nll_trial = NaN(nTrials,nBlocks);
            eps = 1e-9;
            mu_hat(mu_hat <= 0) = eps;
            mu_hat(mu_hat >= 1) = 1 - eps;
            mu = [];
            uniqueBlocks = unique(blocks);
            rewards = rewards;

            for bl = 1:nBlocks
                agent = Agent();
                agent.sigma = sigma;
                agent.task_agent_analysis = 1;
                agent.confirmBias = confirmBias;
                dataBlocks = data(data.blocks == uniqueBlocks(bl),:);
                choices = dataBlocks.choice;
                condiff = dataBlocks.condiff_relative;
                rewardsBlocks = rewards(data.blocks == uniqueBlocks(bl),:);
                mu_hatBlocks = mu_hat(data.blocks == uniqueBlocks(bl),:);
                for t = 1:height(dataBlocks)
                    agent.confirmation_bias = 0;
                    agent.sigma = sigma;
                    % Bayesian agent inference and learning steps
                    agent.o_t = condiff(t);
                    agent.observation_sample(agent.o_t)
                    agent.p_s_giv_o(agent.o_t);
                    % agent.compute_valence();
                    % agent.softmax();
                    agent.decide_e(agent.o_t);
                    agent.a_t = choices(t);
                    agent.confirmation_bias = dataBlocks.confirm_rew(t);
                    agent.learn(rewardsBlocks(t));

                    % Beta distribution parameters
                    % q_transformed = anti_sigmoidPG(agent.G,beta);
                    q_transformed = agent.G;
                    a = q_transformed * kappa;
                    b = (1 - q_transformed) * kappa;
                    if any(a <= 0 | b <= 0)
                        disp('Invalid a or b detected:');
                        disp([a(:), b(:)]);
                    end

                    % Log-likelihood for this trial
                    p = betapdf(mu_hatBlocks(t), a, b);
                    nll_trial(t,bl) = log(p);
                    mu = [mu;agent.G];
                end
            end
            nll = -nansum(nll_trial,"all");
        end


        %% ===================================================================
        %  AIC/BIC COMPUTATION
        %  -------------------------------------------------------------------
        %  Compute AIC and BIC given negative log-likelihood, parameter count, and trial count.
        %  Inputs:
        %    nll        - negative log-likelihood (scalar or vector)
        %    num_params - number of free parameters (scalar or vector)
        %    num_trials - number of data points (scalar)
        %  Outputs:
        %    AIC        - Akaike Information Criterion
        %    BIC        - Bayesian Information Criterion
        % ====================================================================
        function [AIC, BIC] = compute_aic_bic(nll, num_params, num_trials)
            % Compute log-likelihood
            logL = -nll;
            % Compute AIC and BIC
            AIC = 2 * num_params - 2 * logL;
            BIC = log(num_trials) * num_params - 2 * logL;
        end

        %% ===================================================================
        %  REWARD RECODING
        %  -------------------------------------------------------------------
        %  Recode trial rewards into the model's fixed state-0 reference frame.
        %  recoded_rewards is coded relative to stimulus identity (e.g. left/
        %  right), but the RL/Bayesian models track a single Q-value / belief
        %  in a fixed reference frame. On zero-contrast trials there is no
        %  left/right asymmetry to correct for, so the reward is kept as-is;
        %  on non-zero-contrast trials it is flipped (1 - reward) to stay
        %  consistent with that fixed frame.
        %  Inputs:
        %    recoded_rewards - reward outcomes recoded relative to stimulus
        %                      identity (vector)
        %    contrast        - contrast value per trial; 0 marks a
        %                      zero-contrast trial (vector)
        %  Outputs:
        %    rewards - reward outcomes recoded into the state-0 reference
        %              frame (vector)
        % ====================================================================
        function rewards = recode_rewards(recoded_rewards, contrast)
            rewards = recoded_rewards .* (contrast == 0) + ...
                (1 - recoded_rewards) .* (contrast ~= 0);
        end

        %% ===================================================================
        %  LOAD FITTING DATA
        %  -------------------------------------------------------------------
        %  Load the preprocessed slider-fitting dataset and split it into the
        %  "both" (mixed reward + perceptual) and perceptual-only condition
        %  subsets used by the fitting scripts.
        %  Outputs:
        %    data           - both-condition trials (condition == 1), with
        %                      reward-condition trials (choice_cond == 3)
        %                      removed and condiff_relative precomputed
        %    dataPerceptual - perceptual-condition trials (condition == 2)
        %    uniqueID       - unique subject IDs (vector)
        %    numSubjs       - number of subjects (scalar)
        % ====================================================================
        function [data, dataPerceptual, uniqueID, numSubjs] = load_fitting_data()
            data = importdata("preprocessed_dataFitting.mat");
            uniqueID = unique(data.ID);
            data = data(data.choice_cond ~= 3,:);
            numSubjs = length(uniqueID);
            % Precompute contrast difference
            data.condiff_relative = (data.contrast_left - data.contrast_right) ./ 2;
            dataPerceptual = data(data.condition ~= 1,:);
            data(data.condition == 2,:) = [];
        end

    end % methods
end % classdef
