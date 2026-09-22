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

            for n = 1:length(mu_hat)

                % Reset Q-value at block transitions
                if n == 1 || blocks(n) ~= blocks(n-1)
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

            % Belief state (pi_0/pi_1) at each hypothetical observation is
            % computed via Agent.p_s_giv_o -- the same formula/bounds as
            % agentvars.m's set_o/kappa_max defaults used here -- instead
            % of duplicating it inline. The RL update rule below stays
            % specific to this model.
            agent = Agent();
            agent.sigma = sigma;

            for n = 1:length(mu_hat)
                % Reset Q at block transitions
                if n == 1 || blocks(n) ~= blocks(n-1)
                    q_0_0 = 0.5;
                end

                % Observation likelihood under current internal estimate
                p_o_given_u = fitSlider_ALLmodels.observation_weights(set_o, condiff(n), sigma);

                % Initialize matrix to hold hypothetical Q-values
                voi_matrix = NaN(length(set_o), 2);

                for i = 1:length(set_o)
                    % Belief state at this hypothetical observation
                    agent.p_s_giv_o(set_o(i));
                    pi_0 = agent.pi_0;
                    pi_1 = agent.pi_1;

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

                % Avoid invalid beta params
                if a <= 0 || b <= 0
                    % just here for debugging purposes. will remove it at a
                    % later stage
                    fprintf('Invalid beta params at trial %d: q=%.3g, kappa=%.3g, a=%.3g, b=%.3g (alpha=%.3g, sigma=%.3g)\n', ...
                        n, q_transformed, kappa, a, b, alpha, sigma);
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
        %  BASIC RL MODEL - GENERATIVE SIMULATION (WITH INTEGRATION)
        %  -------------------------------------------------------------------
        %  Simulates synthetic mu_hat slider responses from the basic RL
        %  model, using the same observation-noise integration as
        %  nll_basicRL_integrated (rather than sampling a single noisy
        %  observation per trial), so that data generated here matches what
        %  that likelihood function actually assumes -- for a parameter
        %  recovery study, fit the result back with nll_basicRL_integrated.
        %  Inputs:
        %    params  - [alpha, kappa, sigma]
        %    blocks  - block index per trial (vector)
        %    rewards - state-0-referenced reward outcomes (vector)
        %    condiff - contrast difference per trial (vector)
        %  Outputs:
        %    mu_hat    - simulated slider responses (vector)
        %    q_0_0_trace - underlying integrated belief per trial, before
        %                  Beta sampling (vector); useful for diagnosing
        %                  whether the belief update itself looks sensible,
        %                  separate from Beta-sampling noise in mu_hat
        % ====================================================================
        function [mu_hat, q_0_0_trace] = simulate_basicRL_integrated(params, blocks, rewards, condiff)
            alpha = params(1);
            kappa = params(2);
            sigma = params(3);

            n_trials = length(condiff);
            mu_hat = NaN(n_trials, 1);
            q_0_0_trace = NaN(n_trials, 1);
            q_0_0 = 0.5;

            set_o = linspace(-0.1, 0.1, 20);

            agent = Agent();
            agent.sigma = sigma;

            for n = 1:n_trials
                % Reset belief at the start of a new block
                if n == 1 || blocks(n) ~= blocks(n-1)
                    q_0_0 = 0.5;
                end

                % Integrate the hypothetical Q-value update over the agent's
                % observation-noise distribution around condiff(n) -- same
                % mechanism as nll_basicRL_integrated.
                p_o_given_u = fitSlider_ALLmodels.observation_weights(set_o, condiff(n), sigma);
                voi_matrix = NaN(length(set_o), 2);
                for i = 1:length(set_o)
                    agent.p_s_giv_o(set_o(i));
                    pi_0 = agent.pi_0;
                    pi_1 = agent.pi_1;

                    q_sim = q_0_0;
                    if pi_0 >= pi_1
                        q_sim = q_sim + alpha * (rewards(n) - q_sim);
                    else
                        q_sim = q_sim + alpha * ((1 - rewards(n)) - q_sim);
                    end
                    voi_matrix(i, :) = [q_sim, 1 - q_sim];
                end
                q_0_0 = sum(voi_matrix(:,1) .* p_o_given_u');
                q_0_0_trace(n) = q_0_0;

                % Simulated slider response: Beta-distributed around the
                % belief, with concentration kappa (mirrors betapdf(mu_hat,
                % a, b) in nll_basicRL_integrated's likelihood).
                a = max(q_0_0 * kappa, 1e-9);
                b = max((1 - q_0_0) * kappa, 1e-9);
                mu_hat(n) = betarnd(a, b);
            end
        end

        %% ===================================================================
        %  BASIC RL MODEL - GENERATIVE SIMULATION WITH ECONOMIC CHOICE
        %  -------------------------------------------------------------------
        %  Like simulate_basicRL_integrated, but also generates the economic
        %  choice and reward each trial, instead of taking reward as given.
        %  Choice is generated from integrated action values -- combining
        %  this trial's integrated perceptual belief (pi_0/pi_1, marginalized
        %  over the same set_o observation-noise grid used everywhere else)
        %  with the CURRENT tracked value belief q_0_0, via the same formula
        %  Agent.compute_valence() uses (v_a_0 = (pi_0-pi_1)*q_0_0 + pi_1) --
        %  then a softmax with a fixed beta (not one of basicRL's own
        %  parameters; sigma/alpha/kappa are the only things being tested for
        %  recovery, so beta is fixed here rather than estimated). Reward is
        %  then drawn from a state-action contingency (reward the choice with
        %  probability p_reward_correct if it matches the true state, else
        %  1-p_reward_correct), and fed as-is into the same integrated
        %  delta-rule belief update simulate_basicRL_integrated uses -- its
        %  own pi_0>=pi_1 branch already does the perceptual-belief-based
        %  recoding, so no separate true-state recoding step is needed.
        %  Choice generation and belief-update both integrate over the same
        %  set_o grid (option A from the design discussion), for consistency
        %  with what's assumed elsewhere in this trial: an agent doesn't have
        %  privileged access to one single "true" observation when computing
        %  its response, it always marginalizes over its own perceptual
        %  uncertainty.
        %  Inputs:
        %    params            - [alpha, kappa, sigma]
        %    blocks            - block index per trial (vector)
        %    state             - ground-truth state per trial, 0 or 1 (vector)
        %    condiff           - contrast difference per trial (vector)
        %    beta              - fixed softmax inverse-temperature for choice
        %                        (not estimated; higher = more deterministic)
        %    p_reward_correct  - P(reward=1 | choice matches state)
        %  Outputs:
        %    mu_hat      - simulated slider responses (vector)
        %    choice      - simulated economic choice, 0 or 1 (vector)
        %    reward      - simulated reward outcome, 0 or 1 (vector)
        %    q_0_0_trace - underlying integrated value belief per trial (vector)
        % ====================================================================
        function [mu_hat, choice, reward, q_0_0_trace] = simulate_basicRL_integrated_choice(params, blocks, state, condiff, beta, p_reward_correct)
            alpha = params(1);
            kappa = params(2);
            sigma = params(3);

            n_trials = length(condiff);
            mu_hat = NaN(n_trials, 1);
            choice = NaN(n_trials, 1);
            reward = NaN(n_trials, 1);
            q_0_0_trace = NaN(n_trials, 1);
            q_0_0 = 0.5;

            set_o = linspace(-0.1, 0.1, 20);
            n_o = length(set_o);

            agent = Agent();
            agent.sigma = sigma;

            for n = 1:n_trials
                % Reset belief at the start of a new block
                if n == 1 || blocks(n) ~= blocks(n-1)
                    q_0_0 = 0.5;
                end

                p_o_given_u = fitSlider_ALLmodels.observation_weights(set_o, condiff(n), sigma);

                % --- Pass 1: integrated action values -> softmax choice ---
                % (uses q_0_0 from BEFORE this trial's update -- the agent
                % chooses based on what it believed walking into the trial)
                pi_0_grid = NaN(n_o, 1);
                pi_1_grid = NaN(n_o, 1);
                v_a0_grid = NaN(n_o, 1);
                for i = 1:n_o
                    agent.p_s_giv_o(set_o(i));
                    pi_0_grid(i) = agent.pi_0;
                    pi_1_grid(i) = agent.pi_1;
                    % Agent.compute_valence()'s formula, with q_0_0 playing
                    % the role of the tracked contingency belief (E_mu_t)
                    v_a0_grid(i) = (pi_0_grid(i) - pi_1_grid(i)) * q_0_0 + pi_1_grid(i);
                end
                v_a0 = sum(v_a0_grid .* p_o_given_u');
                v_a1 = 1 - v_a0;

                p_choice0 = 1 / (1 + exp(-beta * (v_a0 - v_a1)));
                choice(n) = double(rand >= p_choice0); % 0 w.p. p_choice0, else 1

                % --- Reward: state-action contingency ---
                if choice(n) == state(n)
                    reward(n) = double(rand < p_reward_correct);
                else
                    reward(n) = double(rand < (1 - p_reward_correct));
                end

                % Recode reward relative to action 0 before it feeds the
                % belief update: reward(n)=1 means "action 0 was reinforced"
                % if choice(n)==0, but means "action 1 was reinforced" (i.e.
                % evidence AGAINST action 0) if choice(n)==1. Without this,
                % reward=1 would be treated identically regardless of which
                % action produced it -- and since choice(n) itself depends on
                % q_0_0, that confound creates a self-reinforcing feedback
                % loop with no reliable link to the true state.
                recoded_reward = fitSlider_ALLmodels.recode_rewards_choice(reward(n), choice(n));

                % --- Pass 2: integrated belief update using this reward ---
                % (reuses pi_0_grid/pi_1_grid from pass 1 -- same trial, same
                % observation-noise distribution, no need to recompute)
                voi_matrix = NaN(n_o, 2);
                for i = 1:n_o
                    q_sim = q_0_0;
                    if pi_0_grid(i) >= pi_1_grid(i)
                        q_sim = q_sim + alpha * (recoded_reward - q_sim);
                    else
                        q_sim = q_sim + alpha * ((1 - recoded_reward) - q_sim);
                    end
                    voi_matrix(i, :) = [q_sim, 1 - q_sim];
                end
                q_0_0 = sum(voi_matrix(:,1) .* p_o_given_u');
                q_0_0_trace(n) = q_0_0;

                a = max(q_0_0 * kappa, 1e-9);
                b = max((1 - q_0_0) * kappa, 1e-9);
                mu_hat(n) = betarnd(a, b);
            end
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

            % Belief state (pi_0/pi_1) at each hypothetical observation is
            % computed via Agent.p_s_giv_o -- the same formula/bounds as
            % agentvars.m's set_o/kappa_max defaults used here -- instead
            % of duplicating it inline. The RL update rule below (belief-
            % weighted learning rate) stays specific to this model.
            agent = Agent();
            agent.sigma = sigma;

            for n = 1:length(mu_hat)

                % Reset Q at block transitions
                if n == 1 || blocks(n) ~= blocks(n-1)
                    q_0_0 = 0.5;
                end

                % Observation likelihood under current internal estimate
                p_o_given_u = fitSlider_ALLmodels.observation_weights(set_o, condiff(n), sigma);

                % Initialize matrix to hold hypothetical Q-values
                voi_matrix = NaN(length(set_o), 2);

                for i = 1:length(set_o)
                    % Belief state at this hypothetical observation
                    agent.p_s_giv_o(set_o(i));
                    pi_0 = agent.pi_0;
                    pi_1 = agent.pi_1;

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
                    fprintf('Invalid beta params at trial %d: q=%.3g, kappa=%.3g, a=%.3g, b=%.3g (alpha=%.3g, sigma=%.3g)\n', ...
                        n, q_transformed, kappa, a, b, alpha, sigma);
                    fitSlider_ALLmodels.plot_beta_diagnostics(n, set_o, p_o_given_u, voi_matrix(:,1), ...
                        condiff(n), mu_hat(n), q_transformed, a, b, kappa, ...
                        sprintf('alpha=%.3g, sigma=%.3g', alpha, sigma));
                    a = max(eps, a);
                    b = max(eps, b);
                end

                p = betapdf(mu_hat(n), a, b);
                nll_trial(n) = log(p + eps);
            end

            nll = -nansum(nll_trial,"all");
        end

        %% ===================================================================
        %  RL SIGMA MODEL - GENERATIVE SIMULATION WITH ECONOMIC CHOICE
        %  -------------------------------------------------------------------
        %  Like simulate_basicRL_integrated_choice, but the belief update
        %  uses RLsigma's belief-weighted learning rate (pi_0/pi_1 scaling
        %  alpha) -- matching nll_RLsigma_VOI's update rule -- instead of
        %  basicRL's plain alpha update. Choice generation (softmax over
        %  integrated action values) and the state-action-reward
        %  contingency are identical to simulate_basicRL_integrated_choice,
        %  since that economic-choice layer sits on top of whichever
        %  value-update rule is being tested and isn't itself part of what
        %  distinguishes the two models.
        %  Inputs:
        %    params            - [alpha, kappa, sigma]
        %    blocks            - block index per trial (vector)
        %    state             - ground-truth state per trial, 0 or 1 (vector)
        %    condiff           - contrast difference per trial (vector)
        %    beta              - fixed softmax inverse-temperature for choice
        %                        (not estimated; higher = more deterministic)
        %    p_reward_correct  - P(reward=1 | choice matches state)
        %  Outputs:
        %    mu_hat      - simulated slider responses (vector)
        %    choice      - simulated economic choice, 0 or 1 (vector)
        %    reward      - simulated reward outcome, 0 or 1 (vector)
        %    q_0_0_trace - underlying integrated value belief per trial (vector)
        % ====================================================================
        function [mu_hat, choice, reward, q_0_0_trace] = simulate_RLsigma_integrated_choice(params, blocks, state, condiff, beta, p_reward_correct)
            alpha = params(1);
            kappa = params(2);
            sigma = params(3);

            n_trials = length(condiff);
            mu_hat = NaN(n_trials, 1);
            choice = NaN(n_trials, 1);
            reward = NaN(n_trials, 1);
            q_0_0_trace = NaN(n_trials, 1);
            q_0_0 = 0.5;

            set_o = linspace(-0.1, 0.1, 20);
            n_o = length(set_o);

            agent = Agent();
            agent.sigma = sigma;

            for n = 1:n_trials
                % Reset belief at the start of a new block
                if n == 1 || blocks(n) ~= blocks(n-1)
                    q_0_0 = 0.5;
                end

                p_o_given_u = fitSlider_ALLmodels.observation_weights(set_o, condiff(n), sigma);

                % --- Pass 1: integrated action values -> softmax choice ---
                pi_0_grid = NaN(n_o, 1);
                pi_1_grid = NaN(n_o, 1);
                v_a0_grid = NaN(n_o, 1);
                for i = 1:n_o
                    agent.p_s_giv_o(set_o(i));
                    pi_0_grid(i) = agent.pi_0;
                    pi_1_grid(i) = agent.pi_1;
                    v_a0_grid(i) = (pi_0_grid(i) - pi_1_grid(i)) * q_0_0 + pi_1_grid(i);
                end
                v_a0 = sum(v_a0_grid .* p_o_given_u');
                v_a1 = 1 - v_a0;

                p_choice0 = 1 / (1 + exp(-beta * (v_a0 - v_a1)));
                choice(n) = double(rand >= p_choice0); % 0 w.p. p_choice0, else 1

                % --- Reward: state-action contingency ---
                if choice(n) == state(n)
                    reward(n) = double(rand < p_reward_correct);
                else
                    reward(n) = double(rand < (1 - p_reward_correct));
                end

                % Recode reward relative to action 0 before it feeds the
                % belief update (same reasoning as
                % simulate_basicRL_integrated_choice).
                recoded_reward = fitSlider_ALLmodels.recode_rewards_choice(reward(n), choice(n));

                % --- Pass 2: integrated belief update using this reward ---
                % Belief-weighted learning rate (pi_0/pi_1 scaling alpha) --
                % matches nll_RLsigma_VOI's update rule, unlike basicRL's
                % plain alpha update.
                voi_matrix = NaN(n_o, 2);
                for i = 1:n_o
                    q_sim = q_0_0;
                    if pi_0_grid(i) >= pi_1_grid(i)
                        q_sim = q_sim + pi_0_grid(i) * alpha * (recoded_reward - q_sim);
                    else
                        q_sim = q_sim + pi_1_grid(i) * alpha * ((1 - recoded_reward) - q_sim);
                    end
                    voi_matrix(i, :) = [q_sim, 1 - q_sim];
                end
                q_0_0 = sum(voi_matrix(:,1) .* p_o_given_u');
                q_0_0_trace(n) = q_0_0;

                a = max(q_0_0 * kappa, 1e-9);
                b = max((1 - q_0_0) * kappa, 1e-9);
                mu_hat(n) = betarnd(a, b);
            end
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

            for n = 1:length(mu_hat)

                % Reset Q at block transitions
                if n == 1 || blocks(n) ~= blocks(n-1)
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
                    fprintf('Invalid beta params at trial %d: q=%.3g, kappa=%.3g, a=%.3g, b=%.3g (confirmBias=%.3g, noconfirmBias=%.3g, sigma=%.3g)\n', ...
                        n, q_transformed, kappa, a, b, confirmBias, noconfirmBias, sigma);
                    fitSlider_ALLmodels.plot_beta_diagnostics(n, set_o, p_o_given_u, voi_matrix(:,1), ...
                        condiff(n), mu_hat(n), q_transformed, a, b, kappa, ...
                        sprintf('confirmBias=%.3g, noconfirmBias=%.3g, sigma=%.3g', confirmBias, noconfirmBias, sigma));
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

            for n = 1:length(mu_hat)

                % Reset Q at block transitions
                if n == 1 || blocks(n) ~= blocks(n-1)
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
                    fprintf('Invalid beta params at trial %d: q=%.3g, kappa=%.3g, a=%.3g, b=%.3g (confirmBias=%.3g, noconfirmBias=%.3g, sigma=%.3g)\n', ...
                        n, q_transformed, kappa, a, b, confirmBias, noconfirmBias, sigma);
                    fitSlider_ALLmodels.plot_beta_diagnostics(n, set_o, p_o_given_u, voi_matrix(:,1), ...
                        condiff(n), mu_hat(n), q_transformed, a, b, kappa, ...
                        sprintf('confirmBias=%.3g, noconfirmBias=%.3g, sigma=%.3g', confirmBias, noconfirmBias, sigma));
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

            for n = 1:length(mu_hat)

                % Reset Q at block transitions
                if n == 1 || blocks(n) ~= blocks(n-1)
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

            for n = 1:length(mu_hat)

                % Reset Q-value at block transitions
                if n == 1 || blocks(n) ~= blocks(n-1)
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

            for n = 1:length(mu_hat)

                % Reset Q-values at block transitions
                if n == 1 || blocks(n) ~= blocks(n-1)
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
                        a = max(eps, a);
                        b = max(eps, b);
                    end

                    % Log-likelihood for this trial
                    p = betapdf(mu_hatBlocks(t), a, b);
                    nll_trial(t,bl) = log(p + eps);
                    mu = [mu;agent.G];
                end
            end
            nll = -nansum(nll_trial,"all");
        end

        %% ===================================================================
        %  BAYESIAN AGENT MODEL - GENERATIVE SIMULATION WITH ECONOMIC CHOICE
        %  -------------------------------------------------------------------
        %  Simulates synthetic mu_hat/choice/reward data by driving the real
        %  Agent class the exact same way nll_bayesianAgent does: a fresh
        %  Agent() per block (nll_bayesianAgent's own block-reset mechanism
        %  -- a brand-new agent each block, rather than resetting a scalar
        %  belief like the RL models), agent.task_agent_analysis = 1,
        %  single-observation agent.p_s_giv_o/compute_valence/softmax for
        %  choice, agent.learn(reward) for the value update, and mu_hat
        %  drawn Beta(G*kappa, (1-G)*kappa) from whatever agent.G holds
        %  after learn(). This deliberately reproduces agent.G's real
        %  current behavior (including the one-trial-lag side effect of
        %  compute_q() inside integrate_voi, discussed separately) rather
        %  than a corrected version, so this dataset exercises exactly what
        %  nll_bayesianAgent actually computes today.
        %  Choice is sampled from the agent's own softmax (agent.p_a_t,
        %  using agent.beta as-is -- agentvars.m's default of 100, since
        %  nll_bayesianAgent never overrides it), and reward from a
        %  state-action contingency (matching the basicRL/RLsigma choice
        %  simulators). Reward is passed to agent.learn() RAW (not
        %  pre-recoded) -- the Agent class recodes it internally via
        %  compute_action_dep_rew using agent.a_t (set to the sampled
        %  choice beforehand), exactly like nll_bayesianAgent's own
        %  agent.a_t = choices(t); agent.learn(rewardsBlocks(t)) call.
        %  Inputs:
        %    params            - [kappa, sigma]
        %    blocks            - block index per trial (vector)
        %    state             - ground-truth state per trial, 0 or 1 (vector)
        %    condiff           - contrast difference per trial (vector)
        %    p_reward_correct  - P(reward=1 | choice matches state)
        %  Outputs:
        %    mu_hat  - simulated slider responses (vector)
        %    choice  - simulated economic choice, 0 or 1 (vector)
        %    reward  - simulated RAW reward outcome, 0 or 1 (vector)
        %    G_trace - agent.G read right after learn() each trial (vector);
        %              useful for diagnosing the one-trial-lag behavior
        %              separately from Beta-sampling noise in mu_hat
        % ====================================================================
        function [mu_hat, choice, reward, G_trace] = simulate_bayesianAgent_integrated_choice(params, blocks, state, condiff, p_reward_correct)
            kappa = params(1);
            sigma = params(2);

            n_trials = length(condiff);
            mu_hat = NaN(n_trials, 1);
            choice = NaN(n_trials, 1);
            reward = NaN(n_trials, 1);
            G_trace = NaN(n_trials, 1);

            uniqueBlocks = unique(blocks);
            for bl = 1:length(uniqueBlocks)
                % Fresh Agent() per block -- matches nll_bayesianAgent's own
                % block-reset mechanism (see design note above).
                agent = Agent();
                agent.task_agent_analysis = 1;
                agent.confirmation_bias = 0;
                agent.sigma = sigma;

                trial_idx = find(blocks == uniqueBlocks(bl));
                for k = 1:length(trial_idx)
                    n = trial_idx(k);

                    % --- Choice: single-observation belief -> valence -> softmax ---
                    agent.o_t = condiff(n);
                    agent.p_s_giv_o(agent.o_t);
                    agent.compute_valence();
                    agent.softmax();
                    choice(n) = binornd(1, agent.p_a_t(2));

                    % --- Reward: state-action contingency ---
                    if choice(n) == state(n)
                        reward(n) = double(rand < p_reward_correct);
                    else
                        reward(n) = double(rand < (1 - p_reward_correct));
                    end

                    % --- Value update: raw reward, Agent recodes internally ---
                    agent.a_t = choice(n);
                    agent.learn(reward(n));

                    G_trace(n) = agent.G;
                    a = max(agent.G * kappa, 1e-9);
                    b = max((1 - agent.G) * kappa, 1e-9);
                    mu_hat(n) = betarnd(a, b);
                end
            end
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
                        a = max(eps, a);
                        b = max(eps, b);
                    end

                    % Log-likelihood for this trial
                    p = betapdf(mu_hatBlocks(t), a, b);
                    nll_trial(t,bl) = log(p + eps);
                    mu = [mu;agent.G];
                end
            end
            nll = -nansum(nll_trial,"all");
        end

        %% ===================================================================
        %  PERCEPTUAL CHOICE MODEL
        %  -------------------------------------------------------------------
        %  Negative log-likelihood of a subject's binary perceptual choices
        %  given sigma, using a Bayesian ideal-observer agent (Agent class).
        %  Used to fit a subject's perceptual sensitivity independent of any
        %  reward-learning model (see fitPerceptualChoice.m).
        %  Inputs:
        %    params  - [sigma], perceptual sensitivity (observation noise) parameter
        %    data    - subject's trial table (must include blocks, choice, condiff_relative)
        %    nBlocks - number of blocks
        %    nTrials - number of trials per block
        %    blocks  - block index for each trial
        % ====================================================================
        function nll = nll_perceptualChoice(params, data, nBlocks, nTrials, blocks)
            sigma = params(1);
            nll_trial = NaN(nTrials,nBlocks);   % Per-trial log-likelihood, laid out [trial x block]
            uniqueBlocks = unique(blocks);

            for bl = 1:nBlocks
                % Fresh Bayesian agent for each block, evaluated at the candidate sigma
                agent = Agent();
                agent.task_agent_analysis = 1;   
                agent.confirmation_bias = 0;     % No confirmation bias in this model
                agent.sigma = sigma;

                % This block's trials
                dataBlocks = data(data.blocks == uniqueBlocks(bl),:);
                choices = dataBlocks.choice;
                condiff = dataBlocks.condiff_relative;

                for t = 1:height(dataBlocks)

                    % Bayesian agent inference steps
                    agent.o_t = condiff(t);      % Set this trial's perceptual observation
                    agent.decide_p();            % Compute the agent's perceptual choice probabilities

                    % Probability the agent assigns to the choice actually made,
                    % clipped away from 0/1 to avoid -Inf from log()
                    p = max(min(agent.p_d_t(choices(t) + 1), 1 - 1e-10), 1e-10);
                    nll_trial(t,bl) = log(p);
                end
            end
            % Sum log-likelihoods across all trials/blocks and negate -> total NLL
            nll = -nansum(nll_trial,"all");
        end

        %% ===================================================================
        %  PERCEPTUAL CHOICE MODEL - GENERATIVE SIMULATION
        %  -------------------------------------------------------------------
        %  Simulates a Bayesian ideal-observer agent's binary perceptual
        %  choices at a given sigma -- the generative counterpart of
        %  nll_perceptualChoice, used for parameter recovery.
        %  Inputs:
        %    params  - [sigma], perceptual sensitivity (observation noise) parameter
        %    condiff - relative contrast difference per trial (vector)
        %    blocks  - block index for each trial (vector)
        %    nBlocks - number of blocks
        %  Outputs:
        %    choicesAll - simulated choices, laid out [1 x length(condiff)],
        %                 concatenated block by block (matches condiff's row
        %                 order, since blocks are matched by ID rather than
        %                 assumed to be fixed-size contiguous chunks -- same
        %                 block-handling as nll_perceptualChoice, so this
        %                 works whether trial counts per block are uniform
        %                 (synthetic recovery data) or not (real subject data,
        %                 where a subject can have a different number of
        %                 trials per block after preprocessing)
        % ====================================================================
        function choicesAll = simulate_perceptualChoice(params, condiff, blocks, nBlocks)
            sigma = params(1);
            choicesAll = [];
            agent = Agent();
            uniqueBlocks = unique(blocks);
            for bl = 1:nBlocks
                agent.task_agent_analysis = 1;
                agent.confirmation_bias = 0;
                agent.sigma = sigma;
                condiffBlock = condiff(blocks == uniqueBlocks(bl));
                choices = NaN(1, length(condiffBlock));
                for t = 1:length(condiffBlock)
                    % Bayesian agent inference and learning steps
                    agent.o_t = condiffBlock(t);
                    agent.decide_p();
                    choices(t) = agent.d_t;
                end
                choicesAll = [choicesAll, choices];
            end
        end

        %% ===================================================================
        %  PERCEPTUAL CHOICE MODEL - PARAMETER RECOVERY
        %  -------------------------------------------------------------------
        %  Validates the perceptual-choice sigma fit: simulates synthetic
        %  choice data at known sigma values (simulate_perceptualChoice),
        %  then re-fits sigma from that data (nll_perceptualChoice) with
        %  multi-start fmincon, and compares recovered vs. true sigma.
        %  Inputs:
        %    n_parameters     - number of synthetic subjects/sigma values to test
        %    n_startingPoints - number of random fmincon starting points per subject
        %    n_trials         - number of simulated trials per subject
        %  Outputs:
        %    sigmaRange            - true (generating) sigma per subject (vector)
        %    best_recovered_sigmas - best-fitting recovered sigma per subject (vector)
        %    best_nlls             - NLL at the best fit per subject (vector)
        % ====================================================================
        function [sigmaRange, best_recovered_sigmas, best_nlls] = recover_perceptualChoiceSigma(n_parameters, n_startingPoints, n_trials)
            sigmaRange = unifrnd(0, 0.07, [n_parameters, 1]);
            initSigma = unifrnd(0, 0.05, [n_parameters, n_startingPoints]);
            lb = 0;
            ub = 0.07;

            best_recovered_sigmas = NaN(n_parameters, 1);
            best_nlls = Inf(n_parameters, 1);

            % Synthetic trial data, shared across all subjects
            state = randi([0 1], n_trials, 1);
            condiff = NaN(n_trials, 1);
            condiff(state == 0) = -0.08 + (0 - (-0.08)) .* rand(sum(state == 0), 1); % uniform in [-0.08, 0]
            condiff(state == 1) = 0 + (0.08 - 0) .* rand(sum(state == 1), 1);        % uniform in [0, 0.08]

            block_size = 25; % trials per block
            nBlocks = n_trials / block_size;
            blocks = repelem(1:nBlocks, block_size)';

            parfor p = 1:n_parameters
                choices = fitSlider_ALLmodels.simulate_perceptualChoice(sigmaRange(p), condiff, blocks, nBlocks);
                data = table();
                data.choice = choices.';
                data.condiff_relative = condiff;
                data.blocks = blocks;

                nll_fun = @(params) fitSlider_ALLmodels.nll_perceptualChoice(params, data, nBlocks, block_size, blocks);

                for sp = 1:n_startingPoints
                    options = optimoptions('fmincon', ...
                        'Display', 'off', ...
                        'Algorithm', 'interior-point', ...
                        'FiniteDifferenceType', 'central', ...
                        'MaxIterations', 1000, ...
                        'FunctionTolerance', 1e-6);

                    [recovered_params, nll] = fmincon(nll_fun, initSigma(p,sp), ...
                        [], [], [], [], lb, ub, [], options);

                    if nll < best_nlls(p)
                        best_nlls(p) = nll;
                        best_recovered_sigmas(p) = recovered_params(1);
                    end
                end
                fprintf('Currently processing: Parameter %d/%d\n', p, n_parameters);
            end
        end

        %% ===================================================================
        %  PERCEPTUAL CHOICE MODEL - PLOT PARAMETER RECOVERY
        %  -------------------------------------------------------------------
        %  Plots recovered vs. true sigma from recover_perceptualChoiceSigma().
        % ====================================================================
        function plot_perceptualChoiceRecovery(sigmaRange, best_recovered_sigmas)
            color = copper(5);
            figure
            scatter(sigmaRange, best_recovered_sigmas, 'filled', 'o', 'MarkerFaceColor', ...
                color(3,:), 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.3,'SizeData',50)
            lsline
            xlabel('Actual Sigma Parameter')
            ylabel('Best Recovered Sigma Parameter')
            title('Sigma Recovery')
            axis equal
            grid on
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
        %  OBSERVATION WEIGHTS
        %  -------------------------------------------------------------------
        %  Normalized likelihood of each hypothetical observation on the
        %  set_o grid, given the true observation and sensory noise sigma.
        %  Same calculation as Agent.integrate_voi's p_o_giv_u/p_o_giv_u_norm,
        %  factored out here since nll_basicRL_integrated/nll_RLsigma_VOI use
        %  their own model-specific update rule and so don't call integrate_voi.
        %  Inputs:
        %    set_o - discretized hypothetical-observation grid (vector)
        %    o_t   - true observation for this trial (scalar)
        %    sigma - sensory noise (observation SD) parameter (scalar)
        %  Outputs:
        %    p_o_given_u - normalized observation-likelihood weights over
        %                  set_o (sums to 1)
        % ====================================================================
        function p_o_given_u = observation_weights(set_o, o_t, sigma)
            p_o_given_u = normpdf(set_o, o_t, sigma);
            p_o_given_u = p_o_given_u / sum(p_o_given_u);
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
        %  RECODE REWARD RELATIVE TO ACTION 0 (FOR SIMULATED RECOVERY DATA)
        %  -------------------------------------------------------------------
        %  Recodes a raw, choice-relative reward outcome (reward=1 means "the
        %  action the agent actually chose was reinforced") into a fixed,
        %  action-0-referenced frame (reward=1 means "action 0 was
        %  reinforced"), which is what nll_basicRL_integrated's and
        %  nll_RLsigma_VOI's belief-update delta rule assumes.
        %  This is a DIFFERENT recoding step from recode_rewards above:
        %  recode_rewards corrects real subject data for a presentation/
        %  encoding-convention flip recorded per trial (contrast == 1 means
        %  "actual mu < 0.5" for that block, see preprocess_LR.m's
        %  compute_mu) -- an ambiguity that only exists because real trials
        %  were logged with that flip convention. simulate_basicRL_integrated_
        %  choice / simulate_RLsigma_integrated_choice generate choice and
        %  reward directly with no such raw/flipped-presentation duality, so
        %  the only recoding their output needs before refitting is this
        %  choice-relative one -- used both internally by those two
        %  functions (per-trial) and by recovery_ReducedModelSpace.m
        %  (vectorized, after loading their saved output).
        %  Inputs:
        %    reward - raw reward outcome, 1 if the actually-chosen action
        %             was reinforced, 0 otherwise (scalar or vector)
        %    choice - the action that was chosen, 0 or 1 (scalar or vector,
        %             same size as reward)
        %  Outputs:
        %    recoded_reward - reward re-expressed relative to action 0
        %                      (same size as reward)
        % ====================================================================
        function recoded_reward = recode_rewards_choice(reward, choice)
            recoded_reward = reward;
            recoded_reward(choice == 1) = 1 - recoded_reward(choice == 1);
        end

        %% ===================================================================
        %  LOAD FITTING DATA
        %  -------------------------------------------------------------------
        %  Load the preprocessed slider-fitting dataset and split it into the
        %  "both" (mixed reward + perceptual) and perceptual-only condition
        %  subsets used by the fitting scripts.
        %  Outputs:
        %    data                - both-condition trials (condition == 1);
        %                          reward-condition trials (choice_cond == 3)
        %                          removed and condiff_relative precomputed
        %                          upstream by preprocessAllData.m
        %    dataPerceptual      - perceptual-condition trials (condition == 2)
        %    uniqueID            - unique subject IDs (vector)
        %    numSubjs            - number of subjects (scalar)
        %    dataPerceptualChoice - dataPerceptual, further restricted to
        %                          fitPerceptualChoice.m's needs: warm-up
        %                          trials (trials <= 5) dropped, and choice
        %                          recoded into a fixed reference frame
        %                          (flipped when contrast == 1) so it's
        %                          comparable across contrast conditions.
        %                          Appended as a 5th output so existing
        %                          4-output callers are unaffected.
        % ====================================================================
        function [data, dataPerceptual, uniqueID, numSubjs, dataPerceptualChoice] = load_fitting_data()
            data = importdata("preprocessed_dataFitting.mat");
            uniqueID = unique(data.ID);
            numSubjs = length(uniqueID);
            dataPerceptual = data(data.condition ~= 1,:);
            data(data.condition == 2,:) = [];

            dataPerceptualChoice = dataPerceptual(dataPerceptual.trials > 5,:);
            flipRows = dataPerceptualChoice.contrast == 1;
            dataPerceptualChoice.choice(flipRows) = 1 - dataPerceptualChoice.choice(flipRows);
        end

        %% ===================================================================
        %  PLOT BETA DIAGNOSTICS
        %  -------------------------------------------------------------------
        %  Visualizes why a trial's Beta(a, b) likelihood parameters went
        %  invalid (called from the "Invalid beta params" checks in the
        %  nll_* functions above, right where a/b are computed):
        %    1. how peaked the observation-likelihood is around condiff,
        %       given sigma (a narrow spike means little averaging);
        %    2. whether the hypothetical Q-value collapses to the same
        %       value across the whole observation grid (e.g. a large
        %       alpha/confirmBias makes q_sim jump straight to the reward,
        %       leaving nothing to average over);
        %    3. the resulting Beta(a, b) density against the observed
        %       mu_hat for that trial.
        %
        %  NOTE: parfor workers cannot display figures, so this is a no-op
        %  when called from inside a parfor loop (e.g. the fitting loops in
        %  fitReducedModelSpace.m) -- it only renders when running on the
        %  client. Even then, fmincon can call the objective function (and
        %  so this) many times per fit, so expect multiple figure windows
        %  if this keeps triggering during a single run.
        %  Inputs:
        %    trial_idx     - trial index (n) where a<=0 or b<=0
        %    set_o         - discretized hypothetical-observation grid
        %    p_o_given_u   - observation-likelihood weights over set_o
        %                    (already normalized to sum to 1)
        %    qSimGrid      - hypothetical Q-value at each set_o point
        %                    (e.g. voi_matrix(:,1))
        %    condiff_n     - true contrast difference for this trial
        %    mu_hat_n      - observed slider value for this trial
        %    q_transformed - integrated belief that produced a/b
        %    a, b          - the (invalid) Beta shape parameters
        %    kappa         - concentration parameter
        %    paramLabel    - string describing the candidate free
        %                    parameters (e.g. 'alpha=1, sigma=0.00708')
        % ====================================================================
        function plot_beta_diagnostics(trial_idx, set_o, p_o_given_u, qSimGrid, ...
                condiff_n, mu_hat_n, q_transformed, a, b, kappa, paramLabel)
            if ~isempty(getCurrentTask())
                return % on a parfor worker -- can't display a figure here
            end

            figure('Position', [100, 100, 800, 750]);

            subplot(3,1,1);
            bar(set_o, p_o_given_u, 'FaceColor', [0.3 0.5 0.8]);
            xline(condiff_n, 'r--', 'condiff', 'LineWidth', 1.5);
            xlabel('Hypothetical observation (set_o)');
            ylabel('P(o | u)');
            title(sprintf('Observation likelihood at trial %d', trial_idx));
            grid on;

            subplot(3,1,2);
            bar(set_o, qSimGrid, 'FaceColor', [0.8 0.4 0.3]);
            xlabel('Hypothetical observation (set_o)');
            ylabel('Simulated Q-value (q_{sim})');
            title(sprintf('Hypothetical belief update per observation (%s)', paramLabel));
            ylim([-0.05 1.05]);
            grid on;

            subplot(3,1,3);
            if a > 0 && b > 0
                x = linspace(0.001, 0.999, 200);
                plot(x, betapdf(x, a, b), 'LineWidth', 2, 'Color', [0.2 0.6 0.3]);
            else
                text(0.5, 0.5, sprintf('Beta(%.3g, %.3g) is degenerate (a or b <= 0)', a, b), ...
                    'HorizontalAlignment', 'center');
                xlim([0 1]); ylim([0 1]);
            end
            hold on;
            xline(mu_hat_n, 'k--', 'observed mu_hat', 'LineWidth', 1.5);
            xlabel('mu_hat');
            ylabel('Density');
            title(sprintf('Resulting Beta(%.3g, %.3g), kappa=%.4g', a, b, kappa));
            grid on;

            sgtitle(sprintf('Trial %d: q = %.4g \\rightarrow a = %.4g, b = %.4g', ...
                trial_idx, q_transformed, a, b));
        end

        %% ===================================================================
        %  PROGRESS BAR
        %  -------------------------------------------------------------------
        %  Graphical progress bar shared by all model-fitting scripts.
        %  parfor workers can't open/update a figure directly, so each
        %  completed (subject, starting point) combination is sent through a
        %  parallel.pool.DataQueue and drawn here, on the client, via
        %  afterEach. For scripts with a serial (non-parfor) fitting loop,
        %  'update' can be called directly instead.
        %  IMPORTANT: create a FRESH parallel.pool.DataQueue before each
        %  afterEach() registration -- afterEach() adds a new listener rather
        %  than replacing the previous one, so reusing one queue across
        %  multiple model/condition blocks would leave earlier blocks'
        %  listeners (with their stale captured label) still firing and
        %  overwriting the bar.
        %    'reset'  - (numSubjs, n_startingPoints, label): open/reset the bar
        %    'update' - ([n, sp], numSubjs, n_startingPoints, label): advance it
        %    'close'  - (): close the bar window
        % ====================================================================
        function progress_bar(mode, varargin)
            persistent h count total
            switch mode
                case 'reset'
                    [numSubjs, n_startingPoints, label] = varargin{:};
                    count = 0;
                    total = numSubjs * n_startingPoints;
                    if isempty(h) || ~isvalid(h)
                        h = waitbar(0, '', 'Name', 'Model fitting progress');
                    end
                    waitbar(0, h, sprintf('%s: subject 0/%d, start 0/%d', label, numSubjs, n_startingPoints));
                case 'update'
                    [data, numSubjs, n_startingPoints, label] = varargin{:};
                    count = count + 1;
                    n = data(1);
                    sp = data(2);
                    waitbar(min(count / total, 1), h, ...
                        sprintf('%s: subject %d/%d, start %d/%d', label, n, numSubjs, sp, n_startingPoints));
                case 'close'
                    if ~isempty(h) && isvalid(h)
                        close(h);
                    end
            end
        end

    end % methods
end % classdef
