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
        %  BASIC RL MODEL (OBJECTIVE PE) - BELIEF TRAJECTORY RECONSTRUCTION
        %  -------------------------------------------------------------------
        %  Diagnostic-only utility: re-runs nll_basicRL_integrated's exact
        %  recursive belief update, but returns the trial-by-trial
        %  q_transformed trajectory instead of the resulting NLL. Used to
        %  check, from already-fitted parameters, whether mu_hat needs a
        %  contrast-based flip to stay comparable to q_transformed -- the
        %  same question that turned out to be a real bug for the
        %  subjective-PE model's q_anchor (see conversation re: q_anchor
        %  fix). Not used anywhere in the fitting/comparison pipeline.
        %  Inputs: identical to nll_basicRL_integrated (minus mu_hat's
        %  role in the likelihood -- only its length is used here).
        %  Outputs:
        %    q_transformed_trial - per-trial reconstructed model belief
        % ====================================================================
        function q_transformed_trial = reconstruct_basicRL_integrated_trajectory(params, mu_hat, blocks, rewards, condiff)
            alpha = params(1);
            sigma = params(3);

            q_transformed_trial = NaN(length(mu_hat),1);
            q_0_0 = 0.5;
            set_o = linspace(-0.1, 0.1, 20);

            agent = Agent();
            agent.sigma = sigma;

            for n = 1:length(mu_hat)
                if n == 1 || blocks(n) ~= blocks(n-1)
                    q_0_0 = 0.5;
                end

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
                q_transformed_trial(n) = q_0_0;
            end
        end

        %% ===================================================================
        %  BASIC RL MODEL - SUBJECTIVE PE VARIANT
        %  -------------------------------------------------------------------
        %  Same as nll_basicRL_integrated (integration over observations),
        %  except the delta-rule's anchor each trial is the subject's own
        %  actual previous slider report (mu_hat(n-1)) rather than the
        %  model's own internally-tracked running belief (q_0_0, carried
        %  over from the previous trial's integrated update). This matches
        %  the "subjective PE" definition already used elsewhere in this
        %  codebase for the simple LR regression (preprocess_LR.m's
        %  compute_state_dep_pe: pe = reward - mu_t_1, the subject's own
        %  previous report) rather than an agent/model-derived ("objective")
        %  PE -- see conversation from 2026-09-22 for the full
        %  subjective-vs-objective-PE discussion and trade-offs. Kept as a
        %  separate variant from nll_basicRL_integrated (not a replacement),
        %  so the two can be compared (AIC/BIC, parameter recovery) rather
        %  than committing to one.
        %  Inputs/outputs: identical to nll_basicRL_integrated.
        % ====================================================================
        function nll = nll_basicRL_integrated_subjectivePE(params, mu_hat, blocks, rewards, condiff)
            alpha = params(1);
            kappa = params(2);
            sigma = params(3);

            nll_trial = NaN(length(mu_hat),1);
            eps = 1e-9;
            mu_hat(mu_hat <= 0) = eps;
            mu_hat(mu_hat >= 1) = 1 - eps;

            % Discretize observation space (like obj.set_o)
            set_o = linspace(-0.1, 0.1, 20);

            agent = Agent();
            agent.sigma = sigma;

            for n = 1:length(mu_hat)
                % Anchor for this trial's delta-rule update: 0.5 at the
                % start of a new block (same reset convention as
                % nll_basicRL_integrated), otherwise the subject's own
                % actual previous slider report -- NOT a
                % model-internally-tracked running belief.
                if n == 1 || blocks(n) ~= blocks(n-1)
                    q_anchor = 0.5;
                else
                    q_anchor = mu_hat(n-1);
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
                    q_sim = q_anchor;

                    % Basic RL update (without learning rate modulation like sigmaRL)
                    if pi_0 >= pi_1
                        q_sim = q_sim + alpha * (rewards(n) - q_sim);
                    else
                        q_sim = q_sim + alpha * ((1 - rewards(n)) - q_sim);
                    end

                    voi_matrix(i, :) = [q_sim, 1 - q_sim];
                end

                % Integrate over all possible observations
                q_transformed = sum(voi_matrix(:,1) .* p_o_given_u');

                % Beta parameters for likelihood
                a = q_transformed * kappa;
                b = (1 - q_transformed) * kappa;

                % Avoid invalid beta params
                if a <= 0 || b <= 0
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
        %  BASIC RL MODEL - POINT-OBSERVATION VARIANT (NO INTEGRATION)
        %  -------------------------------------------------------------------
        %  Same as nll_basicRL_integrated, except belief each trial comes
        %  from a single point estimate of the observation (the actual
        %  condiff(n), via normcdf) rather than integrated over a
        %  discretized grid of hypothetical observations -- mirrors
        %  nll_RLsigma's point-estimate belief computation exactly, just
        %  with basicRL's plain alpha update (not belief-weighted) in place
        %  of nll_RLsigma's pi-weighted update. Kept as a separate variant
        %  from nll_basicRL_integrated (not a replacement), so the two can
        %  be compared (AIC/BIC, parameter recovery) rather than committing
        %  to one.
        %  Inputs/outputs: identical to nll_basicRL_integrated.
        % ====================================================================
        function nll = nll_basicRL_pointObs(params, mu_hat, blocks, rewards, condiff)
            alpha = params(1);
            kappa = params(2);
            sigma = params(3);

            nll_trial = NaN(length(mu_hat),1);
            eps = 1e-9;
            mu_hat(mu_hat <= 0) = eps;
            mu_hat(mu_hat >= 1) = 1 - eps;

            q_0_0 = 0.5;

            for n = 1:length(mu_hat)
                if n == 1 || blocks(n) ~= blocks(n-1)
                    q_0_0 = 0.5;
                end

                % Point-estimate belief from the actual observed condiff
                % (no integration over hypothetical observations) -- same
                % mechanism as nll_RLsigma.
                belief_state = normcdf(condiff(n), 0, sigma);
                if condiff(n) < 0
                    pi_0 = 1 - belief_state;
                    pi_1 = belief_state;
                else
                    pi_1 = belief_state;
                    pi_0 = 1 - belief_state;
                end

                % Plain alpha update (not belief-weighted, unlike
                % nll_RLsigma) -- matches nll_basicRL_integrated's update
                % rule.
                if pi_0 >= pi_1
                    q_0_0 = q_0_0 + alpha * (rewards(n) - q_0_0);
                else
                    q_0_0 = q_0_0 + alpha * ((1 - rewards(n)) - q_0_0);
                end

                a = q_0_0 * kappa;
                b = (1 - q_0_0) * kappa;
                if a <= 0 || b <= 0
                    a = max(eps, a);
                    b = max(eps, b);
                end

                p = betapdf(mu_hat(n), a, b);
                nll_trial(n) = log(p + eps);
            end

            nll = -nansum(nll_trial,"all");
        end

        %% ===================================================================
        %  BASIC RL MODEL - POINT-OBSERVATION VARIANT, SUBJECTIVE PE
        %  -------------------------------------------------------------------
        %  Same as nll_basicRL_pointObs, except the delta-rule's anchor each
        %  trial is the subject's own actual previous slider report
        %  (mu_hat(n-1)) rather than the model's own internally-tracked
        %  running belief -- same subjective-PE substitution as
        %  nll_basicRL_integrated_subjectivePE, applied to the point-
        %  estimate (non-integrated) belief mechanism instead of the
        %  integrated one.
        %  Inputs/outputs: identical to nll_basicRL_pointObs.
        % ====================================================================
        function nll = nll_basicRL_pointObs_subjectivePE(params, mu_hat, blocks, rewards, condiff)
            alpha = params(1);
            kappa = params(2);
            sigma = params(3);

            nll_trial = NaN(length(mu_hat),1);
            eps = 1e-9;
            mu_hat(mu_hat <= 0) = eps;
            mu_hat(mu_hat >= 1) = 1 - eps;

            for n = 1:length(mu_hat)
                % Anchor for this trial's delta-rule update: 0.5 at the
                % start of a new block, otherwise the subject's own actual
                % previous slider report -- NOT a model-internally-tracked
                % running belief.
                if n == 1 || blocks(n) ~= blocks(n-1)
                    q_anchor = 0.5;
                else
                    q_anchor = mu_hat(n-1);
                end

                belief_state = normcdf(condiff(n), 0, sigma);
                if condiff(n) < 0
                    pi_0 = 1 - belief_state;
                    pi_1 = belief_state;
                else
                    pi_1 = belief_state;
                    pi_0 = 1 - belief_state;
                end

                q_sim = q_anchor;
                if pi_0 >= pi_1
                    q_sim = q_sim + alpha * (rewards(n) - q_sim);
                else
                    q_sim = q_sim + alpha * ((1 - rewards(n)) - q_sim);
                end

                a = q_sim * kappa;
                b = (1 - q_sim) * kappa;
                if a <= 0 || b <= 0
                    a = max(eps, a);
                    b = max(eps, b);
                end

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
        %  BASIC RL MODEL - GENERATIVE SIMULATION WITH ECONOMIC CHOICE,
        %  SUBJECTIVE PE VARIANT
        %  -------------------------------------------------------------------
        %  Like simulate_basicRL_integrated_choice, except the belief carried
        %  forward from trial to trial is this synthetic subject's own
        %  previous SIMULATED report (mu_hat(n-1)), not a separately-tracked
        %  clean latent q_0_0 -- the fully-generative analog of
        %  nll_basicRL_integrated_subjectivePE's anchor substitution (see its
        %  header for the full rationale). Unlike the objective-PE version,
        %  the Beta-sampling noise in mu_hat(n) is not thrown away after use:
        %  it becomes q_anchor for trial n+1, so noise compounds across a
        %  block here in a way it never does in simulate_basicRL_integrated_
        %  choice (where q_0_0 is deterministic given parameters/rewards,
        %  independent of any particular betarnd draw).
        %  Inputs/outputs: identical to simulate_basicRL_integrated_choice,
        %  except the 4th output is q_anchor_trace (the anchor actually used
        %  each trial, i.e. mu_hat(n-1) or 0.5 at a block boundary) rather
        %  than a clean q_0_0_trace.
        % ====================================================================
        function [mu_hat, choice, reward, q_anchor_trace] = simulate_basicRL_integrated_choice_subjectivePE(params, blocks, state, condiff, beta, p_reward_correct)
            alpha = params(1);
            kappa = params(2);
            sigma = params(3);

            n_trials = length(condiff);
            mu_hat = NaN(n_trials, 1);
            choice = NaN(n_trials, 1);
            reward = NaN(n_trials, 1);
            q_anchor_trace = NaN(n_trials, 1);

            set_o = linspace(-0.1, 0.1, 20);
            n_o = length(set_o);

            agent = Agent();
            agent.sigma = sigma;

            for n = 1:n_trials
                % Anchor for this trial: 0.5 at the start of a new block,
                % otherwise this synthetic subject's own previous simulated
                % report -- NOT a separately-tracked clean belief.
                if n == 1 || blocks(n) ~= blocks(n-1)
                    q_anchor = 0.5;
                else
                    q_anchor = mu_hat(n-1);
                end
                q_anchor_trace(n) = q_anchor;

                p_o_given_u = fitSlider_ALLmodels.observation_weights(set_o, condiff(n), sigma);

                % --- Pass 1: integrated action values -> softmax choice ---
                % (uses q_anchor from BEFORE this trial's update -- same
                % pre-update-belief-drives-choice ordering as
                % simulate_basicRL_integrated_choice's q_0_0)
                pi_0_grid = NaN(n_o, 1);
                pi_1_grid = NaN(n_o, 1);
                v_a0_grid = NaN(n_o, 1);
                for i = 1:n_o
                    agent.p_s_giv_o(set_o(i));
                    pi_0_grid(i) = agent.pi_0;
                    pi_1_grid(i) = agent.pi_1;
                    v_a0_grid(i) = (pi_0_grid(i) - pi_1_grid(i)) * q_anchor + pi_1_grid(i);
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

                recoded_reward = fitSlider_ALLmodels.recode_rewards_choice(reward(n), choice(n));

                % --- Pass 2: belief update from q_anchor using this reward
                % --- (plain alpha update, matching
                % nll_basicRL_integrated_subjectivePE -- reuses
                % pi_0_grid/pi_1_grid from pass 1)
                voi_matrix = NaN(n_o, 2);
                for i = 1:n_o
                    q_sim = q_anchor;
                    if pi_0_grid(i) >= pi_1_grid(i)
                        q_sim = q_sim + alpha * (recoded_reward - q_sim);
                    else
                        q_sim = q_sim + alpha * ((1 - recoded_reward) - q_sim);
                    end
                    voi_matrix(i, :) = [q_sim, 1 - q_sim];
                end
                q_transformed = sum(voi_matrix(:,1) .* p_o_given_u');

                a = max(q_transformed * kappa, 1e-9);
                b = max((1 - q_transformed) * kappa, 1e-9);
                mu_hat(n) = betarnd(a, b); % this trial's report -- becomes next trial's q_anchor
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
        %  RL SIGMA MODEL - SUBJECTIVE PE VARIANT
        %  -------------------------------------------------------------------
        %  Same as nll_RLsigma_VOI (belief-weighted learning rate,
        %  integration over observations), except the delta-rule's anchor
        %  each trial is the subject's own actual previous slider report
        %  (mu_hat(n-1)) rather than the model's own internally-tracked
        %  running belief (q_0_0, carried over from the previous trial's
        %  integrated update). See nll_basicRL_integrated_subjectivePE's
        %  header for the full subjective-vs-objective-PE rationale --
        %  this mirrors nll_RLsigma_VOI exactly except for that anchor
        %  substitution, kept as a separate variant rather than replacing
        %  it, so the two can be compared (AIC/BIC, parameter recovery)
        %  rather than committing to one.
        %  Inputs/outputs: identical to nll_RLsigma_VOI.
        % ====================================================================
        function nll = nll_RLsigma_VOI_subjectivePE(params, mu_hat, blocks, rewards, condiff)
            alpha = params(1);
            kappa = params(2);
            sigma = params(3);
            nll_trial = NaN(length(mu_hat),1);
            eps = 1e-6;
            mu_hat(mu_hat <= 0) = eps;
            mu_hat(mu_hat >= 1) = 1 - eps;

            % Discretize observation space (like obj.set_o)
            set_o = linspace(-0.1, 0.1, 20);

            agent = Agent();
            agent.sigma = sigma;

            for n = 1:length(mu_hat)
                % Anchor for this trial's delta-rule update: 0.5 at the
                % start of a new block (same reset convention as
                % nll_RLsigma_VOI), otherwise the subject's own actual
                % previous slider report -- NOT a model-internally-tracked
                % running belief.
                if n == 1 || blocks(n) ~= blocks(n-1)
                    q_anchor = 0.5;
                else
                    q_anchor = mu_hat(n-1);
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
                    q_sim = q_anchor;
                    if pi_0 >= pi_1
                        q_sim = q_sim + pi_0 * alpha * (rewards(n) - q_sim);
                    else
                        q_sim = q_sim + pi_1 * alpha * ((1 - rewards(n)) - q_sim);
                    end
                    voi_matrix(i, :) = [q_sim, 1 - q_sim];
                end

                % Integrate over all possible observations
                q_transformed = sum(voi_matrix(:,1) .* p_o_given_u');

                if any(voi_matrix(:,1) < 0)
                    fprintf('Negative Q-values detected!');
                    fprintf('Min Q-value: %.6f\n', min(voi_matrix(:,1)));
                end
                if q_transformed < 0
                    disp("Invalid q_0_0 params");
                end
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
        %  RL SIGMA MODEL - GENERATIVE SIMULATION WITH ECONOMIC CHOICE,
        %  SUBJECTIVE PE VARIANT
        %  -------------------------------------------------------------------
        %  Like simulate_RLsigma_integrated_choice, except the belief carried
        %  forward from trial to trial is this synthetic subject's own
        %  previous SIMULATED report (mu_hat(n-1)), not a separately-tracked
        %  clean latent q_0_0 -- the fully-generative analog of
        %  nll_RLsigma_VOI_subjectivePE's anchor substitution (see
        %  simulate_basicRL_integrated_choice_subjectivePE's header for the
        %  full rationale, including the compounding-noise consequence).
        %  Belief-weighted learning rate (pi_0/pi_1 scaling alpha), matching
        %  nll_RLsigma_VOI_subjectivePE's update rule, unlike basicRL's plain
        %  alpha update.
        %  Inputs/outputs: identical to simulate_RLsigma_integrated_choice,
        %  except the 4th output is q_anchor_trace rather than q_0_0_trace.
        % ====================================================================
        function [mu_hat, choice, reward, q_anchor_trace] = simulate_RLsigma_integrated_choice_subjectivePE(params, blocks, state, condiff, beta, p_reward_correct)
            alpha = params(1);
            kappa = params(2);
            sigma = params(3);

            n_trials = length(condiff);
            mu_hat = NaN(n_trials, 1);
            choice = NaN(n_trials, 1);
            reward = NaN(n_trials, 1);
            q_anchor_trace = NaN(n_trials, 1);

            set_o = linspace(-0.1, 0.1, 20);
            n_o = length(set_o);

            agent = Agent();
            agent.sigma = sigma;

            for n = 1:n_trials
                if n == 1 || blocks(n) ~= blocks(n-1)
                    q_anchor = 0.5;
                else
                    q_anchor = mu_hat(n-1);
                end
                q_anchor_trace(n) = q_anchor;

                p_o_given_u = fitSlider_ALLmodels.observation_weights(set_o, condiff(n), sigma);

                pi_0_grid = NaN(n_o, 1);
                pi_1_grid = NaN(n_o, 1);
                v_a0_grid = NaN(n_o, 1);
                for i = 1:n_o
                    agent.p_s_giv_o(set_o(i));
                    pi_0_grid(i) = agent.pi_0;
                    pi_1_grid(i) = agent.pi_1;
                    v_a0_grid(i) = (pi_0_grid(i) - pi_1_grid(i)) * q_anchor + pi_1_grid(i);
                end
                v_a0 = sum(v_a0_grid .* p_o_given_u');
                v_a1 = 1 - v_a0;

                p_choice0 = 1 / (1 + exp(-beta * (v_a0 - v_a1)));
                choice(n) = double(rand >= p_choice0);

                if choice(n) == state(n)
                    reward(n) = double(rand < p_reward_correct);
                else
                    reward(n) = double(rand < (1 - p_reward_correct));
                end

                recoded_reward = fitSlider_ALLmodels.recode_rewards_choice(reward(n), choice(n));

                % Belief-weighted learning rate (pi_0/pi_1 scaling alpha) --
                % matches nll_RLsigma_VOI_subjectivePE's update rule.
                voi_matrix = NaN(n_o, 2);
                for i = 1:n_o
                    q_sim = q_anchor;
                    if pi_0_grid(i) >= pi_1_grid(i)
                        q_sim = q_sim + pi_0_grid(i) * alpha * (recoded_reward - q_sim);
                    else
                        q_sim = q_sim + pi_1_grid(i) * alpha * ((1 - recoded_reward) - q_sim);
                    end
                    voi_matrix(i, :) = [q_sim, 1 - q_sim];
                end
                q_transformed = sum(voi_matrix(:,1) .* p_o_given_u');

                a = max(q_transformed * kappa, 1e-9);
                b = max((1 - q_transformed) * kappa, 1e-9);
                mu_hat(n) = betarnd(a, b); % this trial's report -- becomes next trial's q_anchor
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
                if a <= 0 || b <= 0
                    a = max(eps, a);
                    b = max(eps, b);
                end

                % Log-likelihood for this trial
                p = betapdf(mu_hat(n), a, b);
                nll_trial(n) = log(p + eps);
            end
            nll = -nansum(nll_trial,"all");
        end

        %% ===================================================================
        %  RL SIGMA MODEL - POINT-OBSERVATION VARIANT, SUBJECTIVE PE
        %  -------------------------------------------------------------------
        %  Same as nll_RLsigma (point-estimate belief, no integration over
        %  hypothetical observations), except the delta-rule's anchor each
        %  trial is the subject's own actual previous slider report
        %  (mu_hat(n-1)) rather than the model's own internally-tracked
        %  running belief -- same subjective-PE substitution as
        %  nll_RLsigma_VOI_subjectivePE, applied to the point-estimate
        %  (non-integrated) belief mechanism instead of the integrated one.
        %  Inputs/outputs: identical to nll_RLsigma.
        % ====================================================================
        function nll = nll_RLsigma_subjectivePE(params, mu_hat, blocks, rewards, condiff)
            alpha = params(1);
            kappa = params(2);
            sigma = params(3);
            nll_trial = NaN(length(mu_hat),1);
            eps = 1e-9;
            mu_hat(mu_hat <= 0) = eps;
            mu_hat(mu_hat >= 1) = 1 - eps;

            for n = 1:length(mu_hat)
                % Anchor for this trial's delta-rule update: 0.5 at the
                % start of a new block, otherwise the subject's own actual
                % previous slider report -- NOT a model-internally-tracked
                % running belief.
                if n == 1 || blocks(n) ~= blocks(n-1)
                    q_anchor = 0.5;
                else
                    q_anchor = mu_hat(n-1);
                end

                belief_state = normcdf(condiff(n), 0, sigma);
                if condiff(n) < 0
                    pi_0 = 1 - belief_state;
                    pi_1 = belief_state;
                else
                    pi_1 = belief_state;
                    pi_0 = 1 - belief_state;
                end

                q_sim = q_anchor;
                if pi_0 >= pi_1
                    q_sim = q_sim + pi_0 * alpha * (rewards(n) - q_sim);
                else
                    q_sim = q_sim + pi_1 * alpha * ((1 - rewards(n)) - q_sim);
                end

                a = q_sim * kappa;
                b = (1 - q_sim) * kappa;
                if a <= 0 || b <= 0
                    a = max(eps, a);
                    b = max(eps, b);
                end

                p = betapdf(mu_hat(n), a, b);
                nll_trial(n) = log(p + eps);
            end
            nll = -nansum(nll_trial,"all");
        end

        %% ===================================================================
        %  MIXTURE MODEL (BASIC RL / RL SIGMA), SUBJECTIVE PE, NO INTEGRATION
        %  -------------------------------------------------------------------
        %  Blends basicRL's and RLsigma's belief updates via a free mixture
        %  weight lambda, following GbAgent.py's compute_mixture pattern
        %  (gaborbandit_analysis-master/GbAgent.py, agents 3/6): each
        %  sub-model computes its OWN complete, independent belief update
        %  every trial (nll_basicRL_pointObs_subjectivePE's plain alpha
        %  update vs nll_RLsigma_subjectivePE's pi-weighted update, both
        %  anchored to the same q_anchor), and only the two RESULTING
        %  beliefs are blended afterwards:
        %    q_mixed = lambda * q_RLsigma + (1 - lambda) * q_basicRL
        %  rather than splicing the two update equations together
        %  algebraically. lambda = 1 is pure RLsigma (full confidence-
        %  weighted learning), lambda = 0 is pure basicRL (flat learning
        %  rate); intermediate values estimate the DEGREE of confidence-
        %  weighting in the data directly, instead of picking a winner via
        %  model comparison.
        %  Inputs:
        %    params  - [alpha, kappa, sigma, lambda]
        %    mu_hat  - observed slider reports (vector)
        %    blocks  - block index per trial (vector)
        %    rewards - state-0-referenced reward outcomes (vector)
        %    condiff - contrast difference per trial (vector)
        %  Output:
        %    nll - negative log-likelihood (scalar)
        % ====================================================================
        function nll = nll_mixture_subjectivePE(params, mu_hat, blocks, rewards, condiff)
            alpha = params(1);
            kappa = params(2);
            sigma = params(3);
            lambda = params(4);

            nll_trial = NaN(length(mu_hat),1);
            eps = 1e-9;
            mu_hat(mu_hat <= 0) = eps;
            mu_hat(mu_hat >= 1) = 1 - eps;

            for n = 1:length(mu_hat)
                % Anchor for this trial's delta-rule update: 0.5 at the
                % start of a new block, otherwise the subject's own actual
                % previous slider report -- same convention as
                % nll_basicRL_pointObs_subjectivePE / nll_RLsigma_subjectivePE,
                % shared by both sub-models so their outputs are directly
                % comparable before blending.
                if n == 1 || blocks(n) ~= blocks(n-1)
                    q_anchor = 0.5;
                else
                    q_anchor = mu_hat(n-1);
                end

                belief_state = normcdf(condiff(n), 0, sigma);
                if condiff(n) < 0
                    pi_0 = 1 - belief_state;
                    pi_1 = belief_state;
                else
                    pi_1 = belief_state;
                    pi_0 = 1 - belief_state;
                end

                % Each sub-model's own, independent update from the same
                % anchor -- basicRL's plain alpha step vs RLsigma's
                % pi-weighted step.
                q_basicRL = q_anchor;
                q_RLsigma = q_anchor;
                if pi_0 >= pi_1
                    q_basicRL = q_basicRL + alpha * (rewards(n) - q_basicRL);
                    q_RLsigma = q_RLsigma + pi_0 * alpha * (rewards(n) - q_RLsigma);
                else
                    q_basicRL = q_basicRL + alpha * ((1 - rewards(n)) - q_basicRL);
                    q_RLsigma = q_RLsigma + pi_1 * alpha * ((1 - rewards(n)) - q_RLsigma);
                end

                % Blend the two fully-independent updates (GbAgent.py's
                % compute_mixture pattern).
                q_mixed = lambda * q_RLsigma + (1 - lambda) * q_basicRL;

                a = q_mixed * kappa;
                b = (1 - q_mixed) * kappa;
                if a <= 0 || b <= 0
                    a = max(eps, a);
                    b = max(eps, b);
                end

                p = betapdf(mu_hat(n), a, b);
                nll_trial(n) = log(p + eps);
            end

            nll = -nansum(nll_trial,"all");
        end

        %% ===================================================================
        %  MIXTURE MODEL (BASIC RL / RL SIGMA), SUBJECTIVE PE, INTEGRATED
        %  -------------------------------------------------------------------
        %  Integrated analog of nll_mixture_subjectivePE: each sub-model
        %  marginalizes over the same 20-point hypothetical-observation grid
        %  independently (nll_basicRL_integrated_subjectivePE's plain alpha
        %  update vs nll_RLsigma_VOI_subjectivePE's pi-weighted update, both
        %  anchored to the same q_anchor and the same per-trial observation
        %  likelihood weights p_o_given_u), producing two fully-marginalized
        %  beliefs, which are then blended via lambda -- GbAgent.py's
        %  compute_mixture pattern (agents 3/6) applied AFTER each
        %  sub-model's own complete integration, not inside the grid loop
        %  (matching how GbAgent.py's agent==6 mixes q_0_0_ag_4/q_0_0_ag_5
        %  only after update_a4/update_a5 each finish their own full
        %  computation).
        %  Inputs/outputs: identical to nll_mixture_subjectivePE.
        % ====================================================================
        function nll = nll_mixture_integrated_subjectivePE(params, mu_hat, blocks, rewards, condiff)
            alpha = params(1);
            kappa = params(2);
            sigma = params(3);
            lambda = params(4);

            nll_trial = NaN(length(mu_hat),1);
            eps = 1e-9;
            mu_hat(mu_hat <= 0) = eps;
            mu_hat(mu_hat >= 1) = 1 - eps;

            set_o = linspace(-0.1, 0.1, 20);
            agent = Agent();
            agent.sigma = sigma;

            for n = 1:length(mu_hat)
                if n == 1 || blocks(n) ~= blocks(n-1)
                    q_anchor = 0.5;
                else
                    q_anchor = mu_hat(n-1);
                end

                p_o_given_u = fitSlider_ALLmodels.observation_weights(set_o, condiff(n), sigma);

                voi_basicRL = NaN(length(set_o), 1);
                voi_RLsigma = NaN(length(set_o), 1);
                for i = 1:length(set_o)
                    % Belief state at this hypothetical observation --
                    % shared by both sub-models (pi_0/pi_1 don't depend on
                    % which RL update rule is being evaluated).
                    agent.p_s_giv_o(set_o(i));
                    pi_0 = agent.pi_0;
                    pi_1 = agent.pi_1;

                    q_sim_basicRL = q_anchor;
                    q_sim_RLsigma = q_anchor;
                    if pi_0 >= pi_1
                        q_sim_basicRL = q_sim_basicRL + alpha * (rewards(n) - q_sim_basicRL);
                        q_sim_RLsigma = q_sim_RLsigma + pi_0 * alpha * (rewards(n) - q_sim_RLsigma);
                    else
                        q_sim_basicRL = q_sim_basicRL + alpha * ((1 - rewards(n)) - q_sim_basicRL);
                        q_sim_RLsigma = q_sim_RLsigma + pi_1 * alpha * ((1 - rewards(n)) - q_sim_RLsigma);
                    end
                    voi_basicRL(i) = q_sim_basicRL;
                    voi_RLsigma(i) = q_sim_RLsigma;
                end

                % Each sub-model marginalizes fully and independently over
                % the observation grid...
                q_basicRL_transformed = sum(voi_basicRL .* p_o_given_u');
                q_RLsigma_transformed = sum(voi_RLsigma .* p_o_given_u');

                % ...then the two fully-integrated beliefs are blended
                % (GbAgent.py's compute_mixture pattern).
                q_mixed = lambda * q_RLsigma_transformed + (1 - lambda) * q_basicRL_transformed;

                a = q_mixed * kappa;
                b = (1 - q_mixed) * kappa;
                if a <= 0 || b <= 0
                    fprintf('Invalid beta params at trial %d: q=%.3g, kappa=%.3g, a=%.3g, b=%.3g (alpha=%.3g, sigma=%.3g, lambda=%.3g)\n', ...
                        n, q_mixed, kappa, a, b, alpha, sigma, lambda);
                    a = max(eps, a);
                    b = max(eps, b);
                end

                p = betapdf(mu_hat(n), a, b);
                nll_trial(n) = log(p + eps);
            end

            nll = -nansum(nll_trial,"all");
        end

        %% ===================================================================
        %  BASIC RL MODEL - GENERATIVE SIMULATION WITH ECONOMIC CHOICE,
        %  SUBJECTIVE PE, NO INTEGRATION (POINT OBSERVATION) VARIANT
        %  -------------------------------------------------------------------
        %  Like simulate_basicRL_integrated_choice_subjectivePE, except
        %  belief each trial comes from a single point estimate of the
        %  actual observed condiff (normcdf(condiff(n), 0, sigma)) rather
        %  than integrated over a discretized grid of hypothetical
        %  observations -- the fully-generative analog of nll_basicRL_
        %  pointObs_subjectivePE's point-estimate belief (see its header
        %  for the full rationale). No set_o/observation_weights/Agent
        %  grid is needed here, so pi_0/pi_1 (and the resulting action
        %  value v_a0) are each computed once per trial instead of
        %  marginalized over 20 hypothetical observations.
        %  Inputs/outputs: identical to simulate_basicRL_integrated_
        %  choice_subjectivePE.
        % ====================================================================
        function [mu_hat, choice, reward, q_anchor_trace] = simulate_basicRL_pointObs_choice_subjectivePE(params, blocks, state, condiff, beta, p_reward_correct)
            alpha = params(1);
            kappa = params(2);
            sigma = params(3);

            n_trials = length(condiff);
            mu_hat = NaN(n_trials, 1);
            choice = NaN(n_trials, 1);
            reward = NaN(n_trials, 1);
            q_anchor_trace = NaN(n_trials, 1);

            for n = 1:n_trials
                % Anchor for this trial: 0.5 at the start of a new block,
                % otherwise this synthetic subject's own previous
                % simulated report -- NOT a separately-tracked clean
                % belief.
                if n == 1 || blocks(n) ~= blocks(n-1)
                    q_anchor = 0.5;
                else
                    q_anchor = mu_hat(n-1);
                end
                q_anchor_trace(n) = q_anchor;

                % Point-estimate belief from the actual observed condiff
                % (no integration over hypothetical observations) -- same
                % mechanism as nll_basicRL_pointObs_subjectivePE.
                belief_state = normcdf(condiff(n), 0, sigma);
                if condiff(n) < 0
                    pi_0 = 1 - belief_state;
                    pi_1 = belief_state;
                else
                    pi_1 = belief_state;
                    pi_0 = 1 - belief_state;
                end

                % Action value from this trial's single belief (not
                % marginalized over a grid) -> softmax choice.
                v_a0 = (pi_0 - pi_1) * q_anchor + pi_1;
                v_a1 = 1 - v_a0;

                p_choice0 = 1 / (1 + exp(-beta * (v_a0 - v_a1)));
                choice(n) = double(rand >= p_choice0); % 0 w.p. p_choice0, else 1

                % --- Reward: state-action contingency ---
                if choice(n) == state(n)
                    reward(n) = double(rand < p_reward_correct);
                else
                    reward(n) = double(rand < (1 - p_reward_correct));
                end

                recoded_reward = fitSlider_ALLmodels.recode_rewards_choice(reward(n), choice(n));

                % Plain alpha update (not belief-weighted, unlike
                % simulate_RLsigma_pointObs_choice_subjectivePE) -- matches
                % nll_basicRL_pointObs_subjectivePE's update rule.
                q_sim = q_anchor;
                if pi_0 >= pi_1
                    q_sim = q_sim + alpha * (recoded_reward - q_sim);
                else
                    q_sim = q_sim + alpha * ((1 - recoded_reward) - q_sim);
                end

                a = max(q_sim * kappa, 1e-9);
                b = max((1 - q_sim) * kappa, 1e-9);
                mu_hat(n) = betarnd(a, b); % this trial's report -- becomes next trial's q_anchor
            end
        end

        %% ===================================================================
        %  RL SIGMA MODEL - GENERATIVE SIMULATION WITH ECONOMIC CHOICE,
        %  SUBJECTIVE PE, NO INTEGRATION (POINT OBSERVATION) VARIANT
        %  -------------------------------------------------------------------
        %  Like simulate_RLsigma_integrated_choice_subjectivePE, except
        %  belief each trial comes from a single point estimate of the
        %  actual observed condiff, matching nll_RLsigma_subjectivePE --
        %  see simulate_basicRL_pointObs_choice_subjectivePE's header for
        %  the full rationale, including why no set_o grid is needed here.
        %  Belief-weighted learning rate (pi_0/pi_1 scaling alpha),
        %  matching nll_RLsigma_subjectivePE's update rule, unlike
        %  basicRL's plain alpha update.
        %  Inputs/outputs: identical to simulate_basicRL_pointObs_choice_
        %  subjectivePE.
        % ====================================================================
        function [mu_hat, choice, reward, q_anchor_trace] = simulate_RLsigma_pointObs_choice_subjectivePE(params, blocks, state, condiff, beta, p_reward_correct)
            alpha = params(1);
            kappa = params(2);
            sigma = params(3);

            n_trials = length(condiff);
            mu_hat = NaN(n_trials, 1);
            choice = NaN(n_trials, 1);
            reward = NaN(n_trials, 1);
            q_anchor_trace = NaN(n_trials, 1);

            for n = 1:n_trials
                if n == 1 || blocks(n) ~= blocks(n-1)
                    q_anchor = 0.5;
                else
                    q_anchor = mu_hat(n-1);
                end
                q_anchor_trace(n) = q_anchor;

                belief_state = normcdf(condiff(n), 0, sigma);
                if condiff(n) < 0
                    pi_0 = 1 - belief_state;
                    pi_1 = belief_state;
                else
                    pi_1 = belief_state;
                    pi_0 = 1 - belief_state;
                end

                v_a0 = (pi_0 - pi_1) * q_anchor + pi_1;
                v_a1 = 1 - v_a0;

                p_choice0 = 1 / (1 + exp(-beta * (v_a0 - v_a1)));
                choice(n) = double(rand >= p_choice0);

                if choice(n) == state(n)
                    reward(n) = double(rand < p_reward_correct);
                else
                    reward(n) = double(rand < (1 - p_reward_correct));
                end

                recoded_reward = fitSlider_ALLmodels.recode_rewards_choice(reward(n), choice(n));

                % Belief-weighted learning rate (pi_0/pi_1 scaling alpha)
                % -- matches nll_RLsigma_subjectivePE's update rule.
                q_sim = q_anchor;
                if pi_0 >= pi_1
                    q_sim = q_sim + pi_0 * alpha * (recoded_reward - q_sim);
                else
                    q_sim = q_sim + pi_1 * alpha * ((1 - recoded_reward) - q_sim);
                end

                a = max(q_sim * kappa, 1e-9);
                b = max((1 - q_sim) * kappa, 1e-9);
                mu_hat(n) = betarnd(a, b); % this trial's report -- becomes next trial's q_anchor
            end
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
        %  BAYESIAN AGENT MODEL - NO INTEGRATION (POINT OBSERVATION) VARIANT
        %  -------------------------------------------------------------------
        %  Identical to nll_bayesianAgent, except agent.task_agent_analysis
        %  is set to 0 instead of 1. agent.learn()'s task_agent_analysis==1
        %  branch is what routes the belief update through Agent.
        %  integrate_voi -- a 20-point set_o grid (agentvars.m's default
        %  linspace(-0.1,0.1,20)) weighted by normpdf(set_o,o_t,sigma) and
        %  marginalized, mathematically the same grid-integration mechanism
        %  nll_basicRL_integrated/nll_RLsigma_VOI use inline. Flipping the
        %  flag to 0 instead routes learn() through its own built-in
        %  non-integrated path: compute_q(r_t) is called directly against
        %  whatever pi_0/pi_1 agent.p_s_giv_o(agent.o_t) already set from
        %  the single actually-observed condiff(t) a few lines above (no
        %  set_o, no grid, no marginalization) -- the direct analog of
        %  nll_basicRL_pointObs/nll_RLsigma's point-estimate belief. The
        %  earlier p_s_giv_o/compute_valence/softmax choice steps are
        %  unaffected either way -- they're already single-point-observation
        %  computations in both nll_bayesianAgent and here (this model only
        %  ever calls Agent's single-observation p_s_giv_o/compute_valence,
        %  never its grid-integrated decide_e()).
        %  Inputs/outputs: identical to nll_bayesianAgent.
        % ====================================================================
        function nll = nll_bayesianAgent_pointObs(params, mu_hat, data, nBlocks, nTrials, blocks, rewards)
            kappa = params(1);
            sigma = params(2);
            nll_trial = NaN(nTrials,nBlocks);
            eps = 1e-9;
            mu_hat(mu_hat <= 0) = eps;
            mu_hat(mu_hat >= 1) = 1 - eps;
            mu = [];
            uniqueBlocks = unique(blocks);
            rewards = rewards;

            for bl = 1:nBlocks
                agent = Agent();
                agent.task_agent_analysis = 0;
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
                    q_transformed = agent.G;
                    a = q_transformed * kappa;
                    b = (1 - q_transformed) * kappa;
                    if any(a <= 0 | b <= 0)
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
        %  reward-learning model (see the PERCEPTUAL SENSITIVITY (SIGMA)
        %  FITTING section in fitReducedModelSpace_fixedSigma.m).
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
        %  MODEL COMPARISON (AIC/BIC/delta-BIC/bestModel/Bayesian Model
        %  Selection), generalized over an arbitrary number of models.
        %  -------------------------------------------------------------------
        %  Replaces the previously copy-pasted per-condition/per-PE-variant
        %  AIC/BIC/bestModel/spm_BMS blocks in fitReducedModelSpace*.m with a
        %  single reusable computation. Family-wise (VBA_groupBMC) comparison
        %  is NOT included here -- it requires hand-specified family
        %  membership that doesn't generalize across variants, so it stays
        %  inline where it's used (only meaningful for the 3-way objective-PE
        %  comparison).
        %  Inputs:
        %    nll_matrix  - numSubjs x numModels matrix of each subject's
        %                  best-fit NLL under each model
        %    model_names - 1 x numModels cell array of model name strings,
        %                  column-aligned with nll_matrix
        %    num_params  - 1 x numModels vector of free-parameter counts,
        %                  column-aligned with nll_matrix
        %    num_trials  - numSubjs x 1 vector (or scalar, recycled across
        %                  subjects) of trial counts per subject
        %    uniqueID    - numSubjs x 1 subject ID vector
        %  Outputs:
        %    AICBIC_table   - per-subject AIC/BIC/delta-BIC for every model
        %    bestModel_table - per-subject best-fitting model (by delta-BIC),
        %                  plus evidence_gap (delta-BIC of the next-best
        %                  competitor) and decisive (evidence_gap >= 2, the
        %                  Kass & Raftery 1995 "positive evidence" cutoff --
        %                  below this, the race between the top two models
        %                  is too close to call for that subject)
        %    BMS_table      - per-model Bayesian Model Selection probabilities
        %                     (ModelProbability = exp_r, ExceedanceProbability = xp)
        % ====================================================================
        function [AICBIC_table, bestModel_table, BMS_table] = compute_model_comparison(nll_matrix, model_names, num_params, num_trials, uniqueID)
            numSubjs = size(nll_matrix, 1);
            if isscalar(num_trials)
                num_trials = repelem(num_trials, numSubjs);
            end

            AIC = NaN(size(nll_matrix));
            BIC = NaN(size(nll_matrix));
            for n = 1:numSubjs
                [AIC(n,:), BIC(n,:)] = fitSlider_ALLmodels.compute_aic_bic(nll_matrix(n,:), num_params, num_trials(n));
            end

            aic_names = strcat('AIC_', model_names);
            bic_names = strcat('BIC_', model_names);
            deltaBIC_names = strcat('delta_BIC_', model_names);

            AICBIC_table = array2table([AIC, BIC], 'VariableNames', [aic_names, bic_names]);
            AICBIC_table.SubjectID = uniqueID;
            AICBIC_table = movevars(AICBIC_table, 'SubjectID', 'Before', 1);

            delta_BIC = BIC - min(BIC, [], 2);
            for m = 1:numel(model_names)
                AICBIC_table.(deltaBIC_names{m}) = delta_BIC(:, m);
            end

            [~, best_model_idx] = min(abs(delta_BIC), [], 2);

            % Evidence strength: how decisively the nominal winner beats its
            % closest competitor. delta_BIC's winning column is always 0 by
            % construction, so the second-smallest value in each row is the
            % gap to the runner-up -- small gaps mean the "winner" isn't
            % actually distinguishable from the next-best model.
            sorted_delta_BIC = sort(delta_BIC, 2);
            evidence_gap = sorted_delta_BIC(:, 2);
            decisive = evidence_gap >= 2; % Kass & Raftery (1995) "positive evidence" cutoff

            bestModel_table = table();
            bestModel_table.SubjectID = AICBIC_table.SubjectID;
            bestModel_table.best_model = model_names(best_model_idx)'; % model_names is a 1xN row vector, so indexing it returns a row regardless of best_model_idx's shape -- transpose to match the table's Nx1 column height
            bestModel_table.evidence_gap = evidence_gap;
            bestModel_table.decisive = decisive;
            for m = 1:numel(model_names)
                bestModel_table.(deltaBIC_names{m}) = delta_BIC(:, m);
            end

            lme = -0.5 * BIC;
            [~, exp_r, xp] = spm_BMS(lme, 1e6, 1, 0, 1, []);
            BMS_table = table(model_names', exp_r', xp', 'VariableNames', {'Model', 'ModelProbability', 'ExceedanceProbability'});
        end

        %% ===================================================================
        %  MODEL COMPARISON PLOTS
        %  -------------------------------------------------------------------
        %  Visualizes the output of compute_model_comparison: proportion of
        %  subjects best described by each model, Bayesian Model Selection
        %  probabilities/exceedance probabilities, and a combined plot with
        %  the winning model highlighted. Generalized over an arbitrary
        %  number of models; does not include family-wise comparison plots
        %  (see compute_model_comparison for why that's excluded).
        %  Inputs:
        %    AICBIC_table, bestModel_table, BMS_table - outputs of
        %                  compute_model_comparison
        %    model_names - 1 x numModels cell array, matching the names used
        %                  to compute the tables above
        %    title_suffix - string appended to plot titles (e.g. 'Both
        %                  condition, Sigma Fixed')
        % ====================================================================
        function plot_model_comparison(AICBIC_table, bestModel_table, BMS_table, model_names, title_suffix)
            numModels = numel(model_names);
            colors = lines(numModels);

            % Proportion of subjects best described by each model. Only
            % subjects whose winning model beats the runner-up by at least
            % the Kass & Raftery (1995) "positive evidence" cutoff
            % (bestModel_table.decisive, delta-BIC gap >= 2) are credited to
            % that model; subjects below that cutoff go in a separate
            % "Ambiguous" bar, since the race was too close to call rather
            % than a real win for whichever model happened to edge ahead.
            best_model_idx = NaN(height(bestModel_table), 1);
            for m = 1:numModels
                best_model_idx(strcmp(bestModel_table.best_model, model_names{m}) & bestModel_table.decisive) = m;
            end
            model_counts = histcounts(best_model_idx, 1:numModels+1);
            ambiguous_count = sum(~bestModel_table.decisive);
            bar_labels = [model_names, {'Ambiguous'}];
            bar_counts = [model_counts, ambiguous_count];
            bar_proportions = bar_counts / height(bestModel_table);
            numBars = numModels + 1;

            figure;
            h = bar(1:numBars, bar_proportions, 'FaceColor', 'flat');
            for i = 1:numModels
                h.CData(i,:) = colors(i,:);
            end
            h.CData(numBars,:) = [0.6 0.6 0.6]; % gray for the ambiguous bar
            h.FaceAlpha = 0.4;
            set(gca, 'XTickLabel', bar_labels);
            xlabel('Model');
            ylabel('Proportion of Subjects');
            title(sprintf('Proportion of Subjects Best Described by Each Model (%s)', title_suffix));
            ylim([0, 1]);
            for i = 1:numBars
                text(i, bar_proportions(i) + 0.02, sprintf('%.2f', bar_proportions(i)), ...
                     'HorizontalAlignment', 'center', 'FontWeight', 'bold');
            end
            grid on;
            set(gca, 'GridAlpha', 0.3);

            fprintf('\nModel Selection Results (%s):\n', title_suffix);
            fprintf('------------------------\n');
            for i = 1:numBars
                fprintf('%s: %.1f%% of subjects (%d/%d)\n', ...
                        bar_labels{i}, bar_proportions(i)*100, bar_counts(i), height(bestModel_table));
            end

            % Bayesian Model Selection: probability / exceedance bar plots
            exp_r = BMS_table.ModelProbability';
            xp = BMS_table.ExceedanceProbability';

            fprintf('\n=== BAYESIAN MODEL SELECTION RESULTS (%s) ===\n\n', title_suffix);
            fprintf('Model Probabilities (exp_r):\n');
            for i = 1:numModels
                fprintf('  %s: %.4f\n', model_names{i}, exp_r(i));
            end
            fprintf('\nExceedance Probabilities (xp):\n');
            for i = 1:numModels
                fprintf('  %s: %.4f\n', model_names{i}, xp(i));
            end
            [~, winning_model_idx] = max(exp_r);
            fprintf('\nWinning Model: %s (exp_r = %.4f, xp = %.4f)\n', ...
                model_names{winning_model_idx}, exp_r(winning_model_idx), xp(winning_model_idx));

            figure('Position', [100, 100, 1000, 500]);
            subplot(1, 2, 1);
            b1 = bar(exp_r, 'FaceColor', 'flat');
            b1.CData = colors;
            b1.FaceAlpha = 0.5;
            set(gca, 'XTickLabel', model_names, 'XTickLabelRotation', 45);
            ylabel('Model Probability (exp_r)', 'FontSize', 12, 'FontWeight', 'bold');
            title('Model Probabilities', 'FontSize', 14, 'FontWeight', 'bold');
            ylim([0, max(exp_r) * 1.1]);
            grid on;
            grid minor;
            for i = 1:numModels
                text(i, exp_r(i) + 0.01, sprintf('%.3f', exp_r(i)), ...
                    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
            end

            subplot(1, 2, 2);
            b2 = bar(xp, 'FaceColor', 'flat');
            b2.CData = colors;
            b2.FaceAlpha = 0.5;
            set(gca, 'XTickLabel', model_names, 'XTickLabelRotation', 45);
            ylabel('Exceedance Probability (xp)', 'FontSize', 12, 'FontWeight', 'bold');
            title('Exceedance Probabilities', 'FontSize', 14, 'FontWeight', 'bold');
            ylim([0, 1.05]);
            grid on;
            grid minor;
            for i = 1:numModels
                if xp(i) < 0.001
                    text(i, xp(i) + 0.02, '< 0.001', ...
                        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
                else
                    text(i, xp(i) + 0.02, sprintf('%.3f', xp(i)), ...
                        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
                end
            end
            sgtitle(sprintf('Bayesian Model Selection Results (%s)', title_suffix), 'FontSize', 16, 'FontWeight', 'bold');

            % Combined plot with winning model highlighted
            [~, winner_idx] = max(xp);
            figure('Position', [150, 150, 800, 600]);
            x_pos = 1:numModels;
            b = bar(x_pos, [exp_r; xp]', 'grouped');
            b(1).FaceColor = [0.3 0.5 0.8]; % Model probabilities
            b(2).FaceColor = [0.8 0.4 0.3]; % Exceedance probabilities
            b(1).CData(winner_idx, :) = [0.1 0.7 0.1]; % Green for winner
            b(2).CData(winner_idx, :) = [0.1 0.7 0.1]; % Green for winner
            set(gca, 'XTickLabel', model_names, 'XTickLabelRotation', 45);
            ylabel('Probability', 'FontSize', 12, 'FontWeight', 'bold');
            title(sprintf('Model Comparison: Probabilities and Exceedance (%s)', title_suffix), 'FontSize', 14, 'FontWeight', 'bold');
            legend({'Model Probability (exp_r)', 'Exceedance Probability (xp)'}, ...
                'Location', 'northeast', 'FontSize', 11);
            grid on;
            ylim([0, 1.05]);
            text(winner_idx, max([exp_r(winner_idx), xp(winner_idx)]) + 0.1, ...
                sprintf('WINNER\n%s', model_names{winner_idx}), ...
                'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', ...
                'Color', [0.1 0.7 0.1]);
        end

        %% ===================================================================
        %  PLOT FITTED PARAMETERS
        %  -------------------------------------------------------------------
        %  Integrated from plotParams_reducedMS_fixedSigma.m's local function
        %  of the same purpose, generalized over an arbitrary number of
        %  models (rather than hardcoded to 3) so the same method serves
        %  both the 3-way objective-PE comparison (basicRL/RLsigma/
        %  BayesianAgent) and the 2-way subjective-PE comparison (basicRL_
        %  subjectivePE/RLsigma_subjectivePE), for either condition.
        %  Plots each model's fitted alpha/kappa (where present -- e.g.
        %  BayesianAgent has no alpha) and sigma (fixed or fitted,
        %  depending on the caller's pipeline -- see sigma_title) as
        %  bar+dot+SEM panels, plus summed AIC, summed BIC, and mean
        %  delta-BIC across models.
        %  Inputs:
        %    params_structs - 1 x numModels cell array of structs, each
        %                  with .kappa and .sigma fields, and .alpha for
        %                  models that have one
        %    model_names  - 1 x numModels cell array, matching
        %                  params_structs and AICBIC_table's column-name
        %                  suffixes (AIC_<name>, BIC_<name>,
        %                  delta_BIC_<name>)
        %    AICBIC_table - output of compute_model_comparison for the
        %                  same models/condition
        %    title_suffix - string appended to plot titles (e.g. 'Both
        %                  condition, Sigma Fixed')
        %    sigma_title  - (optional) title for the sigma subplot --
        %                  callers differ on whether sigma is fixed
        %                  (fitReducedModelSpace_fixedSigma.m: 'sigma
        %                  (fixed, from perceptual-choice fit)') or fitted
        %                  jointly with alpha/kappa
        %                  (fitReducedModelSpace.m: 'sigma (fitted)').
        %                  Defaults to the fixed-sigma phrasing if omitted,
        %                  matching this function's original behavior.
        %  Outputs:
        %    bic_ax       - axes handle of the "Sum of BIC" bar plot, so
        %                  callers can match its y-limits across multiple
        %                  calls (e.g. objective vs subjective PE, Both vs
        %                  Perceptual) for visual comparability
        % ====================================================================
        function bic_ax = plot_fitted_params(params_structs, model_names, AICBIC_table, title_suffix, sigma_title)
            if nargin < 5
                sigma_title = 'sigma (fixed, from perceptual-choice fit)';
            end
            numModels = numel(model_names);
            n = length(params_structs{1}.kappa); % number of subjects
            all_facecolors = lines(numModels);

            has_alpha = cellfun(@(p) isfield(p, 'alpha'), params_structs);
            alpha_all = NaN(n, numModels);
            kappa_all = NaN(n, numModels);
            sigma_all = NaN(n, numModels);
            for m = 1:numModels
                if has_alpha(m)
                    alpha_all(:,m) = params_structs{m}.alpha;
                end
                kappa_all(:,m) = params_structs{m}.kappa;
                sigma_all(:,m) = params_structs{m}.sigma;
            end

            param_data = {alpha_all, kappa_all, sigma_all};
            param_names = {'alpha', 'kappa', 'sigma'};
            param_titles = {'alpha (fitted)', 'kappa (fitted)', sigma_title};
            for p = 1:length(param_names)
                y = param_data{p};

                % Drop columns that are all-NaN (e.g. alpha for BayesianAgent,
                % or every column if no model in this comparison has this
                % parameter at all).
                valid_cols = ~all(isnan(y),1);
                if ~any(valid_cols)
                    continue
                end
                y = y(:,valid_cols);
                this_model_names = model_names(valid_cols);
                facecolors = all_facecolors(valid_cols, :);

                mean_all = nanmean(y);
                SEM_all = nanstd(y)./sqrt(sum(~isnan(y)));

                title_name = sprintf('%s across models (%s)', param_titles{p}, title_suffix);

                figure;
                bar_plots_pval(y, mean_all, SEM_all, n, 1, size(y,2), this_model_names, 1, ...
                    {param_names{p}}, title_name, '', param_names{p}, 0, 1, 20, 1, 12, 1, ...
                    'Arial', 1, facecolors);
            end

            %% AIC/BIC/delta-BIC summary
            AIC = NaN(n, numModels);
            BIC = NaN(n, numModels);
            delta_BIC = NaN(n, numModels);
            for m = 1:numModels
                AIC(:,m) = AICBIC_table.(['AIC_' model_names{m}]);
                BIC(:,m) = AICBIC_table.(['BIC_' model_names{m}]);
                delta_BIC(:,m) = AICBIC_table.(['delta_BIC_' model_names{m}]);
            end

            sum_AIC = nansum(AIC, 1);
            sum_BIC = nansum(BIC, 1);

            figure;
            h_aic = bar(1:numModels, sum_AIC, 'FaceColor', 'flat');
            set(gca, 'XTickLabel', model_names);
            title(sprintf('Sum of AIC across all subjects (%s)', title_suffix));
            ylabel('Sum AIC');
            xlabel('Model');
            box off;
            set(gca, 'FontSize', 12, 'FontName', 'Arial', 'LineWidth', 1);
            for i = 1:numModels
                h_aic.CData(i,:) = all_facecolors(i,:);
            end
            h_aic.FaceAlpha = 0.5;

            figure;
            h_bic = bar(1:numModels, sum_BIC, 'FaceColor', 'flat');
            bic_ax = gca; % returned so callers can match this axes' y-limits across multiple calls/figures
            set(bic_ax, 'XTickLabel', model_names);
            title(bic_ax, sprintf('Sum of BIC across all subjects (%s)', title_suffix));
            ylabel(bic_ax, 'Sum BIC');
            xlabel(bic_ax, 'Model');
            box(bic_ax, 'off');
            set(bic_ax, 'FontSize', 12, 'FontName', 'Arial', 'LineWidth', 1);
            for i = 1:numModels
                h_bic.CData(i,:) = all_facecolors(i,:);
            end
            h_bic.FaceAlpha = 0.5;

            mean_delta_BIC = nanmean(delta_BIC, 1);
            SEM_delta_BIC = nanstd(delta_BIC, 1) ./ sqrt(sum(~isnan(delta_BIC), 1));

            figure;
            bar_plots_pval(delta_BIC, mean_delta_BIC, SEM_delta_BIC, n, 1, numModels, model_names, 1, ...
                {'Delta BIC'}, sprintf('Mean Delta BIC across models (%s)', title_suffix), ...
                '', 'Delta BIC', 0, 1, 20, 1, 12, 1, 'Arial', 1, all_facecolors);
        end

        %% ===================================================================
        %  SCATTER + UNITY LINE + SPEARMAN'S RHO
        %  -------------------------------------------------------------------
        %  Generic single-panel scatter helper: plots y vs x into whichever
        %  axes are already current (a subplot set up by the caller),
        %  overlays a y = x unity reference line (both axes forced to the
        %  same shared range so the line is a genuine diagonal rather than
        %  an artifact of mismatched auto-scaled limits), and annotates the
        %  title with Spearman's rho and its p-value. Promoted out of
        %  fitReducedModelSpace_fixedSigma.m's "OBJECTIVE VS SUBJECTIVE PE
        %  PARAMETER COMPARISON" section so fitReducedModelSpace.m's
        %  equivalent section can reuse it without duplicating the function.
        %  Inputs:
        %    x, y                   - vectors to scatter (e.g. objective-PE
        %                           vs subjective-PE fitted parameter)
        %    xlabel_str, ylabel_str - axis labels
        %    title_str              - title prefix; rho/p are appended on a
        %                           second line
        % ====================================================================
        function plot_scatter_spearman(x, y, xlabel_str, ylabel_str, title_str)
            scatter(x, y, 25, 'filled', 'MarkerFaceAlpha', 0.6);
            hold on;
            lims = [min([xlim, ylim]), max([xlim, ylim])];
            plot(lims, lims, 'k--', 'LineWidth', 1);
            xlim(lims);
            ylim(lims);
            [rho, pval] = corr(x, y, 'rows', 'complete', 'Type', 'Spearman');
            if pval < 0.001
                p_str = 'p < .001';
            else
                p_str = sprintf('p = %.3f', pval);
            end
            xlabel(xlabel_str);
            ylabel(ylabel_str);
            title(sprintf('%s\n\\rho = %.2f, %s', title_str, rho, p_str));
            box off;
        end

        %% ===================================================================
        %  FAMILY-WISE MODEL COMPARISON (VBA_groupBMC, with manual fallback)
        %  -------------------------------------------------------------------
        %  Groups models into families (e.g. "Basic RL" vs "LR Modulation")
        %  and runs Bayesian Model Selection at both the family and
        %  within-family model level. Separate from compute_model_comparison
        %  because family membership is an analysis-specific choice the
        %  caller must supply -- it doesn't generalize automatically the way
        %  a plain N-model AIC/BIC comparison does.
        %  Inputs:
        %    lme          - numSubjs x numModels log model evidence matrix
        %                   (e.g. -0.5 * BIC)
        %    families     - 1 x numFamilies cell array; families{f} is a
        %                   vector of model indices (columns of lme)
        %                   belonging to family f
        %    family_names - 1 x numFamilies cell array of family name strings
        %    model_names  - 1 x numModels cell array of model name strings,
        %                   column-aligned with lme
        %  Outputs:
        %    results_summary - struct with family/model names, probabilities,
        %                   and exceedance probabilities (ready to save)
        %    posterior, out  - raw VBA_groupBMC (or manual_family_BMS)
        %                   outputs, kept for saving alongside results_summary
        % ====================================================================
        function [results_summary, posterior, out] = compute_family_model_comparison(lme, families, family_names, model_names)
            options_vba = struct();
            options_vba.families = families;
            options_vba.verbose = 1;
            options_vba.DisplayWin = 0;

            try
                [posterior, out] = VBA_groupBMC(lme', options_vba);
                fprintf('VBA_groupBMC completed successfully!\n');
                vba_success = true;
            catch ME
                fprintf('VBA_groupBMC failed with error: %s\n', ME.message);
                fprintf('Using manual implementation instead...\n');
                [posterior, out] = fitSlider_ALLmodels.manual_family_BMS(lme, families, family_names);
                vba_success = false;
            end

            if vba_success
                family_prob = out.families.Ef;
                family_xp = out.families.ep;
                model_prob = out.Ef;
                model_xp = out.ep;
            else
                family_prob = posterior.r;
                family_xp = out.families.ep;
                model_prob = posterior.r_model;
                model_xp = out.ep;
            end

            results_summary = struct();
            results_summary.family_names = family_names;
            results_summary.families = families;
            results_summary.family_probabilities = family_prob;
            results_summary.family_exceedance_probabilities = family_xp;
            results_summary.model_names = model_names;
            results_summary.model_probabilities = model_prob;
            results_summary.model_exceedance_probabilities = model_xp;
            results_summary.vba_success = vba_success;
        end

        %% ===================================================================
        %  FAMILY-WISE MODEL COMPARISON FALLBACK (manual implementation)
        %  -------------------------------------------------------------------
        %  Used by compute_family_model_comparison when VBA_groupBMC isn't
        %  available / errors. Approximates family- and model-level
        %  posteriors via log-sum-exp pooling within families and spm_BMS
        %  (or a softmax fallback if spm_BMS itself is unavailable).
        % ====================================================================
        function [posterior, out] = manual_family_BMS(lme, families, family_names)
            fprintf('Implementing family-wise BMS manually...\n');

            [n_subjects, n_models] = size(lme);
            n_families = length(families);

            % Step 1: Compute family-level log evidence
            family_lme = zeros(n_subjects, n_families);

            for f = 1:n_families
                models_in_family = families{f};
                % For each subject, compute log sum exp of models in family
                for s = 1:n_subjects
                    family_evidence = lme(s, models_in_family);
                    % Numerical stable log-sum-exp
                    max_ev = max(family_evidence);
                    family_lme(s, f) = max_ev + log(sum(exp(family_evidence - max_ev)));
                end
            end

            % Step 2: Run BMS on family-level evidence using SPM functions
            try
                [alpha_fam, exp_r_fam, xp_fam] = spm_BMS(family_lme, 1e6, 0, 0, 1, ones(1, n_families));
            catch
                % Fallback if SPM not available
                fprintf('SPM not available, using simplified BMS...\n');
                % Simple softmax over mean log evidence
                mean_family_lme = mean(family_lme, 1);
                exp_r_fam = exp(mean_family_lme - max(mean_family_lme));
                exp_r_fam = exp_r_fam / sum(exp_r_fam);
                xp_fam = exp_r_fam;  % Simplified exceedance probabilities
            end

            % Step 3: Compute model-level probabilities within families
            model_prob = zeros(1, n_models);
            for f = 1:n_families
                models_in_family = families{f};
                if length(models_in_family) == 1
                    % Single model in family
                    model_prob(models_in_family) = exp_r_fam(f);
                else
                    % Multiple models in family - run BMS within family
                    family_models_lme = lme(:, models_in_family);
                    try
                        [~, exp_r_within, ~] = spm_BMS(family_models_lme, 1e6, 0, 0, 1, []);
                    catch
                        % Fallback method
                        mean_model_lme = mean(family_models_lme, 1);
                        exp_r_within = exp(mean_model_lme - max(mean_model_lme));
                        exp_r_within = exp_r_within / sum(exp_r_within);
                    end
                    % Weight by family probability
                    model_prob(models_in_family) = exp_r_fam(f) * exp_r_within;
                end
            end

            % Step 4: Format output
            posterior.r = exp_r_fam;           % Family probabilities
            posterior.r_model = model_prob;     % Model probabilities

            out.families.ep = xp_fam;          % Family exceedance probabilities
            out.ep = zeros(1, n_models);       % Model exceedance probabilities

            % Approximate model exceedance probabilities
            for m = 1:n_models
                out.ep(m) = model_prob(m);  % Simplified approximation
            end

            fprintf('Manual family BMS completed.\n');
        end

        %% ===================================================================
        %  FAMILY-WISE MODEL COMPARISON PLOTS
        %  -------------------------------------------------------------------
        %  Visualizes compute_family_model_comparison's results_summary:
        %  family probabilities/exceedance probabilities and individual model
        %  probabilities (copper colormap, 3-panel figure), a stacked bar
        %  showing each model's relative contribution within its family, and
        %  a printed statistical summary (winning family/model, evidence
        %  strength). Generalized over an arbitrary number of families.
        % ====================================================================
        function plot_family_model_comparison(results_summary, title_suffix)
            family_names = results_summary.family_names;
            families = results_summary.families;
            family_prob = results_summary.family_probabilities;
            family_xp = results_summary.family_exceedance_probabilities;
            model_names = results_summary.model_names;
            model_prob = results_summary.model_probabilities;
            model_xp = results_summary.model_exceedance_probabilities;

            n_models = numel(model_names);
            num_families = numel(families);

            label = sprintf('(%s)', title_suffix);

            % Display family results
            for i = 1:num_families
                fprintf('Family %d (%s):\n', i, family_names{i});
                fprintf('  Probability: %.3f\n', family_prob(i));
                fprintf('  Exceedance Probability: %.3f\n', family_xp(i));
                fprintf('  Models included: %s\n', mat2str(families{i}));
                fprintf('\n');
            end

            % Display model results
            fprintf('=== MODEL-LEVEL RESULTS %s ===\n', label);
            for i = 1:n_models
                fprintf('Model %d (%s): Prob=%.3f, XP=%.3f\n', i, model_names{i}, model_prob(i), model_xp(i));
            end

            %% Enhanced Visualization with Copper Colormap
            copper_colors = copper(num_families + n_models);
            dark_gray = [0.3, 0.3, 0.3];

            figure('Position', [100, 100, 1400, 500], 'Color', 'white');

            % Plot 1: Family Probabilities
            subplot(1, 3, 1);
            h1 = bar(family_prob, 'FaceColor', 'flat', 'EdgeColor', 'none', 'BarWidth', 0.6, 'FaceAlpha', 0.3);
            for f = 1:num_families
                h1.CData(f,:) = copper_colors(f, 1:3);
            end
            ylim([0, 1]);
            ylabel('Family Probability', 'FontSize', 12, 'FontWeight', 'bold');
            title(sprintf('Family Probabilities %s', label), 'FontSize', 14, 'FontWeight', 'normal', 'Color', dark_gray);
            set(gca, 'XTickLabel', family_names, 'FontSize', 11, 'Box', 'off', 'LineWidth', 1.2);
            grid off;
            for i = 1:length(family_prob)
                text(i, family_prob(i) + 0.03, sprintf('%.3f', family_prob(i)), ...
                     'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
                     'FontSize', 11, 'Color', dark_gray);
            end
            ax1 = gca;
            ax1.Color = [0.98, 0.98, 0.98];

            % Plot 2: Family Exceedance Probabilities
            subplot(1, 3, 2);
            h2 = bar(family_xp, 'FaceColor', 'flat', 'EdgeColor', 'none', 'BarWidth', 0.6, 'FaceAlpha', 0.3);
            for f = 1:num_families
                h2.CData(f,:) = copper_colors(f, 1:3) * 0.8;
            end
            ylim([0, 1]);
            ylabel('Exceedance Probability', 'FontSize', 12, 'FontWeight', 'bold');
            title(sprintf('Family Exceedance Probabilities %s', label), 'FontSize', 14, 'FontWeight', 'normal', 'Color', dark_gray);
            set(gca, 'XTickLabel', family_names, 'FontSize', 11, 'Box', 'off', 'LineWidth', 1.2);
            grid off;
            for i = 1:length(family_xp)
                text(i, family_xp(i) + 0.03, sprintf('%.3f', family_xp(i)), ...
                     'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
                     'FontSize', 11, 'Color', dark_gray);
            end
            ax2 = gca;
            ax2.Color = [0.98, 0.98, 0.98];

            % Plot 3: Individual Model Probabilities
            subplot(1, 3, 3);
            model_copper_colors = copper(n_models);
            bar_colors = model_copper_colors(:, 1:3);
            h3 = bar(1:n_models, model_prob, 'FaceColor', 'flat', 'EdgeColor', 'none', 'BarWidth', 0.7, 'FaceAlpha', 0.3);
            h3.CData = bar_colors;
            ylim([0, max(model_prob) * 1.15]);
            xlabel('Model', 'FontSize', 12, 'FontWeight', 'bold');
            ylabel('Model Probability', 'FontSize', 12, 'FontWeight', 'bold');
            title(sprintf('Individual Model Probabilities %s', label), 'FontSize', 14, 'FontWeight', 'normal', 'Color', dark_gray);
            set(gca, 'XTick', 1:n_models, 'XTickLabel', model_names, ...
                'XTickLabelRotation', 45, 'FontSize', 10, 'Box', 'off', 'LineWidth', 1.2);
            grid off;
            for i = 1:n_models
                if model_prob(i) > 0.01
                    text(i, model_prob(i) + max(model_prob) * 0.02, sprintf('%.3f', model_prob(i)), ...
                         'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
                         'FontSize', 9, 'Color', dark_gray);
                end
            end
            ax3 = gca;
            ax3.Color = [0.98, 0.98, 0.98];

            %% Within-family stacked bar (relative model contribution)
            if num_families == 1
                family_colors = [0.2 0.6 0.9];
            elseif num_families == 2
                family_colors = [
                    0.0000 0.4470 0.7410;
                    0.9 0.3250 0.0980
                ];
            else
                family_colors = parula(num_families);
            end

            fprintf('\n=== WITHIN-FAMILY MODEL CONTRIBUTIONS %s ===\n', label);
            figure;
            hold on;
            handles = gobjects(0);
            labels = {};
            for f = 1:num_families
                models_in_family = families{f};
                family_model_probs = model_prob(models_in_family);

                fprintf('\nFamily %d (%s):\n', f, family_names{f});
                if sum(family_model_probs) > 0
                    rel = family_model_probs / sum(family_model_probs);
                    for m = 1:length(models_in_family)
                        model_idx = models_in_family(m);
                        fprintf('  %s (Model %d): %.3f (%.1f%% of family)\n', ...
                            model_names{model_idx}, model_idx, family_model_probs(m), rel(m) * 100);
                    end
                else
                    rel = zeros(size(family_model_probs));
                end

                n = numel(rel);
                shades = linspace(0.35, 1.00, n)';
                base = family_colors(min(f,size(family_colors,1)), :);
                model_colors = bsxfun(@times, shades, base);

                b = bar(f, rel, 0.6, 'stacked');
                for i = 1:numel(b)
                    b(i).FaceColor = model_colors(i,:);
                    b(i).EdgeColor = 'none';
                    model_idx = models_in_family(i);
                    handles(end+1) = b(i); %#ok<SAGROW>
                    labels{end+1} = sprintf('%s (Family: %s)', model_names{model_idx}, family_names{f}); %#ok<SAGROW>
                end

                cum = 0;
                for i = 1:n
                    h = rel(i);
                    if h > 0
                        yc = cum + h/2;
                        rgb = model_colors(i,:);
                        lum = 0.299*rgb(1) + 0.587*rgb(2) + 0.114*rgb(3);
                        txtc = 'w';
                        if lum > 0.7, txtc = 'k'; end
                        text(f, yc, sprintf('%.1f%%', h*100), ...
                             'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
                             'FontSize',9, 'Color', txtc);
                    end
                    cum = cum + h;
                end
            end
            xlim([0.5, num_families+0.5]);
            set(gca, 'XTick', 1:num_families, 'XTickLabel', family_names, 'FontSize', 12);
            ylabel('Relative Model Contribution (within family)');
            title(sprintf('Within-Family Model Contributions (stacked) %s', label));
            ylim([0 1]);
            grid on; box on;
            legend(handles, labels, 'Location', 'eastoutside');
            hold off;

            %% Statistical Summary
            fprintf('\n=== STATISTICAL SUMMARY %s ===\n', label);
            fprintf('Winning Family: %s (Prob = %.3f, XP = %.3f)\n', ...
                family_names{find(family_prob == max(family_prob), 1)}, ...
                max(family_prob), max(family_xp));
            fprintf('Winning Model: %s (Prob = %.3f, XP = %.3f)\n', ...
                model_names{find(model_prob == max(model_prob), 1)}, ...
                max(model_prob), max(model_xp));

            if max(family_xp) > 0.95
                evidence_strength = 'Very Strong';
            elseif max(family_xp) > 0.90
                evidence_strength = 'Strong';
            elseif max(family_xp) > 0.75
                evidence_strength = 'Moderate';
            else
                evidence_strength = 'Weak';
            end
            fprintf('Evidence Strength: %s (XP = %.3f)\n', evidence_strength, max(family_xp));
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
        %                          the perceptual-sigma fit's needs: warm-up
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
