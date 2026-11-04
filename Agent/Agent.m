classdef Agent < agentvars
    % AGENT class simulates belief states, choices and update contingency
    % parameter based on task variables.


    properties
        o_t = NaN % current observation of contrast difference
        a_t = NaN % current economic decision
        r_t = NaN % current reward
        c_t = NaN % uniform prior distribution over mu
        pi_0 = NaN % belief state in favor of s_t = 0
        pi_1 = NaN  % belief state in favor of s_t = 1
        E_mu_t = 0.5 % expected value
        v_a_0 = NaN % valence of action = 0
        v_a_1 = NaN % valence of action = 1
        v_a_t = NaN % vector of valences
        p_a_t = NaN % vector with choice probabilities
        t = NaN % degree of polynomial
        p_o_giv_u  % probabilities of observations given contrast differences
        p_o_giv_u_norm  % normalised probabilities
        C = NaN % polynomial coefficients
        q_0 = NaN % fraction for polynomial update
        q_1 = NaN
        d = NaN % temporary, trial-specific polynomial coefficients
        G = NaN % Gamma factor
        d_t = NaN % perceptual decision
        product = ones(1,100)
        mu = repelem(1,100)
        p_mu = linspace(0, 1, 100)
        mu_for_ev
        u = NaN
        v = NaN
        w = NaN
        p_d_0
        p_d_t
        s0
        s1
        dist
        q_0_1
        q_1_0
        q_1_1
        confirmation_bias = 0 % confirmation bias parameter (0 = no bias, >0 = bias)
        outcome_confirms_choice = false % whether current outcome confirms the choice
        bias_scaling = 1.0; % default no bias
        confirmBias = 0.5; % default no bias
    end

    methods

        function obj=Agent
            % The contructor methods initialises all other properties of
            % the class that are computed based on exisitng static properties of
            % the class.

            obj.p_o_giv_u = zeros(1,length(obj.set_o)); % probabilities of observations given contrast differences
            obj.p_o_giv_u_norm = zeros(1,length(obj.set_o)); % normalised probabilities of observations
            obj.mu_for_ev = obj.mu; % contingency parameter
            obj.mu_for_ev = obj.mu_for_ev/sum(obj.mu_for_ev); % expected value
            obj.c_t = obj.c0; % prior distribution over mu
        end

        function getSigma(obj,varargin)
            % function getSigma re-initialises the sigma property according to
            % the experimental condition.
            %
            % INPUT:
            %   obj: current object
            if nargin > 1
                obj.sigma =  cell2mat(varargin(1));
            else
                if obj.condition == 1 || obj.condition == 2 % condition dependent sensitivity parameter for agent
                    obj.sigma = 0.03; %0.03;
                elseif obj.condition == 3
                    obj.sigma = 0.0001;
                elseif obj.condition == 5 % high PU
                    obj.sigma = 0.3;
                end
            end
        end


        function getBeta(obj,trueBeta)
            % re-initialises the beta property according to
            % the experimental condition.
            %
            % INPUT:
            %   obj: current object
            obj.beta = trueBeta(1);
        end

        function observation_sample(obj,c_t)
            % function observation_sample computes the contrast difference dependent
            % observation (obj.o_t) returns observation that is sampled from a normal
            % distribution with presented contrast difference as mean
            % and perceptual sensitivity as variance.
            %
            % INPUT:
            %   c_t: task generated contrast difference
            %   obj: current object

            % Noisy-o_t fix (scoped to task_agent_analysis==1 &&
            % eval_ana==0), preserved here in case it's wanted again --
            % see conversation for the full derivation of the overshoot
            % bug this was fixing, and why it was reverted below: once
            % state_0/state_1 reporting was itself fixed (see
            % integrate_voi's pi_matrix marginalization), the exact-c_t
            % design is being re-run deliberately to study its (corrected)
            % belief-state logging, not to avoid the overshoot.
            % if obj.task_agent_analysis == true && obj.eval_ana == 0
            %     obj.o_t = normrnd(c_t,obj.sigma); % genuinely noisy observation -- fixes the with-integration, eval_ana=0 overshoot
            % elseif obj.task_agent_analysis == true
            %     obj.o_t = c_t; % eval_ana==1: original exact-observation behavior, unchanged
            % else
            %     obj.o_t = normrnd(c_t,obj.sigma); % no-integration: unchanged, always noisy
            % end

            % Reverted to the original behavior for BOTH eval_ana settings:
            % task_agent_analysis==true means o_t is the exact presented
            % contrast difference (no noise); task_agent_analysis==false
            % means a genuinely noisy observation is sampled.
            if obj.task_agent_analysis == true
                obj.o_t = c_t; % for task_agent_data_analysis, o_t is the same as the presented contrast difference
            else
                obj.o_t = normrnd(c_t,obj.sigma); % for simulations, the observation is sampled
            end
        end

        function p_s_giv_o(obj, o_t)
            % function p_s_giv_o computes belief state (obj.pi_0, obj.pi_1) given observation of
            % contrast difference.
            %
            % INPUT:
            %   obj: current object
            %   o_t: observed contrast differnece

            % COMPUTE CUMULATIVE DISTRIBUTION FUNCTIONS
            obj.u = normcdf(0, o_t, obj.sigma);
            obj.v = normcdf(-obj.kappa_max, o_t, obj.sigma);
            obj.w = normcdf(obj.kappa_max, o_t, obj.sigma);

            % COMPUTE BELIEF STATES
            % Numerical safety guard: when o_t falls many sigma beyond
            % +-kappa_max (more likely with small sigma, e.g. sigma<=0.02),
            % u/v/w all saturate to the same extreme (0 or 1), so w-v
            % underflows to exactly 0 and (u-v)/(w-v) evaluates to NaN.
            % In that regime the observation is unambiguous: fall back to
            % a confident belief based on which side of 0 it falls on
            % (matching what the ratio converges to analytically as
            % sigma -> 0, rather than leaving pi_0/pi_1 undefined).
            if (obj.w - obj.v) == 0
                if o_t < 0
                    obj.pi_0 = 1;
                    obj.pi_1 = 0;
                else
                    obj.pi_0 = 0;
                    obj.pi_1 = 1;
                end
            else
                obj.pi_0 = (obj.u - obj.v) / (obj.w - obj.v);
                obj.pi_1 = (obj.w - obj.u) / (obj.w - obj.v);
            end
        end

        function p_s_giv_oConfirm(obj, o_t, sigma)
            % function p_s_giv_o computes belief state (obj.pi_0, obj.pi_1) given observation of
            % contrast difference.
            %
            % INPUT:
            %   obj: current object
            %   o_t: observed contrast differnece

            % COMPUTE CUMULATIVE DISTRIBUTION FUNCTIONS
            obj.u = normcdf(0, o_t, sigma);
            obj.v = normcdf(-obj.kappa_max, o_t, sigma);
            obj.w = normcdf(obj.kappa_max, o_t, sigma);

            % COMPUTE BELIEF STATES
            % Numerical safety guard -- see p_s_giv_o's identical guard
            % above for the full rationale.
            if (obj.w - obj.v) == 0
                if o_t < 0
                    obj.pi_0 = 1;
                    obj.pi_1 = 0;
                else
                    obj.pi_0 = 0;
                    obj.pi_1 = 1;
                end
            else
                obj.pi_0 = (obj.u - obj.v) / (obj.w - obj.v);
                obj.pi_1 = (obj.w - obj.u) / (obj.w - obj.v);
            end
        end

        function decide_p(obj)
            % function decide_p makes a perceptual decision (obj.d_t)
            % for the categorical agent.
            %
            % INPUT:
            %   obj: current object

            % COMPUTE BELIEF STATES
            if obj.agent == 0
                obj.pi_0 = 0.5;
                obj.pi_1 = 0.5;
                obj.p_d_0 = 0.5;
            else
                obj.p_s_giv_o(obj.o_t); % compute belief states
                if obj.task_agent_analysis
                    obj.p_d_0 = normcdf(0, obj.o_t, obj.sigma);
                else
                    if obj.pi_0 >= obj.pi_1 % get perceptual decision
                        obj.p_d_0 = 1;
                    else
                        obj.p_d_0 = 0;
                    end
                end

                % PERCEPTUAL DECISION
                obj.p_d_t = [obj.p_d_0, 1-obj.p_d_0];
                obj.d_t = binornd(1, obj.p_d_t(2));
            end
        end

        function cat_bs(obj)
            % function cat_bs computes the categorical belief states (obj.pi_0, obj.pi_1)
            % based of the current perceptual decision.
            %
            % INPUT:
            %   obj: current object

            obj.decide_p();
            if obj.d_t == 0
                obj.pi_0 = 0;
                obj.pi_1 = 1;
            else
                obj.pi_0 = 1;
                obj.pi_1 = 0;
            end
        end

        function ev_cat(obj)
            % function ev_cat computes expected value for categorical belief states.
            %
            % INPUT:
            %   obj: current object

            obj.cat_bs(); % categorical belief states
            obj.compute_valence(); % expected values
        end

        function poly_eval = eval_poly(obj)
            % function poly_eval evaluates the polynomial.
            %
            % INPUT:
            %   obj: current object

            poly_int = polyint([obj.c_t, 0]); % anti-derivative of the polynomial
            poly_eval = polyval(poly_int, [0, 1]); % evaluate the polynomial
            poly_eval = diff(poly_eval); % difference of evaluated polynomial
        end

        function compute_valence(obj)
            % function compute_valence computes the expected value of each
            % action (obj.v_a_t) based on belief states and contingency parameter.
            %
            % INPUT:
            %   obj: current object

            if obj.eval_ana == 1
                obj.E_mu_t = obj.eval_poly();
            else
                obj.E_mu_t = dot(obj.mu_for_ev, obj.p_mu); % approximate EV
                obj.G = dot(obj.mu_for_ev, obj.p_mu);
            end
            obj.v_a_0 = (obj.pi_0 - obj.pi_1) * obj.E_mu_t + obj.pi_1;
            obj.v_a_1 = (obj.pi_1 - obj.pi_0) * obj.E_mu_t + obj.pi_0;
            obj.v_a_t = [obj.v_a_0, obj.v_a_1]; % concatenate action valences
        end

        function softmax(obj)
            % function softmax returns choice probabilities (obj.p_a_t) based on the computed
            % action values and beta parameter
            %
            % INPUT:
            %   obj: current object

            obj.p_a_t = exp(obj.v_a_t.*obj.beta) / sum(exp(obj.v_a_t.*obj.beta));
        end

        function hardmax(obj)
            % function hardmax returns choice probabilities (obj.p_a_t)
            % that assign probability 1 to the action(s) with the maximum value
            % and 0 to all other actions.

            % Find the maximum value among action values
            max_val = max(obj.v_a_t);

            % Initialize all probabilities to zero
            obj.p_a_t = zeros(size(obj.v_a_t));

            % Assign probability 1 to all actions with the maximum value
            obj.p_a_t(obj.v_a_t == max_val) = 1;

            % If there are multiple max actions, normalize (optional)
            obj.p_a_t = obj.p_a_t / sum(obj.p_a_t);
        end


        function int_voi = integrate_voi(obj,voi,varargin)
            % function integrate_voi returns the integral of a particular variable
            % conditional of contrast difference.
            %
            % INPUT:
            %    obj: current object
            %    voi: 0 is action values, voi = 1 is polynomial updates.

            if nargin > 2
                obj.r_t = varargin{1}; % access the first varargin because the only additional input is r_t
            end
            voi_matrix = NaN(length(obj.set_o), 2); % initialize voi matrix
            pi_matrix = NaN(length(obj.set_o), 2); % pi_0(i)/pi_1(i) per hypothetical grid point, for a correctly marginalized belief state (see below)
            obj.p_o_giv_u = normpdf(obj.set_o, obj.o_t, obj.sigma); % observation probabilities
            % Numerical safety guard: when o_t falls many sigma beyond the
            % entire set_o grid (more likely with small sigma, e.g.
            % sigma<=0.02), normpdf underflows to exactly 0 at every grid
            % point, so sum(p_o_giv_u)==0 and normalizing below would
            % evaluate to NaN everywhere. Fall back to a one-hot weighting
            % on the grid point closest to o_t, matching what the
            % normalized Gaussian weighting converges to in this regime
            % (all mass concentrated at the nearest point).
            if sum(obj.p_o_giv_u) == 0
                [~, closest_idx] = min(abs(obj.set_o - obj.o_t));
                obj.p_o_giv_u_norm = zeros(size(obj.p_o_giv_u));
                obj.p_o_giv_u_norm(closest_idx) = 1;
            else
                obj.p_o_giv_u_norm = obj.p_o_giv_u / sum(obj.p_o_giv_u); % normalised probabilities
            end
            for i = 1:length(obj.set_o)
                if obj.confirmation_bias == 1
                    obj.p_s_giv_oConfirm(obj.set_o(i),varargin{2}) % state given observations
                else
                    obj.p_s_giv_o(obj.set_o(i)) % state given observations
                end
                pi_matrix(i, 1) = obj.pi_0; % per-grid-point belief, captured before the next iteration overwrites obj.pi_0/obj.pi_1
                pi_matrix(i, 2) = obj.pi_1;
                if voi == 0 % action values
                    obj.compute_valence(); % action values
                    voi_matrix(i, 1) = obj.v_a_t(1); % value for action_0
                    voi_matrix(i, 2) = obj.v_a_t(2);
                elseif voi == 1 % polynomial updates
                    obj.compute_q(obj.r_t);
                    voi_matrix(i, 1) = obj.q_0;
                    voi_matrix(i, 2) = obj.q_1;
                end
            end
            obj.q_0 = sum(voi_matrix(:, 1) .* obj.p_o_giv_u_norm');
            obj.q_1 = sum(voi_matrix(:, 2) .* obj.p_o_giv_u_norm');
            % Expose the correctly marginalized belief state here --
            % otherwise obj.pi_0/obj.pi_1 are left holding whatever the
            % LAST grid point's p_s_giv_o call computed (a constant
            % depending only on sigma and set_o's fixed range, unrelated
            % to this trial's actual observation), which is what
            % task_agent_int.m was recording into its state_0/state_1
            % columns for every with-integration simulation (see
            % conversation).
            obj.pi_0 = sum(pi_matrix(:, 1) .* obj.p_o_giv_u_norm');
            obj.pi_1 = sum(pi_matrix(:, 2) .* obj.p_o_giv_u_norm');
            int_voi = [obj.q_0, obj.q_1];

        % int_voi = [sum(voi_matrix(:, 1) .* obj.p_o_giv_u_norm'), ...
        %            sum(voi_matrix(:, 2) .* obj.p_o_giv_u_norm')];
        end

        function decide_e(obj, o_t)
            % function decide_e makes an economic decision (obj.a_t) based for the agent and
            % uses softmax policy to make a choice for the agent.
            %
            % INPUT:
            %   obj: current object
            %   o_t: observed contrast difference

            if obj.agent == 0 % random decision agent
                obj.v_a_t = [0.5, 0.5];
                obj.p_a_t = [0.5, 0.5];
            elseif obj.agent == 1 % normative bayesian agent
                if obj.task_agent_analysis == 1 % compute action values by integrating over observations
                    voi = 0;
                    obj.v_a_t = obj.integrate_voi(voi); % multipled by contrast differences contingent observation
                else
                    obj.p_s_giv_o(o_t); % Compute belief state based on direct observations
                    obj.compute_valence(); % action valences based on direct observations
                end
            elseif obj.agent == 2 % categorical agent
                obj.ev_cat();
            end
            if obj.agent ~= 0
                % for all agents except 0, use the softmax decision policy
                % for action selection.
                % obj.hardmax();
                obj.softmax();
            end
            obj.a_t = binornd(1, obj.p_a_t(2)); % action
        end

        function r_t = compute_action_dep_rew(obj, r_t)
            % function compute_action_dep_rew recodes task generated reward
            % contingent on action.
            %
            % INPUT:
            %   obj: current object
            %   r_t: task generated reward
            %
            % OUTPUT:
            %   r_t: agent recoded reward

            r_t = r_t + (obj.a_t *((-1) .^ (2 + r_t)));
        end

        function compute_q(obj,r_t)
            % function compute_q computes q_0 and q_1 for the polynomial update.
            %
            % INPUT:
            %   obj: current object
            %   r_t: task generated reward

            obj.t = length(obj.c_t) + 1; % degree of polynomial
            obj.r_t = obj.compute_action_dep_rew(r_t); % recode reward to get action contingent reward
            if obj.eval_ana == 1
                obj.G = obj.eval_poly();% valuated polynomials
                obj.C = (obj.pi_0 - obj.pi_1) * ((1 - obj.G) .^ (1 - obj.r_t)) * (obj.G .^ obj.r_t) + obj.pi_1; % common denominator
                obj.q_0 = (obj.pi_1 .^ obj.r_t) * (obj.pi_0 .^ (1-obj.r_t)) / obj.C; % fraction for polynomial updates
                obj.q_1 = ((-1) .^ (obj.r_t + 1)) * (obj.pi_0 - obj.pi_1) / obj.C;

            else
                % NOTE: this branch deliberately does NOT update
                % obj.product/obj.mu/obj.mu_for_ev -- compute_q is called
                % once per trial for the no-integration path, but also
                % once PER HYPOTHETICAL GRID POINT from integrate_voi's
                % voi==1 loop (the with-integration learning path).
                % Mutating this persistent belief state here would apply
                % it once per grid point with mutually-inconsistent
                % hypothetical factors instead of once per trial with the
                % final marginalized q_0/q_1, corrupting the belief
                % (observed: "with integration" mu stuck at 0.5). The
                % actual update now happens in update_mu_from_q, called
                % explicitly and exactly once per trial from learn().

                % ORIGINAL (no normalization) -- kept for reference/revert,
                % not deleted. See the NORMALIZATION FIX note below for why
                % this was changed.
                % q_0_num = (obj.pi_1 ^ obj.r_t) * (obj.pi_0 ^ (1 - obj.r_t)); % numerical polynomial updates
                % q_1_num = ((-1) ^ (obj.r_t + 1)) * (obj.pi_0 - obj.pi_1);
                % obj.q_0 = q_0_num;
                % obj.q_1 = q_1_num ;

                % NORMALIZATION FIX: eval_ana==1 (above) divides by a
                % normalizing constant C built from its own current G,
                % which makes the update exactly neutral when the outcome
                % matches the model's own current prediction -- proven
                % algebraically and numerically (see conversation) to be
                % why eval_ana==1 resists the o_t=c_t overshoot bug while
                % this branch (previously undivided, above) doesn't. This
                % mirrors that exact formula, using the numerical EV
                % estimate (dot(mu_for_ev, p_mu), freshly computed here
                % rather than read off obj.G -- obj.G defaults to NaN
                % before any trial has run, so it isn't valid on trial 1)
                % in place of eval_ana==1's eval_poly()-based G.
                G_num = dot(obj.mu_for_ev, obj.p_mu); % current numerical EV estimate, i.e. this trial's pre-update belief
                C_num = (obj.pi_0 - obj.pi_1) * ((1 - G_num) .^ (1 - obj.r_t)) * (G_num .^ obj.r_t) + obj.pi_1; % normalizing constant, mirrors eval_ana==1's C
                q_0_num = (obj.pi_1 ^ obj.r_t) * (obj.pi_0 ^ (1 - obj.r_t)) / C_num; % numerical polynomial updates, now normalized
                q_1_num = ((-1) ^ (obj.r_t + 1)) * (obj.pi_0 - obj.pi_1) / C_num;
                obj.q_0 = q_0_num;
                obj.q_1 = q_1_num;

            end
        end

        function update_mu_from_q(obj)
            % function update_mu_from_q applies the belief (obj.mu /
            % obj.mu_for_ev) update using the current obj.q_0/obj.q_1 --
            % the "numerical" (eval_ana==0) counterpart to
            % update_coefficients's polynomial update. Called explicitly,
            % exactly once per trial, by learn() -- see compute_q's
            % header for why this must not happen inside
            % compute_q/compute_q_confirmation/compute_q_alternative
            % themselves.
            %
            % INPUT:
            %   obj: current object

            obj.product = obj.product.' .* (obj.q_1 * obj.p_mu + obj.q_0);
            % Numerical safety guard: obj.product accumulates via repeated
            % multiplication across trials with no rescaling, which
            % underflows to exactly 0 over long blocks (e.g. 1000 trials),
            % after which mu_for_ev = mu/sum(mu) becomes 0/0 = NaN.
            % Renormalizing to sum to 1 every trial keeps it in a safe
            % numeric range without changing the relative per-point
            % weighting that matters for mu_for_ev or the next trial's
            % multiplicative update (each grid point's update factor
            % doesn't depend on product's overall scale).
            if sum(obj.product) > 0
                obj.product = obj.product / sum(obj.product);
            else
                obj.product = ones(size(obj.product)) / numel(obj.product); % fall back to uniform if every factor underflowed to exactly 0
            end
            obj.mu = obj.product; % updated mu
            obj.mu_for_ev = obj.mu / sum(obj.mu); % mu to calculate EV
        end


        % Add this method to check if outcome confirms choice
        % function check_confirmation(obj, r_t, confirmRew)
        %     % function check_confirmation determines if the outcome confirms the agent's choice
        %     % Based on the contrast-dependent reward structure from compute_confirm
        %     %
        %     % INPUT:
        %     %   obj: current object
        %     %   r_t: task generated reward (obtained_reward)
        %     %   contrast_level: current trial contrast (1 for high contrast, other for low)
        %     %   state: current trial state
        % 
        %     % Determine confirmation based on contrast level and state-action relationship
        %     % if obj.pi_0 > obj.pi_1
        %     %     state = 0;
        %     % else
        %     %     state = 1;
        %     % end
        %     % state_matches_action = (state == obj.a_t);
        %     % 
        %     % % Low contrast trials (mu > 0.5 - should go with the state)
        %     % if state_matches_action
        %     %     % When state matches action, confirmation = reward
        %     %     % (getting rewarded for matching confirms the choice was right)
        %     %     obj.outcome_confirms_choice = (r_t == 1);
        %     % else
        %     %     % When state doesn't match action, confirmation = 1 - reward
        %     %     % (getting punished for not matching confirms the choice was wrong)
        %     %     obj.outcome_confirms_choice = (r_t == 0);
        %     % end
        %     % obj.outcome_confirms_choice = confirmRew;
        % end


        % Modified compute_q method with confirmation bias
        function compute_q_confirmation(obj, r_t)
            % function compute_q computes q_0 and q_1 for the polynomial update with confirmation bias.
            %
            % INPUT:
            %   obj: current object
            %   r_t: task generated reward

            obj.t = length(obj.c_t) + 1; % degree of polynomial
            obj.r_t = obj.compute_action_dep_rew(r_t); % recode reward to get action contingent reward

            % Check if outcome confirms choice
            % obj.check_confirmation(r_t);

            % Apply confirmation bias scaling
            if obj.confirmation_bias > 0
                if obj.outcome_confirms_choice
                    obj.bias_scaling = 1.01 * obj.confirmation_bias; % amplify confirming updates
                end
            end

            if obj.eval_ana == 1
                obj.G = obj.eval_poly();% valuated polynomials
                obj.C = (obj.pi_0 - obj.pi_1) * ((1 - obj.G) .^ (1 - obj.r_t)) * (obj.G .^ obj.r_t) + obj.pi_1; % common denominator

                % Apply confirmation bias to polynomial update fractions
                obj.q_0 = obj.bias_scaling * (obj.pi_1 .^ obj.r_t) * (obj.pi_0 .^ (1-obj.r_t)) / obj.C;
                obj.q_1 = obj.bias_scaling * ((-1) .^ (obj.r_t + 1)) * (obj.pi_0 - obj.pi_1) / obj.C;

            else
                q_0_num = (obj.pi_1 ^ obj.r_t) * (obj.pi_0 ^ (1 - obj.r_t)); % numerical polynomial updates
                q_1_num = ((-1) ^ (obj.r_t + 1)) * (obj.pi_0 - obj.pi_1);

                % Apply confirmation bias scaling
                q_0_num = obj.bias_scaling * q_0_num;
                q_1_num = obj.bias_scaling * q_1_num;

                % NOTE: does not update obj.product/mu/mu_for_ev here --
                % see compute_q's identical note for why (update_mu_from_q
                % applies this once per trial from learn() instead).
                obj.q_0 = q_0_num;
                obj.q_1 = q_1_num;
            end
        end

        % Alternative implementation: Confirmation bias through learning rate modulation
        function compute_q_alternative(obj, r_t)
            % Alternative implementation using prediction error-based confirmation bias
            %
            % INPUT:
            %   obj: current object
            %   r_t: task generated reward

            obj.t = length(obj.c_t) + 1;
            obj.r_t = obj.compute_action_dep_rew(r_t);

            % Calculate prediction error
            expected_reward = obj.v_a_t(obj.a_t + 1); % expected value of chosen action
            prediction_error = obj.r_t - expected_reward;

            % Confirmation bias: reduce learning from negative prediction errors
            if obj.confirmation_bias > 0
                if prediction_error < 0 % disconfirming outcome
                    learning_rate = 1 / (1 + obj.confirmation_bias);
                else % confirming outcome
                    learning_rate = 1 + obj.confirmation_bias;
                end
            else
                learning_rate = 1.0;
            end

            % Rest of the computation with modified learning rate
            if obj.eval_ana == 1
                obj.G = obj.eval_poly();
                obj.C = (obj.pi_0 - obj.pi_1) * ((1 - obj.G) .^ (1 - obj.r_t)) * (obj.G .^ obj.r_t) + obj.pi_1;

                obj.q_0 = learning_rate * (obj.pi_1 .^ obj.r_t) * (obj.pi_0 .^ (1-obj.r_t)) / obj.C;
                obj.q_1 = learning_rate * ((-1) .^ (obj.r_t + 1)) * (obj.pi_0 - obj.pi_1) / obj.C;
            else
                q_0_num = learning_rate * (obj.pi_1 ^ obj.r_t) * (obj.pi_0 ^ (1 - obj.r_t));
                q_1_num = learning_rate * ((-1) ^ (obj.r_t + 1)) * (obj.pi_0 - obj.pi_1);

                % NOTE: does not update obj.product/mu/mu_for_ev here --
                % see compute_q's identical note for why (update_mu_from_q
                % applies this once per trial from learn() instead).
                obj.q_0 = q_0_num;
                obj.q_1 = q_1_num;
            end
        end

        % Method to set confirmation bias parameter
        function setConfirmationBias(obj, bias_strength)
            % function setConfirmationBias sets the confirmation bias parameter
            %
            % INPUT:
            %   obj: current object
            %   bias_strength: strength of confirmation bias (0 = no bias, >0 = increasing bias)

            obj.confirmation_bias = bias_strength;
        end

        function q_a2(obj,r_t)
            % function q_a2 computes q values (obj.q_0,obj.q_1) for polynomial updates of categorical
            % agent.
            %
            % INPUTS:
            %   obj: current object
            %   r_t: task generated reward

            obj.cat_bs();
            obj.compute_q(r_t);
        end

        function update_coefficients(obj)
            % update_coefficients updates the prior probability using the
            % evaluated polynomials (obj.c_t).
            %
            % INPUT:
            %   obj: current object

            obj.d = zeros(1,obj.t);
            obj.d(end) = obj.q_0 * obj.c_t(end);  % update last element
            for n = 0:(obj.t - 3)
                obj.d(end-(n+1)) = obj.q_1 * obj.c_t(end-n) + obj.q_0 * obj.c_t(end-(n+1)); % update all elements except last and first
            end
            obj.d(1) = obj.q_1 * obj.c_t(1); % update first element
            obj.c_t = obj.d;
        end

        function sigma = update_sigma(obj)
            obj.outcome_confirms_choice = obj.confirmation_bias;
            if obj.outcome_confirms_choice == 1
                sigma = obj.sigma * obj.confirmBias;
            end
        end
  

        function learn(obj,r_t)
            % learn updates the q-values to help the agent learn and
            % update contingency parameter.
            %
            % INPUT:
            %   obj: current object
            %   r_t: task generated reward

            if obj.agent == 0
                obj.G = 0.5;
            elseif obj.agent == 1
                if obj.task_agent_analysis == 1
                    % compute qs over contrast based observations
                    if obj.confirmation_bias == 1
                        % Check if outcome confirms choice
                        % obj.check_confirmation(r_t);
                        sigma = obj.update_sigma;
                        voi = 1 ;
                        q_s = obj.integrate_voi(voi, r_t,sigma);
                    else
                        voi = 1 ;
                        q_s = obj.integrate_voi(voi, r_t);
                    end
                    obj.q_0 = q_s(1);
                    obj.q_1 = q_s(2);
                    obj.update_coefficients(); % updating using q-values
                    if obj.eval_ana == 1 % only when evaluating analytically
                        obj.G = obj.eval_poly();
                        % obj.G(obj.G > 1) = 1;  % Set any values above 1 to 1
                    else
                        % Belief update applied exactly once per trial,
                        % using the final marginalized q_0/q_1 from
                        % integrate_voi above -- see compute_q's header
                        % and update_mu_from_q's header for why this must
                        % not happen inside compute_q itself (which
                        % integrate_voi calls once per hypothetical grid
                        % point, not once per trial).
                        obj.update_mu_from_q();
                        obj.E_mu_t = dot(obj.mu_for_ev, obj.p_mu); % approximate EV
                        obj.G = dot(obj.mu_for_ev, obj.p_mu);
                    end
                else
                    if obj.confirmation_bias == 1
                        % Check if outcome confirms choice
                        % obj.check_confirmation(r_t);
                        sigma = obj.update_sigma;
                        obj.p_s_giv_oConfirm(obj.o_t,sigma);
                    end
                    % obj.p_s_giv_o(obj.o_t); % belief states
                    obj.compute_q(r_t);
                    obj.update_coefficients(); % update coefficients
                    if obj.eval_ana == 1 % only when evaluating analytically
                        obj.G = obj.eval_poly();
                        % obj.G(obj.G > 1) = 1;  % Set any values above 1 to 1
                    else
                        obj.update_mu_from_q(); % belief update applied exactly once per trial, using this trial's q_0/q_1
                        obj.E_mu_t = dot(obj.mu_for_ev, obj.p_mu); % approximate EV
                        obj.G = dot(obj.mu_for_ev, obj.p_mu);
                    end
                    % obj.sigma = 0.03;
                end
            elseif obj.agent == 2 % updating q values for categorical agent
                obj.q_a2(r_t); % compute q-values
                obj.update_coefficients(); % update the values
                obj.G = obj.eval_poly();
            end
        end
    end
end