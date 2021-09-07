classdef Agent < agentvars
    properties
        o_t = NaN % current observation
        a_t = NaN % current economic decision
        r_t = NaN % current reward
        c_t = NaN % uniform prior distribution over mu
        pi_0 = NaN% belief state in favor of s_t = 0
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
        product = ones(1,100)
        mu = ones(1,100)
        p_mu = linspace(0, 1, 100)
        mu_for_ev 
    end
    methods
        function obj=Agent
            % initializes all other properities of the class
            obj.p_o_giv_u = zeros(1,length(obj.set_o)); % probabilities of observations given contrast differences
            obj.p_o_giv_u_norm = zeros(1,length(obj.set_o)); % normalised probabilities of observations
            obj.mu_for_ev = obj.mu
            obj.mu_for_ev = obj.mu_for_ev/sum(obj.mu_for_ev);
            obj.c_t = obj.c0 % prior distribution over mu
        end
        function observation_sample(obj,c_t) 
            % contrast difference dependent observation
            % for task_agent_data_analysis
            % o_t is the same as the presented contrast difference
            % for simulations, the observation is sampled from a normal
            % distribution with presented contrast as mean and perceptual
            % sensitivity as SD. 
            if obj.task_agent_analysis == true 
                obj.o_t = obj.c_t 
            else
                obj.o_t = normrnd(c_t,obj.sigma)
            end
        end
        function p_s_giv_o(obj, o_t)
            % computes belief state given observation of contrast
            % difference. the absolute minimum contrast difference is
            % decided based on the experimental condition.
            if obj.condition == 3 || obj.condition == 4 
                u = normcdf(obj.kappa_min, obj.o_t, obj.sigma)
            else
                u = normcdf(0, obj.o_t, obj.sigma)
            end
            v = normcdf(-obj.kappa_max, obj.o_t, obj.sigma)
            w = normcdf(obj.kappa_max, obj.o_t, obj.sigma)
            if obj.sigma <= 0.015 && obj.o_t == -0.2 
                % for very low sensitivity parameters
                % for very low observations
                obj.pi_0 = 1.0
                obj.pi_1 = 0.0
            else
            %For all other cases
                obj.pi_0 = (u - v) / (w - v)
                obj.pi_1 = (w - u) / (w - v)
            end
        end
        function poly_eval = eval_poly(obj)
            % Evaluated polynomial
            obj.c_t = obj.c_t(:)' % convert to row vector
            poly_int = polyint(obj.c_t, 0) % anti-derivative of the polynomial
            poly_eval = polyval(poly_int, [0, 1]) % evaluate the polynomial
            poly_eval = diff(poly_eval) % difference of evaluated polynomial
        end
        function compute_valence(obj)
            % computes the expected value weighted by belief states for
            % each action.
            if obj.eval_ana == 1
                obj.E_mu_t = obj.eval_poly() % does this work?
            else
                obj.E_mu_t = dot(obj.mu_for_ev, obj.p_mu) % approximate EV
                obj.G = dot(obj.mu_for_ev, obj.p_mu)
            end 
        obj.v_a_0 = (obj.pi_0 - obj.pi_1) * obj.E_mu_t + obj.pi_1 
        obj.v_a_1 = (obj.pi_1 - obj.pi_0) * obj.E_mu_t + obj.pi_0
        %Concatenate action valences
        obj.v_a_t = [obj.v_a_0, obj.v_a_1]
        end
        function softmax(obj, v_a_t)
            % action selection based on the softmax rule
            % uses the computed action values and beta parameter
            obj.p_a_t = exp(obj.v_a_t*obj.beta) / sum(exp(obj.v_a_t*obj.beta)) 
        end
        function compute_mixture(first_comp, second_comp)
            % computes the mixture between agent 1 & agent 2
            % first_comp is agent 1's mixture component
            % second_comp is agent 2's mixture component
            mixture_0 = first_comp(1) * obj.lambda + second_comp(2) * (1 - obj.lambda)
            mixture_1 = first_comp(2) * obj.lambda + second_comp(1) * (1 - obj.lambda)
            mixture = [mixture_0, mixture_1]
        end
        function compute_control_mixture(first_comp, second_comp)
            % computes the mixture between agent 1 & agent 2
            % first_comp is agent 1's mixture component
            % second_comp is agent 2's mixture component
            % control mixture agent uses a fixed lambda parameter i.e.
            % lambda_control
            mixture_0 = first_comp(1) * obj.lambda_control + second_comp(2) * (1 - obj.lambda_control)
            mixture_1 = first_comp(2) * obj.lambda_control + second_comp(1) * (1 - obj.lambda_control)
            mixture = [mixture_0, mixture_1]
        end
        function int_voi = integrate_voi(obj, voi,varargin)
            % integral of a particular variable conditional of contrast
            % difference. voi == 0 is action values, voi = 1 is polynomial
            % updates. 
            obj.r_t = varargin{1} % access the first varargin because the only additional input is r_t
            voi_matrix = NaN(length(obj.set_o), 2) % initialize voi matrix
            obj.p_o_giv_u = normpdf(obj.set_o, obj.o_t, obj.sigma) % observation probabilities
            obj.p_o_giv_u_norm = obj.p_o_giv_u / sum(obj.p_o_giv_u) % normalised probabilities
            for i = 1:length(obj.set_o) 
                obj.p_s_giv_o(obj.set_o(i)) % state given observations
                if voi == 0 % action values
                        obj.v_a_t = obj.compute_valence() % action values
                        voi_matrix(i, 1) = obj.v_a_t(1) % value for action_0
                        voi_matrix(i, 2) = obj.v_a_t(2)
                elseif voi == 1 % polynomial updates
                    obj.q_0, obj.q_1 = obj.compute_q(obj.r_t, obj.pi_0, obj.pi_1)
                    voi_matrix(i, 1) = obj.q_0
                    voi_matrix(i, 2) = obj.q_1;
                end
            end
                obj.q_0 = sum(voi_matrix(:, 1) * obj.p_o_giv_u_norm)
                obj.q_1 = sum(voi_matrix(:, 2) * obj.p_o_giv_u_norm)  
                int_voi = [obj.q_0, obj.q_1]
        end
        function ev_cat (obj) 
            % action values for the salience based categorical agent
            if obj.pi_0 >= obj.pi_1 % compare belief states
                    v_a_1 = 1
                    v_a_t = [1-v_a_1, v_a_1]
            else
                    v_a_0 = 0
                    v_a_t = [v_a_0, 1-v_a_0]
            end
        end
        function ev_cat2(obj)
            % action values for the salience based categorical agent
            % the values are not based on the agent's belief states
            % check for the patch with higher contrast level w/o the BS
            if obj.o_t > 0 
                    v_a_1 = 1 
                    v_a_t = [1-v_a_1, v_a_1]
            else
                    v_a_0 = 0
                    v_a_t = [1-v_a_1, v_a_1]
            end
        end
        function decide_e(obj) 
            % makes an economic decision based on the agent
            % if task_agent_analysis is true, then compute action values by
            % integrating over observations
            % finally, uses softmax policy to make a choice
            if obj.agent == 0 % random decision agent
                obj.v_a_t = [0.5, 0.5];
                obj.p_a_t = [0.5, 0.5];  
            elseif obj.agent == 1 % bayesian agent
                if obj.task_agent_analysis == 1
                    voi = 0
                    obj.v_a_t = obj.integrate_voi(voi) % multipled by contrast differences contingent observation
                else
                    % Compute belief state based on direct observations
                    obj.p_s_giv_o(); 
                    % Compute action valences
                    obj.compute_valence(); % action valences based on direct observations
                end 
            elseif obj.agent == 2 % BS based categorical decisions
                obj.p_s_giv_o(obj.o_t) % calculate the belief state based on observations
                obj.ev_cat()
            elseif obj.agent == 2.1 % purely observations based another version of categorical model
                obj.ev_cat2()
            elseif obj.agent == 3
                 if obj.task_agent_analysis
                    voi = 0
                    v_a_t_ag_1 = obj.integrate_voi(voi)
                    v_a_t_ag_2 = obj.ev_cat();
                    v_a_t_ag_2_1 = obj.ev_cat2();
                    obj.v_a_t = obj.compute_mixture(v_a_t_ag_1, v_a_t_ag_2)
                 else
                    [obj.pi_0, obj.pi_1] = obj.p_s_giv_o(); 
                    v_a_t_ag_1 = obj.compute_valence(obj.pi_0, obj.pi_1);
                    v_a_t_ag_2 = obj.ev_cat();
                    v_a_t_ag_2_1 = obj.ev_cat2();
                    obj.v_a_t = obj.compute_mixture(v_a_t_ag_1, v_a_t_ag_2)
                 end
             elseif obj.agent == 4 % control mixture agent
                 if obj.task_agent_analysis
                    voi = 0
                    v_a_t_ag_1 = obj.integrate_voi(voi)
                    v_a_t_ag_2 = obj.ev_cat();
                    v_a_t_ag_2_1 = obj.ev_cat2();
                    obj.v_a_t = obj.compute_control_mixture(v_a_t_ag_1, v_a_t_ag_2)
                 else
                    [obj.pi_0, obj.pi_1] = obj.p_s_giv_o(); 
                    v_a_t_ag_1 = obj.compute_valence(obj.pi_0, obj.pi_1);
                    v_a_t_ag_2 = obj.ev_cat();
                    v_a_t_ag_2_1 = obj.ev_cat2();
                    obj.v_a_t = obj.compute_control_mixture(v_a_t_ag_1, v_a_t_ag_2)
                 end
            end
        if obj.agent ~= 0 % for all agents except 0
            obj.softmax(obj.v_a_t); % decision policy
            obj.a_t = binornd(1, obj.p_a_t(1)); % action
        end
        end 
        function compute_action_dep_rew(obj, r_t) % add extra line in task_agent_int?
            % reward contingent on action 
            obj.r_t = r_t + (obj.a_t * ((-1) * (2 + r_t)))
        end
        function compute_q(obj,r_t)
            % computes q_0 and q_1
            obj.t = length(obj.c_t) + 1; % degree of polynomial
            obj.compute_action_dep_rew(r_t); % reward %??
            if obj.eval_ana == 1
                obj.G = obj.eval_poly() % evaluated polynomials
                obj.C = (obj.pi_0 - obj.pi_1) * ((1 - obj.G) ^ (1 - obj.r_t)) * (obj.G ^ obj.r_t) + obj.pi_1 % common denominator
                obj.q_0 = (obj.pi_1 ^ obj.r_t) * (obj.pi_0 ^ (1 - obj.r_t)) / obj.C
                obj.q_1 = ((-1) ^ (obj.r_t + 1)) * (obj.pi_0 - obj.pi_1) / obj.C
            else
                obj.q_0_num = (obj.pi_1 ^ obj.r_t) * (obj.pi_0 ^ (1 - obj.r_t)) % numerical polynomial updates
                obj.q_1_num = ((-1) ^ (obj.r_t + 1)) * (obj.pi_0 - obj.pi_1)
                obj.product = obj.product * (obj.q_1_num * obj.p_mu + obj.q_0_num) 
                obj.mu = obj.product % updated mu
                obj.mu_for_ev = obj.mu / np.sum(obj.mu) % mu to calculate EV
                obj.q_0 = obj.q_0_num
                obj.q_1 = obj.q_1_num 
            end
        end
        function a2_bs(obj)
            % BS for agent 2
            if obj.a_t == 1
                obj.pi_0 = 1
                obj.pi_1 = 0
            else
                obj.pi_0 = 0
                obj.pi_1 = 1
            end
        end
        function q_a2(obj,r_t)
            % compute q values for polynomial updates of a2
            obj.a2_bs()
            obj.compute_q(r_t)
        end
        function update_coefficients(obj)
            % updates the prior probability using polynomials
%             if numel(obj.c_t) > 3
%                 obj.c_t = reshape(obj.c_t,[2,2]) % reshape only when it is 2 by 2 ?
%             end
            obj.d = zeros(1,obj.t)
            obj.d(end) = obj.q_0 * obj.c_t(end)  % last element
        if numel(obj.c_t) > 3 % do you really need this? 
            for n = 0: (obj.t - 3)
                obj.d(end-n) = obj.q_1 * obj.c_t(end - n) + obj.q_0 * obj.c_t(end - (n+1))
            end
        end
        obj.d(1) = obj.q_1 * obj.c_t(1) % first element
        obj.c_t = obj.d
        end
        function learn(obj,r_t)
            % agent learns from the reward by updating the q-values
            if obj.agent == 0
                obj.G = 0.5
            elseif obj.agent == 1
                if obj.task_agent_analysis == 1
                    % compute qs over contrast based observations
                    voi = 1 
                    q_s = obj.integrate_voi(voi, r_t);
                    obj.q_0 = q_s(1); 
                    obj.q_1 = q_s(2);
                    obj.update_coefficients() % updating using q-values
                else
                    obj.p_s_giv_o(); % belief states
                    obj.compute_q(r_t)
                    obj.update_coefficients();    
                end
            elseif obj.agent == 2 % updating q values for agent 2
                obj.q_a2(r_t) % compute q-values
                obj.update_coefficients() % update the values
            elseif obj.agent == 3
                if obj.task_agent_analysis == 1
                    voi = 1
                    q_s_ag_1 = obj.integrate_voi(voi)
                    q_s_ag_2 = obj.q_a2(r_t)
                    q = obj.compute_mixture(q_s_ag_1, q_s_ag_2)
                    obj.q_0 = q(1)
                    obj.q_1 = q(2)
                    obj.update_coefficients()
                else
                    pi_0, pi_1 = obj.p_s_giv_o(obj.o_t)
                    q_0_ag_1, q_1_ag_1 = obj.compute_q(r_t, pi_0, pi_1)
                    q_s_ag_1 = [q_0_ag_1, q_1_ag_1]
                    q_s_ag_2 = obj.q_a2(r_t)
                    q = obj.compute_mixture(q_s_ag_1, q_s_ag_2)
                    obj.q_0 = q(1)
                    obj.q_1 = q(2)
                    obj.update_coefficients()
                end
                elseif obj.agent == 4 
                if obj.task_agent_analysis == 1
                    voi = 1
                    q_s_ag_1 = obj.integrate_voi(voi)
                    q_s_ag_2 = obj.q_a2(r_t)
                    q = obj.compute_control_mixture(q_s_ag_1, q_s_ag_2)
                    obj.q_0 = q(1)
                    obj.q_1 = q(2)
                    obj.update_coefficients()
                else
                    pi_0, pi_1 = obj.p_s_giv_o(obj.o_t)
                    q_0_ag_1, q_1_ag_1 = obj.compute_q(r_t, pi_0, pi_1)
                    q_s_ag_1 = [q_0_ag_1, q_1_ag_1]
                    q_s_ag_2 = obj.q_a2(r_t)
                    q = obj.compute_control_mixture(q_s_ag_1, q_s_ag_2)
                    obj.q_0 = q(1)
                    obj.q_1 = q(2)
                    obj.update_coefficients()
                end
            end  
            end
        end
end  

%% Questions

% How to compute lambda based on RU in agentvars? 
% Two versions of the salience based categorical models exist. One that
% uses BS to choose the
% most salient target, while the other just uses plain observations to make
% the decision about the most salient stimulus. Which is to be retained?
