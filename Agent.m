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
%         alpha = NaN % salience value for learning models
        phi_0 = NaN % effective salience
        phi_1 = NaN
        epsilon_m_0 = NaN % acquired salience
        epsilon_m_1 = NaN
        epsilon_p_0 = NaN % acquired salience
        epsilon_p_1 = NaN
        r_t_0 = 1 % current reward for trial 1
        r_t_1 = 0
        pe_0 = NaN % prediction errors
        pe_1 = NaN
    end
    methods
        function obj=Agent
            % this contructor methods initialises all other properties of
            % the class that are calculated based on exisitng properties of
            % Agent. 
            
            obj.p_o_giv_u = zeros(1,length(obj.set_o)); % probabilities of observations given contrast differences
            obj.p_o_giv_u_norm = zeros(1,length(obj.set_o)); % normalised probabilities of observations
            obj.mu_for_ev = obj.mu
            obj.mu_for_ev = obj.mu_for_ev/sum(obj.mu_for_ev);
            obj.c_t = obj.c0 % prior distribution over mu
            
        end
        function observation_sample(obj,c_t) 
            % this function computes the contrast difference dependent
            % observation. The function takes in contrast difference as
            % c_t and returns observation that is sampled from a normal
            % distribution with presented contrast as mean and perceptual
            % sensitivity as SD. 
            
            % For task_agent_data_analysis, 
            % o_t is the same as the presented contrast difference
            % for simulations, the observation is sampled 
            
            if obj.task_agent_analysis == true 
                obj.o_t = obj.c_t 
            else
                obj.o_t = normrnd(c_t,obj.sigma)
            end
        end
        function p_s_giv_o(obj, o_t)
            
%             This function computes belief state given observation of
%             contrast difference. o_t is the observed contrast
%             differnece and the function returns the belief state. The
%             absolute minimum contrast difference is decided based on the
%             experimental condition.

            if obj.condition == 3 || obj.condition == 4 
                u = normcdf(obj.kappa_min, o_t, obj.sigma);
            else
                u = normcdf(0, o_t, obj.sigma);
            end
            v = normcdf(-obj.kappa_max, o_t, obj.sigma);
            w = normcdf(obj.kappa_max, o_t, obj.sigma);
            
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
        
        function effective_salience(obj) 
            % This function computes the effective salience based on belief
            % states.
            obj.p_s_giv_o(obj.o_t)
            if obj.pi_0 >= obj.pi_1
                obj.phi_1 = obj.pi_0
                obj.phi_0 = 1 - obj.phi_1
            else
                obj.phi_0 = obj.pi_1
                obj.phi_1 = 1 - obj.phi_0  
            end
        end
        
        function get_q_s_a(obj)
%             This function computes q-values q_0_1, q_1_0 and q_1_1 based
%             on q_0_0 of the Q-learning model
            q_0_1 = 1-obj.q_0_0
            q_1_0 = 1-obj.q_0_0
            q_1_1 = obj.q_0_0
        end
        
        function compute_capital_q(q_0_0, q_1_0, q_0_1, q_1_1)
            
%         This function computes Q-values Q_0 and Q_1 of the Q-learning model
%         q_0_0: State 0, action 1
%         q_1_0: State 1, action 0
%         q_0_1: State 0, action 1
%         q_1_1: State 1, action 1
%         pi_0: Belief over state 0
%         pi_1: Belief over state 1

        capital_q_0 = q_0_0 * obj.pi_0 + q_1_0 * obj.pi_1
        capital_q_1 = q_0_1 * obj.pi_0 + q_1_1 * obj.pi_1
        capital_q_a = [capital_q_0, capital_q_1]
        end
        
        function prediction_errors(obj,capital_q_0, capital_q_1)
            
            % This function computes prediction errors based on current
            % reward and the expected Q-values of a choice.
            
            obj.get_q_s_a(obj.o_t)
            obj.compute_capital_q
            obj.pe_0 = obj.r_t_0 - capital_q_0
            obj.pe_1 = obj.r_t_1 - capital_q_1
        end
        function acquired_salience(obj)
            
            % This function calculates the acquired salience of a choice
            % based on its predictability of reward. The exact computation
            % of salience depends on the agent. 
            
            if obj.agent == 5
                % Mackintosh based salience computation
                obj.epsilon_m_0 % to be calculated based on the PEs
                obj.epsilon_m_1 
            elseif obj.agent == 6 && obj.agent == 7
                % Pearce Hall based salience computation
                obj.epsilon_p_0 % to be calculated based on the PEs
                obj.epsilon_p_1 
            end
        end
        
        function poly_eval = eval_poly(obj)
            
            % Evaluates the polynomial 
            %obj.c_t = obj.c_t(:)' % convert to row vector ?? do we need
            %this? 
            poly_int = polyint([obj.c_t, 0]) % anti-derivative of the polynomial
            poly_eval = polyval(poly_int, [0, 1]) % evaluate the polynomial
            poly_eval = diff(poly_eval) % difference of evaluated polynomial
        end
        
        function compute_valence(obj)
            
            % This function computes the expected value of each action
            % based on belief states and contingency parameter. For agent 5
            % and 6, the action value is a sum of acquired and effective
            % salience. 
            
            if obj.agent == 5 || obj.agent == 6
                obj.v_a_0 = obj.phi_0 + obj.epsilon_0
                obj.v_a_1 = obj.phi_1 + obj.epsilon_1
            else
                if obj.eval_ana == 1
                    obj.E_mu_t = obj.eval_poly() 
                else
                    obj.E_mu_t = dot(obj.mu_for_ev, obj.p_mu) % approximate EV
                    obj.G = dot(obj.mu_for_ev, obj.p_mu)
                end 
            obj.v_a_0 = (obj.pi_0 - obj.pi_1) * obj.E_mu_t + obj.pi_1 
            obj.v_a_1 = (obj.pi_1 - obj.pi_0) * obj.E_mu_t + obj.pi_0
            end       
        %Concatenate action valences
        obj.v_a_t = [obj.v_a_0, obj.v_a_1]
        end
        
        function softmax(obj)
            % This function guides action selection based on the softmax rule
            % uses the computed action values and beta parameter
            
            obj.p_a_t = exp(obj.v_a_t*obj.beta) / sum(exp(obj.v_a_t*obj.beta)) 
        end
        
        function compute_mixture(first_comp, second_comp)
            
            % This function computes the mixture between agent 1 & agent 2.
            % first_comp is agent 1's mixture component
            % second_comp is agent 2's mixture component
            mixture_0 = first_comp(1) * obj.lambda + second_comp(2) * (1 - obj.lambda)
            mixture_1 = first_comp(2) * obj.lambda + second_comp(1) * (1 - obj.lambda)
            mixture = [mixture_0, mixture_1]
        end
        
        function compute_control_mixture(first_comp, second_comp)
            
            % This function computes the mixture between agent 1 & agent 2
            % first_comp is agent 1's mixture component
            % second_comp is agent 2's mixture component
            % control mixture agent uses a fixed lambda parameter i.e.
            % lambda_control.
            
            mixture_0 = first_comp(1) * obj.lambda_control + second_comp(2) * (1 - obj.lambda_control)
            mixture_1 = first_comp(2) * obj.lambda_control + second_comp(1) * (1 - obj.lambda_control)
            mixture = [mixture_0, mixture_1]
        end
        
        function compute_mixture_learning(first_comp, second_comp)
            mixture_0 = first_comp(1) * obj.lambda + second_comp(2) * (1 - obj.lambda)
            mixture_1 = first_comp(2) * obj.lambda + second_comp(1) * (1 - obj.lambda)
            mixture = [mixture_0, mixture_1]
        end
        function int_voi = integrate_voi(obj, voi,varargin)
            
            % This function computes the integral of a particular variable
            % conditional of contrast difference. 
            % voi == 0 is action values, voi = 1 is polynomial
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
        
        function decide_p(obj)
%         This function implements the agent's perceptual decision
%         strategy.
            if obj.agent == 0
                obj.pi_0 = 0.5
                obj.pi_1 = 0.5
                p_d_0 = 0.5
            else
                obj.pi_0, obj.pi_1 = obj.p_s_giv_o(obj.o_t)
                
                if obj.task_agent_analysis
                    p_d_0 = norm.cdf(0, obj.o_t, obj.sigma)
                else
                    if obj.pi_0 >= obj.pi_1
                        p_d_0 = 1
                    else
                        p_d_0 = 0
                    end
                end

            obj.p_d_t = [p_d_0, 1-p_d_0]
            obj.d_t = binornd(1, obj.p_d_t(2))
            end
        end
        function cat_bs(obj)
            
%           This function computes the categorical belief states based of
%           the current perceptual decision and returns pi_0, pi_1 i.e.
%           belief states for the categorical agent.

            if obj.d_t == 0
                pi_0 = 1
                pi_1 = 0
            else
                pi_0 = 0
                pi_1 = 1
            end
        end
        
        function ev_cat (obj) 
            
            % This function computes action values for the salience based
            % categorical agent based on the categorical belief states. 
            obj.cat_bs()
            if obj.pi_0 >= obj.pi_1 % compare belief states
                    v_a_1 = 1
                    v_a_t = [1-v_a_1, v_a_1]
            else
                    v_a_0 = 0
                    v_a_t = [v_a_0, 1-v_a_0]
            end
        end
        
        function decide_e(obj, o_t) 
            % This function makes an economic decision based on the agent
            % by using the observed contrast difference. 
            % if task_agent_analysis is true, then compute action values by
            % integrating over observations
            % finally, uses softmax policy to make a choice for the agent
            if obj.agent == 0 % random decision agent
                obj.v_a_t = [0.5, 0.5];
                obj.p_a_t = [0.5, 0.5];  
            elseif obj.agent == 1 % bayesian agent
                if obj.task_agent_analysis == 1
                    voi = 0
                    obj.v_a_t = obj.integrate_voi(voi) % multipled by contrast differences contingent observation
                else
                    % Compute belief state based on direct observations
                    obj.p_s_giv_o(o_t);
                    % Compute action valences
                    obj.compute_valence(); % action valences based on direct observations
                end 
            elseif obj.agent == 2 
                % Belief states based categorical decisions
                
                obj.ev_cat()
            elseif obj.agent == 3
                % mixture agent
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
             elseif obj.agent == 4 
                 % control mixture agent
                 if obj.task_agent_analysis
                    voi = 0
                    v_a_t_ag_1 = obj.integrate_voi(voi)
                    v_a_t_ag_2 = obj.ev_cat();
                    obj.v_a_t = obj.compute_control_mixture(v_a_t_ag_1, v_a_t_ag_2)
                 else
                    [obj.pi_0, obj.pi_1] = obj.p_s_giv_o(); 
                    v_a_t_ag_1 = obj.compute_valence(obj.pi_0, obj.pi_1);
                    v_a_t_ag_2 = obj.ev_cat();
                    obj.v_a_t = obj.compute_control_mixture(v_a_t_ag_1, v_a_t_ag_2)
                 end
            elseif obj.agent == 5 || obj.agent == 6
                % Computes action values based on salience of an
                % alternative for the learning agents. 
                obj.effective_salience()
                obj.acquired_salience()
                obj.compute_valence()
            end
            if obj.agent ~= 0 
                % for all agents except 0, use the softmax decision policy
                % for action selection.
                obj.softmax() 
            end
        obj.a_t = binornd(1, obj.p_a_t(2)) % action
        end 
        
        function r_t = compute_action_dep_rew(obj, r_t) 
            % This function generates the reward contingent on action. 
            obj.r_t = r_t + (obj.a_t * ((-1) * (2 + r_t)))
        end
        
        function compute_q(obj,r_t)
            % This function computes q_0 and q_1 for the polynomial update.
            % r_t is the action dependent reward calculated by the agent. 
            
            obj.t = length(obj.c_t) + 1; % degree of polynomial
            obj.r_t = obj.compute_action_dep_rew(r_t); 
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
        
        function q_a2(obj,r_t)
            % This function computes q values for polynomial updates of
            % agent 2.
            obj.cat_bs()
            obj.compute_q(r_t)
        end
        
        function update_coefficients(obj)
            % This function updates the prior probability using the
            % evaluated polynomials.

            obj.d = zeros(1,obj.t)
            obj.d(end) = obj.q_0 * obj.c_t(end)  % last element
        
            for n = 0: (obj.t - 3)
                obj.d(end-(n+1)) = obj.q_1 * obj.c_t(end - n) + obj.q_0 * obj.c_t(end - (n+1))
            end
        
        obj.d(1) = obj.q_1 * obj.c_t(1) % first element
        obj.c_t = obj.d
        end
        
        function compute_q_0_0(obj, pi_0, pi_1)
%         """ This function computes q_0_0
%         :param pi_0: Belief over state 0
%         :param pi_1: Belief over state 1
%         :return: Computed q_0
%         """
           
            if pi_0 >= pi_1
                q_0_0 = obj.q_0_0 + pi_0 * obj.alpha * obj.epsilon_0 * (obj.r_t - obj.q_0_0)
            else
                q_0_0 = obj.q_0_0 + pi_1 * obj.alpha * obj.epsilon_0 *((1 - obj.r_t) - obj.q_0_0)
            end
        end
        
        function learn(obj,r_t)
            % This function updates the q-values to help the agent learn
            % from the reward (r_t).
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
                    obj.p_s_giv_o(obj.o_t); % belief states
                    obj.compute_q(r_t) % compute q-values for updae
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
            elseif obj.agent == 5
                % calculate PE
                % multiply it with LR, epsilon value i.e. the acquired
                % salience 
                % update capital Q values? 
                pi_0, pi_1 = obj.p_s_giv_o(obj.o_t)

                obj.r_t = obj.compute_action_dep_rew(r_t)

                obj.q_0_0 = obj.compute_q_0_0(pi_0, pi_1)

                obj.E_mu_t = obj.q_0_0
                obj.G = obj.E_mu_t
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

% To do: 
% Add agents in q_0_0
% finish learning
