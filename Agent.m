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
        product = ones(1,100)
        mu = repelem(1,100)
        p_mu = linspace(0, 1, 100)
        mu_for_ev 
        u = NaN
        v = NaN
        w = NaN
%         d_t = NaN % perceptual decision
        p_d_0
        p_d_t
        s0
        s1
        dist
        q_0_1
        q_1_0
        q_1_1
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
        function observation_sample(obj,c_t) 
            % OBSERVATION_SAMPLE computes the contrast difference dependent
            % observation. 
            % INPUT:
                % c_t: task generated contrast difference 
                % obj: current object
            % OUTPUT:
                % o_t: returns observation that is sampled from a normal
                % distribution with presented contrast difference as mean 
                % and perceptual sensitivity as variance. 
            if obj.task_agent_analysis == true 
                obj.o_t = c_t; % for task_agent_data_analysis, o_t is the same as the presented contrast difference
            else
                obj.o_t = normrnd(c_t,obj.sigma); % for simulations, the observation is sampled 
            end
        end
        function p_s_giv_o(obj, o_t)     
            % P_S_GIV_O computes belief state given observation of 
            % contrast difference. 
            % INPUT: 
                % obj: current object
                % o_t: observed contrast differnece
            % OUTPUT: 
                % pi_0,pi_1: returns the belief state.
            obj.u = normcdf(0, o_t, obj.sigma);
            obj.v = normcdf(-obj.kappa_max, o_t, obj.sigma);
            obj.w = normcdf(obj.kappa_max, o_t, obj.sigma);
            obj.pi_0 = (obj.u - obj.v) / (obj.w - obj.v);
            obj.pi_1 = (obj.w - obj.u) / (obj.w - obj.v);
        end  

        function poly_eval = eval_poly(obj)         
            % POLY_EVAL evaluates the polynomial.
            % INPUT:
                % obj: current object
            % OUTPUT:
                % poly_eval: evaluated polynomial
            poly_int = polyint([obj.c_t, 0]); % anti-derivative of the polynomial
            poly_eval = polyval(poly_int, [0, 1]); % evaluate the polynomial
            poly_eval = diff(poly_eval); % difference of evaluated polynomial
        end
        
        function compute_valence(obj)       
            % COMPUTE_VALENCE computes the expected value of each action
            % based on belief states and contingency parameter. 
            % INPUT:
                % obj: current object
            % OUTPUT:
                % v_a_t: array with action valences
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
            % SOFTMAX returns choice probabilities based on the computed
            % action values and beta parameter 
            % INPUT:
                % obj: current object
            % OUTPUT:
                % p_a_t: vector with choice probabilities
            obj.p_a_t = exp(obj.v_a_t.*obj.beta) / sum(exp(obj.v_a_t.*obj.beta));
        end
             
       function int_voi = integrate_voi(obj,voi,varargin)
            % INTEGRATE_VOI returns the integral of a particular variable
            % conditional of contrast difference. 
            % INPUT:
                % obj: current object
                % voi: 0 is action values, voi = 1 is polynomial updates.
            % OUTPUT:
                % int_voi: array with polynomial updates
            obj.r_t = varargin{1}; % access the first varargin because the only additional input is r_t
            voi_matrix = NaN(length(obj.set_o), 2); % initialize voi matrix
            obj.p_o_giv_u = normpdf(obj.set_o, obj.o_t, obj.sigma); % observation probabilities
            obj.p_o_giv_u_norm = obj.p_o_giv_u / sum(obj.p_o_giv_u); % normalised probabilities
            for i = 1:length(obj.set_o) 
                obj.p_s_giv_o(obj.set_o(i)) % state given observations
                if voi == 0 % action values
                        obj.v_a_t = obj.compute_valence(); % action values
                        voi_matrix(i, 1) = obj.v_a_t(1); % value for action_0
                        voi_matrix(i, 2) = obj.v_a_t(2);
                elseif voi == 1 % polynomial updates
                    obj.q_0, obj.q_1 = obj.compute_q(obj.r_t, obj.pi_0, obj.pi_1);
                    voi_matrix(i, 1) = obj.q_0;
                    voi_matrix(i, 2) = obj.q_1;
                end
            end
                obj.q_0 = sum(voi_matrix(:, 1) * obj.p_o_giv_u_norm);
                obj.q_1 = sum(voi_matrix(:, 2) * obj.p_o_giv_u_norm); 
                int_voi = [obj.q_0, obj.q_1];
        end

        function decide_e(obj, o_t) 
            % DECIDE_E makes an economic decision based for the agent and 
            % uses softmax policy to make a choice for the agent
            % INPUT:
                % obj: current object
                % o_t: observed contrast difference
            % OUTPUT:
                % a_t: economic decision
            if obj.agent == 0 % random decision agent
                obj.v_a_t = [0.5, 0.5];
                obj.p_a_t = [0.5, 0.5];  
            elseif obj.agent == 1 % bayesian agent
                if obj.task_agent_analysis == 1 % compute action values by integrating over observations
                    voi = 0;
                    obj.v_a_t = obj.integrate_voi(voi); % multipled by contrast differences contingent observation
                else
                    obj.p_s_giv_o(o_t); % Compute belief state based on direct observations
                    obj.compute_valence(); % action valences based on direct observations
                end 
            end
            if obj.agent ~= 0 
                % for all agents except 0, use the softmax decision policy
                % for action selection.
                obj.softmax(); 
            end
        obj.a_t = binornd(1, obj.p_a_t(2)); % action
        end 
        
        function r_t = compute_action_dep_rew(obj, r_t) 
            % COMPUTE_ACTION_DEP_REW recodes task generated reward 
            % contingent on action. 
            % INPUT: 
                % obj: current object
                % r_t: task generated reward
            % OUTPUT:
                % r_t: agent recoded reward
            r_t = r_t + (obj.a_t *((-1) .^ (2 + r_t)));
        end
        
        function compute_q(obj,r_t)
            % COMPUTE_Q computes q_0 and q_1 for the polynomial update.
            % INPUT: 
                % obj: current object
                % r_t: task generated reward
            % OUTPUT:
                % q_0,q_1: fractions for polynomial update
            obj.t = length(obj.c_t) + 1; % degree of polynomial
            obj.r_t = obj.compute_action_dep_rew(r_t); % recode reward to get action contingent reward
            if obj.eval_ana == 1
                obj.G = obj.eval_poly();% valuated polynomials
                obj.C = (obj.pi_0 - obj.pi_1) * ((1 - obj.G) .^ (1 - obj.r_t)) * (obj.G .^ obj.r_t) + obj.pi_1; % common denominator
                obj.q_0 = (obj.pi_1 .^ obj.r_t) * (obj.pi_0 .^ (1-obj.r_t)) / obj.C; % fraction for polynomial updates
                obj.q_1 = ((-1) .^ (obj.r_t + 1)) * (obj.pi_0 - obj.pi_1) / obj.C;
                
            else
                obj.q_0_num = (obj.pi_1 ^ obj.r_t) * (obj.pi_0 ^ (1 - obj.r_t)); % numerical polynomial updates
                obj.q_1_num = ((-1) ^ (obj.r_t + 1)) * (obj.pi_0 - obj.pi_1);
                obj.product = obj.product * (obj.q_1_num * obj.p_mu + obj.q_0_num);
                obj.mu = obj.product; % updated mu
                obj.mu_for_ev = obj.mu / np.sum(obj.mu); % mu to calculate EV
                obj.q_0 = obj.q_0_num;
                obj.q_1 = obj.q_1_num ;
            end
        end

        function update_coefficients(obj)
            % UPDATE_COEFFICIENTS updates the prior probability using the
            % evaluated polynomials.
            % INPUT:
                % obj: current object
            % OUTPUT:
                % c_t: updated coefficient
            obj.d = zeros(1,obj.t);
            obj.d(end) = obj.q_0 * obj.c_t(end);  % last element
            for n = 0:(obj.t - 3)
                obj.d(end-(n+1)) = obj.q_1 * obj.c_t(end-n) + obj.q_0 * obj.c_t(end-(n+1)); % all elements except last and first
            end
            obj.d(1) = obj.q_1 * obj.c_t(1); % first element
            obj.c_t = obj.d;
        end

        function learn(obj,r_t)
            % LEARN updates the q-values to help the agent learn and
            % update contingency parameter.
            % INPUT:
                % obj: current object
                % r_t: task generated reward
            % OUTPUT:
                % G: updated contingency parameter
            if obj.agent == 0
                obj.G = 0.5;
            elseif obj.agent == 1
                if obj.task_agent_analysis == 1
                    % compute qs over contrast based observations
                    voi = 1 ;
                    q_s = obj.integrate_voi(voi, r_t);
                    obj.q_0 = q_s(1); 
                    obj.q_1 = q_s(2);
                    obj.update_coefficients(); % updating using q-values
                else
                    obj.p_s_giv_o(obj.o_t); % belief states
                    obj.compute_q(r_t); % compute q-values for updaTe
                    obj.update_coefficients(); % update coefficients   
                    obj.G = obj.eval_poly();
                end
            end
       end
   end
end  
  