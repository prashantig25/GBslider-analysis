classdef learning_agents < agentvars 
    properties
        phi_0 = NaN % unacquired salience
        phi_1 = NaN
        epsilon_m_0 = NaN % acquired salience for mackintosh model
        epsilon_m_1 = NaN
        epsilon_p_0 = NaN % acquired salience for pearce hall model 
        epsilon_p_1 = NaN
        r_t_0 = 1 % current reward for trial 1 % where are we using this?
        r_t_1 = 0
        pe_0 = 0.5 % prediction errors 
        pe_1 = 0.5
        o_t = NaN % current observation
        a_t = NaN % current economic decision
        r_t = NaN % current reward
        pi_0 = NaN% belief state in favor of s_t = 0
        pi_1 = NaN  % belief state in favor of s_t = 1
        v_a_0 = NaN % valence of action = 0
        v_a_1 = NaN % valence of action = 1
        v_a_t = NaN % vector of valences
        p_a_t = NaN % vector with choice probabilities
        E_mu_t = NaN % % expected value
        G = NaN % Gamma factor
    end
    methods
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
        function observation_sample(obj,c_t) 
            % This function computes the contrast difference dependent
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
        function unacquired_salience(obj) 
            % This function computes the unacquired salience i.e.
            % perceptual salience based on belief states. Depending on the
            % BS of the agent, it assigns unacquired salience to both
            % alternatives. 
            obj.p_s_giv_o(obj.o_t)
            if obj.pi_0 >= obj.pi_1
                obj.phi_1 = obj.pi_0
                obj.phi_0 = 1 - obj.phi_1
            else
                obj.phi_0 = obj.pi_1
                obj.phi_1 = 1 - obj.phi_0  
            end
        end
        function [q_0_1, q_1_0, q_1_1] = get_q_s_a(obj) 
%             This function computes q-values q_0_1, q_1_0 and q_1_1 based
%             on q_0_0
%             That is, expected value for alternatives under different
%             states and actions. 
%             Will be used to for PE calculation for the Pearce-Hall models.
            q_0_1 = 1-obj.q_0_0
            q_1_0 = 1-obj.q_0_0
            q_1_1 = obj.q_0_0
        end      
        function [capital_q_0, capital_q_1] = compute_capital_q(obj, q_1_0, q_0_1, q_1_1, o_t)
            
%         This function computes Q-values Q_0 and Q_1 for the P/H model. 
%         q_0_0: State 0, action 1
%         q_1_0: State 1, action 0
%         q_0_1: State 0, action 1
%         q_1_1: State 1, action 1
%         pi_0: Belief over state 0
%         pi_1: Belief over state 1
        obj.p_s_giv_o(o_t)
        capital_q_0 = obj.q_0_0 * obj.pi_0 + q_1_0 * obj.pi_1
        capital_q_1 = q_0_1 * obj.pi_0 + q_1_1 * obj.pi_1
%         capital_q_a = [capital_q_0, capital_q_1]
        end
        function r_t = compute_action_dep_rew(obj, r_t) 
            % This function generates the reward contingent on action. 
            obj.r_t = r_t + (obj.a_t * ((-1) * (2 + r_t)))
        end
        function prediction_errors(obj, o_t,r_t) 
            % This function computes prediction errors based on current
            % reward and the expected Q-values of a choice.
            
            [q_0_1, q_1_0, q_1_1] = obj.get_q_s_a()
            [capital_q_0, capital_q_1] = obj.compute_capital_q(q_1_0, q_0_1, q_1_1, o_t)
            r_t = obj.compute_action_dep_rew(r_t)
            if obj.a_t == 0 
                obj.pe_0 = r_t - capital_q_0
                obj.pe_1 = (1-r_t) - capital_q_1
            elseif obj.a_t == 1
                obj.pe_1 = r_t - capital_q_1
                obj.pe_0 = (1-r_t) - capital_q_0
            end
        end
        function acquired_salience(obj,o_t,r_t)
            
            % This function calculates the acquired salience of a choice
            % based on its predictability of reward. The exact computation
            % of salience depends on the agent. 
            obj.prediction_errors(o_t,r_t)
            if obj.agent == 5 
                % attentional associability
                % Mackintosh based salience computation
                % How to compute this is not explicitly in the Mackintosh
                % model. The rule is epsilon_m_o > 0, if PE_0 < PE_1.
                obj.epsilon_m_0 
                obj.epsilon_m_1 
            elseif obj.agent == 6 || obj.agent == 7 
                % salience associability
                % Pearce Hall based salience computation
                % acquired salience is to be calculated based on the PEs
                obj.epsilon_p_0 = obj.pe_0 
                obj.epsilon_p_1 = obj.pe_1
            end
        end
        function softmax(obj)
            % This function guides action selection based on the softmax rule
            % uses the computed action values and beta parameter
            
            obj.p_a_t = exp(obj.v_a_t*obj.beta) / sum(exp(obj.v_a_t*obj.beta)) 
        end
        function v_a_t_ag_1 = compute_valence(obj)
            
            % This function computes the expected value of each action
            % based on belief states and contingency parameter. For agent 5
            % and 6, the action value is a sum of acquired and unacquired
            % salience. 
                obj.v_a_0 = obj.phi_0 + obj.epsilon_p_0
                obj.v_a_1 = obj.phi_1 + obj.epsilon_p_1  
        %Concatenate action valences
        obj.v_a_t = [obj.v_a_0, obj.v_a_1]
        v_a_t_ag_1 = obj.v_a_t
        end
        
        function decide_e(obj, o_t, r_t) 
            % This function makes an economic decision based on the agent
            % by adding the acquired and unacquired salience to compute valence. 
            % finally, uses softmax policy to make a choice for the agent
            obj.unacquired_salience()
            obj.prediction_errors(o_t,r_t)
            obj.acquired_salience(o_t,r_t)
            obj.compute_valence()
            if obj.agent ~= 0 
                % for all agents except 0, use the softmax decision policy
                % for action selection.
                obj.softmax() 
            end
        obj.a_t = binornd(1, obj.p_a_t(2)) % action
        end
        function compute_q_0_0(obj)
%         This function computes q_0_0 based on the salience associability
%         and learning rate weighted prediction errors.
           
            if obj.pi_0 >= obj.pi_1
                obj.q_0_0 = obj.q_0_0 + obj.pi_0 * obj.alpha * obj.epsilon_p_0 * (obj.r_t - obj.q_0_0)
            else
                obj.q_0_0 = obj.q_0_0 + obj.pi_1 * obj.alpha * obj.epsilon_p_0 *((1 - obj.r_t) - obj.q_0_0)
            end
        end
        function learn(obj,r_t)
            % This function updates the q-values to help the agent learn
            % from the reward (r_t). calculate PE weight it with LR,
            % epsilon value i.e. the acquired salience
            % updates the mu parameter. 
            obj.p_s_giv_o(obj.o_t)
            obj.r_t = obj.compute_action_dep_rew(r_t)
            obj.compute_q_0_0()
            obj.E_mu_t = obj.q_0_0
            obj.G = obj.E_mu_t
            end  
      end
end
