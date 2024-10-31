classdef Task < taskvars
    % TASK class specifies task-specific instance variables for the simulation
    % of state dependent contrast differences and action dependent rewards. The
    % methods of this class generate a trial's state, contrast differences, and
    % rewards.

    properties
        p_cs  % initialize vector for p(c_t|s_t)
        p_ras = NaN(2, 2) % initialize vector for p^(a_t)(r_t|s, \mu)
        s_t  % sampled state of the trial
        c_t  % state-dependent sampled contrast difference
        r_t = NaN(1) % action-dependent sampled reward
        s_index = NaN(1) % s_index instead of s_t
        a_index = NaN(1) % a_index instead of a_t
    end

    methods

        function taskMu(obj)
            % function taskMu re-initialises the mu property according to
            % the experimental condition.
            %
            % INPUT:
            %   obj: current object
            
            if obj.condition == 1 || obj.condition == 3 % condition dependent contingency parameter
                obj.mu = 0.7;
            else
                obj.mu = 1;
            end
            obj.p_cs = NaN(length(obj.C), 2);
            obj.p_cs(:, 1) = (obj.C < 0) ./ sum(obj.C < 0);  % p(c_t|s_t = 0)
            obj.p_cs(:, 2) = (obj.C >= 0) ./ sum(obj.C >= 0);  % p(c_t|s_t = 1)
            obj.p_ras(1, 2) = 1 - obj.mu;  % p^(a_t = 0)(r_t = 1|s = 1, 1-\mu)
            obj.p_ras(2, 1) = 1 - obj.mu;  % p^(a_t = 1)(r_t = 1|s = 0, 1-\mu)
            obj.p_ras(1, 1) = obj.mu;  % p^(a_t = 0)(r_t = 1|s = 0, \mu)
            obj.p_ras(2, 2) = obj.mu;
        end

        function state_sample(obj)
            % function state_sample samples a trial's state (obj.s_t) 
            % from a binomial distribution and an index to generate rewards and contrast
            % difference.
            %
            % INPUT:
            %   obj: current object

            obj.s_t = binornd(1, obj.Theta); % sample state from binomial distribution
            obj.s_index = obj.s_t + 1; % s_index because s_t can't be used for indexing in MATLAB
        end

        function contrast_sample(obj)
            % function contrast_sample samples the trial's state dependent contrast
            % difference level (obj.c_t)
            %
            % INPUT:
            %   obj: current object

            p_cs_giv_s = obj.p_cs(:, obj.s_index);  % contrast differences conditional on state
            s_cs_giv_s = mnrnd(1, p_cs_giv_s, 1); % sample contrast difference
            i_cs_giv_s = find(s_cs_giv_s>0); % index of sampled contrast difference
            obj.c_t = obj.C(i_cs_giv_s);  % select contrast difference according to index
        end

        function reward_sample(obj,a_t)
            % function reward_sample samples the trial's action dependent reward 
            % (obj.r_t) for that trial.
            %
            % INPUT:
            %   obj: current object
            %   a_t: agent's action

            obj.s_index = obj.s_t + 1; % s_index because s_t can't be used for indexing in MATLAB
            obj.a_index = a_t + 1; % a_index because a_t can't be used for indexing in MATLAB
            obj.r_t = binornd(1, obj.p_ras(obj.a_index,obj.s_index)); % sample reward from binomial distribution
        end

    end
end
