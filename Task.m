classdef Task < handle
    properties
        T = taskvars.T
        Theta = taskvars.Theta
        kappa = taskvars.kappa
        mu = taskvars.mu
        experiment = taskvars.experiment
        C   % range of discrete contrast differences
        p_cs
        p_ras = NaN(2, 2)  % initialize vector for p^(a_t)(r_t|s, \mu)
        s_t = NaN(1)
        c_t = NaN(1)
        r_t = NaN(1)
        condition = taskvars.condition
        min = taskvars.min
        s_index = NaN(1) % s_index instead of s_t
        a_index = NaN(1) % a_index instead of a_t
    end
    methods
        function obj=Task
            if obj.condition == 1 || obj.condition == 2 % condition dependent absolute maximum contrast levels 
                obj.kappa = 0.08;
            else
                obj.kappa = 0.38;
            end
            if obj.condition == 1 || obj.condition == 2 % condition dependent contrast level range
                obj.C = linspace(-obj.kappa,obj.kappa,20);
            else
                obj.C = [linspace(-obj.kappa,-obj.min,10),linspace(obj.min,obj.kappa,10)];
            end
            if obj.condition == 1 || obj.condition == 3 % condition dependent reward contingency parameter
                obj.mu = 0.6;
            else
                obj.mu = 0.9;
            end
            obj.p_cs = NaN(length(obj.C), 2);  % initialize vector for p(c_t|s_t)
            obj.p_cs(:, 1) = (obj.C < 0) / sum(obj.C < 0);  % p(c_t|s_t = 0)
            obj.p_cs(:, 2) = (obj.C >= 0) / sum(obj.C >= 0);  % p(c_t|s_t = 1)    
            obj.p_ras(1, 2) = 1 - obj.mu;  % p^(a_t = 0)(r_t = 1|s = 1, 1-\mu)
            obj.p_ras(2, 1) = 1 - obj.mu;  % p^(a_t = 1)(r_t = 1|s = 0, 1-\mu)
            obj.p_ras(1, 1) = obj.mu;  % p^(a_t = 0)(r_t = 1|s = 0, \mu)
            obj.p_ras(2, 2) = obj.mu;
        end
        function state_sample(obj)
            obj.s_t = binornd(1, obj.Theta)      
            obj.s_index = obj.s_t + 1 % s_index because s_t can't be used for indexing in MATLAB
        end
        function contrast_sample(obj)
            obj.state_sample()
            if obj.experiment == 1 || obj.experiment == 3
                p_cs_giv_s = obj.p_cs(:, obj.s_index)  % contrast differences conditional on state
                s_cs_giv_s = mnrnd(1, p_cs_giv_s, 1) % sample contrast difference
                i_cs_giv_s = nonzeros(s_cs_giv_s)  % index of sampled contrast difference
                obj.c_t = obj.C(i_cs_giv_s)  % select contrast difference according to index
            else
                if obj.s_t == 0
                    obj.c_t = obj.C(1)  % most negative contrast difference
                else
                    obj.c_t = obj.C(2)  % most positive contrast difference
                end
            end
        end
        
        function reward_sample(obj)
            obj.random_action()
            obj.a_index = obj.a_t + 1 % a_index because a_t can't be used for indexing in MATLAB
            obj.contrast_sample()
            obj.r_t = binornd(1, obj.p_ras(obj.a_index, obj.s_index))
        end
    end
end

%%
%Questions
% Can we convert state 0,1, to 1,2, because 0 indexing does not work in
% MATLAB or can we just substitute 1,2 wherever indexing is needed. 