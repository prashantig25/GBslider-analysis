classdef Task < handle
%{
This class specifies task-specific instance variables for the simulation of state dependent contrast differences and action dependent rewards. 
The methods of this call generate a trial's state, contrast differences, and rewards. 
%}
    properties
        T = taskvars.T % number of trials in a block
        Theta = taskvars.Theta % parameter for state generation
        kappa = taskvars.kappa % 
        mu = taskvars.mu % contingency parameter
        C   % range of discrete contrast differences
        p_cs  % initialize vector for p(c_t|s_t)
        p_ras = NaN(2, 2)  % initialize vector for p^(a_t)(r_t|s, \mu)
        s_t = NaN(1)
        c_t = NaN(1)
        r_t = NaN(1)
        condition = taskvars.condition % experimental condition i.e. 1 = HH...4 = LL
        min = taskvars.min
        kappa_min = taskvars.kappa_min
        kappa_max = taskvars.kappa_max
        s_index = NaN(1) % s_index instead of s_t
        a_index = NaN(1) % a_index instead of a_t
    end
    methods
        function obj=Task
        %{
        This function changes the value of p_cs, p_ras depending on the contrast level range and reward contingency parameter.
        %}
            obj.p_cs = NaN(length(obj.C), 2); 
            obj.p_cs(:, 1) = (obj.C < 0) / sum(obj.C < 0);  % p(c_t|s_t = 0)
            obj.p_cs(:, 2) = (obj.C >= 0) / sum(obj.C >= 0);  % p(c_t|s_t = 1)    
            obj.p_ras(1, 2) = 1 - obj.mu;  % p^(a_t = 0)(r_t = 1|s = 1, 1-\mu)
            obj.p_ras(2, 1) = 1 - obj.mu;  % p^(a_t = 1)(r_t = 1|s = 0, 1-\mu)
            obj.p_ras(1, 1) = obj.mu;  % p^(a_t = 0)(r_t = 1|s = 0, \mu)
            obj.p_ras(2, 2) = obj.mu;
        end
        function state_sample(obj)
        %{
        state_sample samples the trial's state randomly from a binomial distribution.
        %}
            obj.s_t = binornd(1, obj.Theta)      
            obj.s_index = obj.s_t + 1 % s_index because s_t can't be used for indexing in MATLAB
        end
        function contrast_sample(obj)
        %{
        contrast_sample samples the trial's state dependent contrast difference level.
        %}
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
        %{
        reward_sample samples the trial's action dependent reward for that trial.
        %}
            obj.random_action()
            obj.a_index = obj.a_t + 1 % a_index because a_t can't be used for indexing in MATLAB
            obj.contrast_sample()
            obj.r_t = binornd(1, obj.p_ras(obj.a_index, obj.s_index))
        end
    end
end


%Questions
% Can we convert state 0,1, to 1,2, because 0 indexing does not work in
% MATLAB or can we just substitute 1,2 wherever indexing is needed. 
