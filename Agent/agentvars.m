classdef agentvars < handle
% AGENTVARS class specifies and generates parameters for the agent. 
    properties
        set_o = linspace(-0.1,0.1,20) % set of observations
        sigma % sensitivity parameter 
        beta = 100 % inverse temperature for softmax rule
        lambda = 0.7 % standard mixture parameter that does not change with RU
        c0 = ones(1) % uniform prior distribution over mu
        kappa_max = 0.1; % maximum contrast level
        kappa_min = -0.1; % minimum contrast level
        agent = 1 % agent type
        eval_ana = 1  % evaluate agent analytically (1) or numerically (0)
        alpha = 0.1  % learning-rate parameter 
        task_agent_analysis = 0 % if this is true, integration over observations
        condition = 1 % experimental condition
        q_0_0 = 0.5 % for learning models
    end 
    methods
        function obj = agentvars
            %
            % The contructor methods initialises all other properties of
            % the class that are computed based on exisitng static properties of
            % the class.
            %
            if obj.condition == 1 || obj.condition == 2 % condition dependent sensitivity parameter for agent
                obj.sigma = 0.03;
            else 
                obj.sigma = 0.0001;
            end
        end
    end
end