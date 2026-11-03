classdef agentvars < handle
% AGENTVARS class specifies and generates parameters for the agent. 
    
    properties
        set_o = linspace(-0.25,0.25,20) % set of observations -- spans 2.5x
            % beyond kappa_max (matching GbAgentVars.py's own set_o/kappa
            % padding ratio, 0.2/0.08), rather than MATLAB's previous 1:1
            % match to kappa_max, which left no margin for an observation
            % near the contrast range's edge to fall outside the grid at
            % realistic sigma values (see conversation: this caused
            % "with integration" to overconfidently overshoot the true mu
            % at sigma >= ~0.06, since hypothetical far-off perceptions got
            % truncated out of integrate_voi's marginalization entirely).
        sigma % sensitivity parameter 
        beta = 100 % inverse temperature for softmax rule
        lambda = 0.7 % standard mixture parameter that does not change with RU
        c0 = ones(1) % uniform prior distribution over mu
        kappa_max = 0.1; % maximum contrast level
        kappa_min = -0.1; % minimum contrast level
        agent = 1 % agent type
        eval_ana = 1 % evaluate agent analytically (1) for simulations or numerically (0) for model fitting recovery
        alpha = 0.1  % learning-rate parameter 
        task_agent_analysis = 0 % if this is true, integration over observations
        condition = 2 % experimental condition
        q_0_0 = 0.5 % for learning models
    end 
    
    methods

        function obj = agentvars           
            % The contructor methods initialises all other properties of
            % the class that are computed based on exisitng static properties of
            % the class.
            
            if obj.condition == 1 || obj.condition == 2 % condition dependent sensitivity parameter for agent
                % obj.sigma = 0.001 + (0.05 - 0.001) * rand;
                obj.sigma = 0.06; %0.03;
            elseif obj.condition == 3 || obj.condition == 4
                obj.sigma = 0.0001;
            end
        end
        
    end
end