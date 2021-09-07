classdef agentvars < handle
    properties
        set_o = linspace(-0.2,0.2,20) % set of observations
        sigma = 0.04 % sensitivity parameter
        beta = 100 % inverse temperature for softmax rule
        lambda % how to compute the RU based mixture parameter?
        lambda_control % standard mixture parameter that does not change with RU
        c0 = ones(1) % uniform prior distribution over mu
        kappa_min % minimum contrast level
        kappa_max % maximum contrast level
        agent = 2 % agent number
        eval_ana = 1  % evaluate agent analytically (1) or numerically (0)
        alpha = 0.1  % learning-rate parameter 
        task_agent_analysis = 0 % if this is true, integration over observations
        condition = 3 % experimental condition
    end 
    
    methods
        function obj = agentvars
%         This function sets kappa_min and kappa_max contingent on the
%         experimental condition. It also generates the range of contrast
%         levels depending on kappa_min and kappa_max. It also sets the
%         reward contingency parameter depending on the experimental
%         condition.
            if obj.condition == 1 || obj.condition == 2 % condition dependent absolute maximum contrast levels 
                obj.kappa_max  = 0.08;
                obj.kappa_min = -0.08;
            else 
                obj.kappa_min = 0.30;
                obj.kappa_max = 0.38;
            end
        end
    end
end
