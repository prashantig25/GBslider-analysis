classdef taskvars < handle
% TASKVARS class specifies task-specific instance variables for the simulation
% of state dependent contrast differences and action dependent rewards.
    properties 
        T = 25 % number of trials
        B = 396 % number of blocks
        Theta = 0.5 % parameter for sampling the state of the trial
        mu % contingency parameter
        condition = 1 % experimental condition i.e. both = 1....none = 4
        C   % range of discrete contrast differences
        kappa_max = 0.1; % maximum contrast level
        kappa_min = -0.1; % minimum contrast level
    end
    methods
        function obj = taskvars
            %
            % The contructor methods initialises all other properties of
            % the class that are computed based on exisitng static properties of
            % the class.
            %
            obj.C = linspace(obj.kappa_min,obj.kappa_max,20); % generate contrast difference
            if obj.condition == 1 || obj.condition == 3 % condition dependent contingency parameter
                obj.mu = 0.7;
            else
                obj.mu = 1;
            end
        end
    end
end
