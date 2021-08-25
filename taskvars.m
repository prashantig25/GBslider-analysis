classdef taskvars < handle % task variables
%{
This class specifies task-specific instance variables for the simulation of state dependent contrast differences and action dependent rewards. 
%}
    properties 
        T = 25 % number of trials
        B = 4 % number of blocks
        Theta = 0.5 % the parameter
        mu % just initialised without attributing a value. Value has been set depending on the condition in the Task class
        experiment = 3 % ???
        condition = 1 % experimental condition i.e. HH = 1....LL = 4
        kappa % just initialised without attributing a value. Value has been set depending on the condition in the Task class
        min = 0.3 % absolute minimum contrast level for low PU conditions
        kappa_min 
        kappa_max
        C   % range of discrete contrast differences
    end
    methods
        function obj = taskvars
        %{
        This function sets kappa_min and kappa_max contingent on the experimental condition.
        It also generates the range of contrast levels depending on kappa_min and kappa_max.
        It also sets the reward contingency parameter depending on the experimental condition.
        %}
            if obj.condition == 1 || obj.condition == 2 % condition dependent absolute maximum contrast levels 
                obj.kappa_max  = 0.08;
                obj.kappa_min = -0.08;
            else 
                obj.kappa_min = 0.30;
                obj.kappa_max = 0.38;
            end
            if obj.condition == 1 || obj.condition == 2 % condition dependent contrast level range
                obj.C = linspace(obj.kappa_min,obj.kappa_max,20);
            else
                obj.C = [linspace(-obj.kappa_max,-obj.kappa_min,10),linspace(obj.kappa_min,obj.kappa_max,10)];
            end
            if obj.condition == 1 || obj.condition == 3 % condition dependent reward contingency parameter
                obj.mu = 0.6;
            else
                obj.mu = 0.9;
            end
        end
    end
end
