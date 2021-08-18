classdef taskvars % task variables
    properties (Constant)
        T = 25 % number of trials
        B = 4 % number of blocks
        Theta = 0.5 % the parameter
        mu % just initialised without attributing a value. Value has been set depending on the condition in the Task class
        experiment = 3 % ???
        condition = 1 % experimental condition i.e. HH = 1....LL = 4
        kappa % just initialised without attributing a value. Value has been set depending on the condition in the Task class
        min = 0.3 % absolute minimum contrast level for low PU conditions
    end
end