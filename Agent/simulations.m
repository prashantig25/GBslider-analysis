function  [sim_data] = simulations(varargin)
    % SIMULATIONS runs agent simulations for multiple blocks of the task.      
    % OUTPUT:
        % sim_data: table with simulated data

    % INITIALIZATION OF REQUIRED CLASSES
    task = Task();
    n_blocks = task.B; % number of blocks
    
    % PRE-ALLOCATE TABLE TO STORE SIMULATED DATA
    a = zeros(1,14);
    sim_data = array2table(a);
    sim_data.Properties.VariableNames = {'state' 'action' 'reward' 'state_0',...
        'state_1' 'mu' 'trials' 'correct' 'contrast_diff' 'subjective_obs',...
        'condition' 'contrast' 'value_a1' 'congruence'};

    % SIMULATE
    for blocks = 1:n_blocks
        data_int = task_agent_int();
        sim_data = [sim_data;data_int];
    end
clear agent_vars
clear task_vars
end
