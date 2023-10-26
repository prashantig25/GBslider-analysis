function  [sim_data] = simulations(varargin)
    % simulations runs agent simulations for multiple blocks of the task.

    % INPUT:
        % varargin: could use task-based variables from participant's files to
        % simulate agent choices and learning

    % OUTPUT:
        % sim_data: table with simulated data
    % INITIALIZATION OF REQUIRED CLASSES
    task = Task();

    n_blocks = task.B; % number of blocks
    
    % PRE-ALLOCATE TABLE TO STORE SIMULATED DATA
    a = zeros(1,14);
    sim_data = array2table(a);
    sim_data.Properties.VariableNames = {'state' 'action' 'reward' 'state_0'
        'state_1' 'mu' 'trial' 'correct' 'objective_obs' 'subjective_obs' 
        'condition' 'contrast' 'value_a1' 'congruence'};

    % SIMULATE
    for blocks = 1:n_blocks
        if nargin > 0 % use task based variables from participant's files
            rows = (varargin{1}.blocks == blocks);    
            index = rows == 1;
            expt_data = varargin{1}(index,:); % extracts data only for this block
            data_int = task_agent_int(expt_data);
        else % use Task() generated variables
            data_int = task_agent_int();
        end
        sim_data = [sim_data;data_int];
    end
clear agent_vars
clear task_vars
end
