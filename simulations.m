function [df] = simulations
    % This function runs simulation of different agents.
    
    
    % task and agent parameters required for the simulations
    task_vars = taskvars();;
    agent_vars = agentvars()
    
    % number of trials and blocks
    n_trials = task_vars.T
    n_blocks = task_vars.B
    
    % Cycle over number of blocks
    for b = 1:n_blocks
        
        % Single block task-agent-interaction simulation
        data_sims = task_agent_int();
        
        % save dataframes in an excel sheet for different blocks
        data_xls = sprintf('simulations.xlsx');
        writetable(data_sims,data_xls,'sheet',b);
    end 
end
