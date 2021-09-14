function [df] = simulations
    
    % parameters from the taskvars class
    task_vars = taskvars()
    agent_vars = agentvars()
    
    % number of trials and blocks
    n_trials = task_vars.T
    n_blocks = task_vars.B
    
    df_sims = DataFrame();
    
    for b = 1:n_blocks
        
        % Single block task-agent-interaction simulation
        data_sims = task_agent_int();
%         df_sims.data_sims = data_sims;
%         df = df_sims
        data_xls = sprintf('simulations.xlsx');
        writetable(data_sims,data_xls,'sheet',b);
    end 
end
