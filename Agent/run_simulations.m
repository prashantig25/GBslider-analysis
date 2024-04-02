function [simulations] = run_simulations(num_sims,num_blocks,simulations)
    %
    % run_simulations runs task-agent interaction simulations to get choice
    % and learning data.
    %
    % INPUTS:
    % num_sims: number of simulations required
    % num_blocks: number of blocks within each simulation
    % simulations: initialised array for simulations
    %
    % OUTPUT:
    % simulations: stored simulations
    %
    for n = 1:num_sims % for number of simulations
        for nb = 1:num_blocks % for number of blocks within each simulation
            multi_sims = task_agent_int; % run simulation
            simulations = [multi_sims; simulations];
        end
    end
end