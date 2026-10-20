function [simulations] = run_simulations(num_sims,num_blocks,simulations,condition,agent,rangeMu,task_agent_analysis,varargin)
    % function run_simulations runs task-agent interaction simulations to get choice
    % and learning data.
    %
    % INPUTS:
    %   num_sims: number of simulations required
    %   num_blocks: number of blocks within each simulation
    %   simulations: initialised array for simulations
    %   condition: task condition for which we simulate data
    %   agent: agent type
    %   rangeMu: whether to sweep mu across a range (1) or use the
    %            condition-fixed mu (0)
    %   task_agent_analysis: passed straight through to task_agent_int --
    %       see its header for what 0 vs 1 means
    %
    % OUTPUT:
    %   simulations: stored simulations

    for n = 1:num_sims % for number of simulations
        for nb = 1:num_blocks % for number of blocks within each simulation
            if nargin > 7
                multi_sims = task_agent_int(condition,agent,rangeMu,task_agent_analysis,varargin{1},varargin{2}); % run simulation
            else
                multi_sims = task_agent_int(condition,agent,rangeMu,task_agent_analysis); % run simulation
            end
            simulations = [multi_sims; simulations];
        end
    end
end