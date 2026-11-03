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
    %   varargin{1}: sigma (optional, see task_agent_int's header --
    %                requires varargin{2} too)
    %   varargin{2}: beta (optional, see task_agent_int's header --
    %                requires varargin{1} too)
    %   varargin{3}: numTrials (optional, overrides the default 25-trial
    %                block length -- see task_agent_int's header; requires
    %                varargin{1}/varargin{2} too)
    %   varargin{4}: evalAna (optional, overrides the default analytical
    %                polynomial belief update -- see task_agent_int's
    %                header; requires varargin{1}/varargin{2}/varargin{3} too)
    %
    % OUTPUT:
    %   simulations: stored simulations

    for n = 1:num_sims % for number of simulations
        for nb = 1:num_blocks % for number of blocks within each simulation
            if length(varargin) >= 4
                multi_sims = task_agent_int(condition,agent,rangeMu,task_agent_analysis,varargin{1},varargin{2},varargin{3},varargin{4}); % run simulation
            elseif length(varargin) >= 3
                multi_sims = task_agent_int(condition,agent,rangeMu,task_agent_analysis,varargin{1},varargin{2},varargin{3}); % run simulation
            elseif length(varargin) >= 2
                multi_sims = task_agent_int(condition,agent,rangeMu,task_agent_analysis,varargin{1},varargin{2}); % run simulation
            else
                multi_sims = task_agent_int(condition,agent,rangeMu,task_agent_analysis); % run simulation
            end
            simulations = [multi_sims; simulations];
        end
    end
end