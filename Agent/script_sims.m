clc
clearvars

% INITIALIZE VARS
num_sims = 99; % number of simulations
num_blocks = 4; % number of blocks per simulation 
num_trials = 25; % number of trials in a block
contrast_vars = [1,1,0,0]; % value for a contrast in a condition
congruence_vars = [1,0,1,0]; % value for congruence in a condition
simulations = []; % store the simulations
contrast = []; % store contrast level for each trial
congruence = []; % store congruence level for each trial
condition = 1; % experimental condition; change accordingly in agentvars and taskvars

% Get the current working directory
currentDir = pwd;
save_dir = strcat("..", filesep, 'Data', filesep, 'agent_simulations'); 
mkdir(save_dir);

% RUN SIMULATIONS
[simulations] = run_simulations(num_sims,num_blocks,simulations); % use run simulations function
for n = 1:num_blocks % add task-based variables 
    contrast = [contrast; repelem(contrast_vars(n),num_trials,1);];
    congruence = [congruence; repelem(congruence_vars(n),num_trials,1)];
end
condition = repelem(condition,num_trials*num_blocks,1);
simulations.contrast = repmat(contrast,num_sims,1);
simulations.congruence = repmat(congruence,num_sims,1);
simulations.choice_cond = repmat(condition,num_sims,1);
safe_saveall(fullfile(save_dir,'data_agent_condition1.xlsx'),simulations);

% Todo: save all files

% SAVE ALL CONDITION SIMULATIONS
agent_cond1 = readtable(fullfile(save_dir,"data_agent_condition1.xlsx"));
agent_cond2 = readtable(fullfile(save_dir,"data_agent_condition2.xlsx"));
agent_cond3 = readtable(fullfile(save_dir,"data_agent_condition3.xlsx"));
agent = [agent_cond1;agent_cond2;agent_cond3];
safe_saveall(fullfile(save_dir,"data_agent.xlsx"),agent);