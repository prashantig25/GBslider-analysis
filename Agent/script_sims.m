clc
clearvars

% INITIALIZE VARS
num_sims = 99; % number of simulations
num_blocks = 4; % number of blocks per simulation 
num_trials = 25; % number of trials in a block
contrast_vars = [1,1,0,0]; % value for a contrast in a condition
congruence_vars = [1,0,1,0]; % value for congruence in a condition
simulations_condition1 = []; % store the simulations
simulations_condition2 = []; % store the simulations
simulations_condition3 = []; % store the simulations
simulations_condition5 = []; % store the simulations
simulations_agent2 = []; % store the simulations for categorical agent
contrast = []; % store contrast level for each trial
congruence = []; % store congruence level for each trial


% Get the current working directory
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)'; % to which directory one must save in
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    disp('Current directory is already the desired path. No need to run createSavePaths.');
    desiredPath = currentDir;
else
    % Call the function to create the desired path
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = strcat(desiredPath, filesep, 'Data', filesep, 'agent simulations'); 
% mkdir(save_dir);

% RUN SIMULATIONS FOR NORMATIVE AGENT
condition = 1; % task condition
agent = 1; % agent type normative (1) or categorical (2)
[simulations_condition1] = run_simulations(num_sims,num_blocks,simulations_condition1,condition,agent); % use run simulations function
condition = repelem(condition,num_trials*num_blocks,1);
simulations_condition1.choice_cond = repmat(condition,num_sims,1);
condition = 2; % task condition
[simulations_condition2] = run_simulations(num_sims,num_blocks,simulations_condition2,condition,agent); % use run simulations function
condition = repelem(condition,num_trials*num_blocks,1);
simulations_condition2.choice_cond = repmat(condition,num_sims,1);
condition = 3; % task condition
[simulations_condition3] = run_simulations(num_sims,num_blocks,simulations_condition3,condition,agent); % use run simulations function
condition = repelem(condition,num_trials*num_blocks,1);
simulations_condition3.choice_cond = repmat(condition,num_sims,1);
condition = 5; % task condition
[simulations_condition5] = run_simulations(num_sims,num_blocks,simulations_condition5,condition,agent); % use run simulations function
for n = 1:num_blocks % add task-based variables 
    contrast = [contrast; repelem(contrast_vars(n),num_trials,1);];
    congruence = [congruence; repelem(congruence_vars(n),num_trials,1)];
end
simulations_condition1.contrast = repmat(contrast,num_sims,1);
simulations_condition1.congruence = repmat(congruence,num_sims,1);
simulations_condition2.contrast = repmat(contrast,num_sims,1);
simulations_condition2.congruence = repmat(congruence,num_sims,1);
simulations_condition3.contrast = repmat(contrast,num_sims,1);
simulations_condition3.congruence = repmat(congruence,num_sims,1);

% SIMULATE FOR CATEGORICAL AGETN
condition = 2; % task condition
agent = 2; % agent type normative (1) or categorical (2)
[simulations_agent2] = run_simulations(num_sims,num_blocks,simulations_agent2,condition,agent); % use run simulations function

safe_saveall(fullfile(save_dir,'data_agent_condition1.txt'),simulations_condition1);
safe_saveall(fullfile(save_dir,'data_agent_condition1_highPU.txt'),simulations_condition5);
safe_saveall(fullfile(save_dir,'data_agent_condition2.txt'),simulations_condition2);
safe_saveall(fullfile(save_dir,'data_agent_condition3.txt'),simulations_condition3);
safe_saveall(fullfile(save_dir,'agent2_learning.txt'),simulations_agent2);

% SAVE ALL CONDITION SIMULATIONS
agent_cond1 = readtable(fullfile(save_dir,"data_agent_condition1.txt"));
agent_cond2 = readtable(fullfile(save_dir,"data_agent_condition2.txt"));
agent_cond3 = readtable(fullfile(save_dir,"data_agent_condition3.txt"));
agent = [agent_cond1;agent_cond2;agent_cond3];
safe_saveall(fullfile(save_dir,"data_agent.txt"),agent);