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
simulations_condition6 = []; % store the simulations
simulations_condition7 = []; % store the simulations
simulations_condition9 = []; % store the simulations
simulations_agent2 = []; % store the simulations for categorical agent
contrast = []; % store contrast level for each trial
congruence = []; % store congruence level for each trial
task_agent_analysis = 1; % toggle passed to task_agent_int: 1 = observation
    % is the exact presented contrast difference (deterministic decisions);
    % 0 = observation is sampled with perceptual noise (stochastic simulation)


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
save_dir = strcat(desiredPath, filesep, 'Data', filesep, 'model fitting', filesep, 'agent simulations for mu range', filesep, 'full mu range'); 
mkdir(save_dir);

% RUN SIMULATIONS FOR NORMATIVE AGENT
condition = 2; % task condition
agent = 1; % agent type normative (1) or categorical (2)
[simulations_condition2] = run_simulations(num_sims,num_blocks,simulations_condition2,condition,agent,0,task_agent_analysis); % use run simulations function -- rangeMu=0: these are real task conditions, not a mu sweep
simulations_condition2.choice_cond = repmat(condition,num_sims*100,1);
condition = 6; % task condition
agent = 1; % agent type normative (1) or categorical (2)
[simulations_condition6] = run_simulations(num_sims,num_blocks,simulations_condition6,condition,agent,0,task_agent_analysis); % use run simulations function -- rangeMu=0: these are real task conditions, not a mu sweep
simulations_condition6.choice_cond = repmat(condition,num_sims*100,1);
condition = 7; % task condition
agent = 1; % agent type normative (1) or categorical (2)
[simulations_condition7] = run_simulations(num_sims,num_blocks,simulations_condition7,condition,agent,0,task_agent_analysis); % use run simulations function -- rangeMu=0: these are real task conditions, not a mu sweep
simulations_condition7.choice_cond = repmat(condition,num_sims*100,1);
condition = 1; % task condition
agent = 1; % agent type normative (1) or categorical (2)
[simulations_condition1] = run_simulations(num_sims,num_blocks,simulations_condition1,condition,agent,0,task_agent_analysis); % use run simulations function -- rangeMu=0: these are real task conditions, not a mu sweep
simulations_condition1.choice_cond = repmat(condition,num_sims*100,1);
condition = 9; % task condition
agent = 1; % agent type normative (1) or categorical (2)
[simulations_condition9] = run_simulations(num_sims,num_blocks,simulations_condition9,condition,agent,0,task_agent_analysis); % use run simulations function -- rangeMu=0: these are real task conditions, not a mu sweep
simulations_condition9.choice_cond = repmat(condition,num_sims*100,1);
for n = 1:num_blocks % add task-based variables 
    contrast = [contrast; repelem(contrast_vars(n),num_trials,1);];
    congruence = [congruence; repelem(congruence_vars(n),num_trials,1)];
end
simulations_condition1.contrast = repmat(contrast,num_sims,1);
simulations_condition1.congruence = repmat(congruence,num_sims,1);
simulations_condition2.contrast = repmat(contrast,num_sims,1);
simulations_condition2.congruence = repmat(congruence,num_sims,1);
simulations_condition6.contrast = repmat(contrast,num_sims,1);
simulations_condition6.congruence = repmat(congruence,num_sims,1);
simulations_condition7.contrast = repmat(contrast,num_sims,1);
simulations_condition7.congruence = repmat(congruence,num_sims,1);
simulations_condition9.contrast = repmat(contrast,num_sims,1);
simulations_condition9.congruence = repmat(congruence,num_sims,1);

safe_saveall(fullfile(save_dir,'data_agent_condition1.txt'),simulations_condition1);
safe_saveall(fullfile(save_dir,'data_agent_condition6.txt'),simulations_condition6);
safe_saveall(fullfile(save_dir,'data_agent_condition7.txt'),simulations_condition7);
safe_saveall(fullfile(save_dir,'data_agent_condition2.txt'),simulations_condition2);
safe_saveall(fullfile(save_dir,'data_agent_condition9.txt'),simulations_condition9);

% SAVE ALL CONDITION SIMULATIONS
agent_cond1 = readtable(fullfile(save_dir,"data_agent_condition1.txt"));
agent_cond6 = readtable(fullfile(save_dir,"data_agent_condition6.txt"));
agent_cond7 = readtable(fullfile(save_dir,"data_agent_condition7.txt"));
agent_cond2 = readtable(fullfile(save_dir,"data_agent_condition2.txt"));
agent_cond9 = readtable(fullfile(save_dir,"data_agent_condition9.txt"));
agent = [agent_cond1;agent_cond2;agent_cond7;agent_cond9;agent_cond6];
safe_saveall(fullfile(save_dir,"data_agent.txt"),agent);