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
rangeMu = 1; % simulate for a bunch of mus
condition = 0; % set as 0 for range of mus
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
save_dir = strcat(desiredPath, filesep, 'Data', filesep, 'model fitting', filesep, 'agent simulations for mu range', filesep, 'range mu'); 
mkdir(save_dir);

% RUN SIMULATIONS FOR NORMATIVE AGENT
agent = 1; % agent type normative (1) or categorical (2)
[simulations] = run_simulations(num_sims,num_blocks,simulations,condition,agent,rangeMu,task_agent_analysis); % use run simulations function
simulations.choice_cond = repmat(condition,num_sims*100,1);

for n = 1:num_blocks % add task-based variables 
    contrast = [contrast; repelem(contrast_vars(n),num_trials,1);];
    congruence = [congruence; repelem(congruence_vars(n),num_trials,1)];
end
simulations.contrast = repmat(contrast,num_sims,1);
simulations.congruence = repmat(congruence,num_sims,1);

safe_saveall(fullfile(save_dir,'data_agent_rangeMu.txt'),simulations);