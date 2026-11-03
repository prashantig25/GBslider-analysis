clc
clearvars

% INITIALIZE VARS
num_sims = 300; % number of simulations
num_blocks = 4; % number of blocks per simulation
num_trials = 25; % number of trials in a block
task_agent_analysis = 1; % toggle passed to task_agent_int: 1 = observation
    % is the exact presented contrast difference (deterministic decisions);
    % 0 = observation is sampled with perceptual noise (stochastic simulation)
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
reqPath = 'GBslider-analysis'; % to which directory one must save in
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

%% RUN SIMULATIONS FOR NORMATIVE AGENT
condition = 4; % task condition
agent = 1; % agent type normative (1) or categorical (2)
[simulations_condition4] = run_simulations(num_sims,num_blocks,simulations_condition1,condition,agent,0,task_agent_analysis); % use run simulations function
condition = repelem(condition,num_trials*num_blocks,1);
simulations_condition4.choice_cond = repmat(condition,num_sims,1);

condition = 1; % task condition
agent = 1; % agent type normative (1) or categorical (2)
[simulations_condition1] = run_simulations(num_sims,num_blocks,simulations_condition1,condition,agent,0,task_agent_analysis); % use run simulations function
condition = repelem(condition,num_trials*num_blocks,1);
simulations_condition1.choice_cond = repmat(condition,num_sims,1);

condition = 2; % task condition
[simulations_condition2] = run_simulations(num_sims,num_blocks,simulations_condition2,condition,agent,0,task_agent_analysis); % use run simulations function
condition = repelem(condition,num_trials*num_blocks,1);
simulations_condition2.choice_cond = repmat(condition,num_sims,1);

condition = 3; % task condition
[simulations_condition3] = run_simulations(num_sims,num_blocks,simulations_condition3,condition,agent,0,task_agent_analysis); % use run simulations function
condition = repelem(condition,num_trials*num_blocks,1);
simulations_condition3.choice_cond = repmat(condition,num_sims,1);

condition = 5; % task condition
[simulations_condition5] = run_simulations(num_sims,num_blocks,simulations_condition5,condition,agent,0,task_agent_analysis); % use run simulations function
for n = 1:num_blocks % add task-based variables 
    contrast = [contrast; repelem(contrast_vars(n),num_trials,1);];
    congruence = [congruence; repelem(congruence_vars(n),num_trials,1)];
end
simulations_condition4.contrast = repmat(contrast,num_sims,1);
simulations_condition4.congruence = repmat(congruence,num_sims,1);
simulations_condition1.contrast = repmat(contrast,num_sims,1);
simulations_condition1.congruence = repmat(congruence,num_sims,1);
simulations_condition2.contrast = repmat(contrast,num_sims,1);
simulations_condition2.congruence = repmat(congruence,num_sims,1);
simulations_condition3.contrast = repmat(contrast,num_sims,1);
simulations_condition3.congruence = repmat(congruence,num_sims,1);

% SIMULATE FOR CATEGORICAL AGETN
condition = 5; % task condition
agent = 2; % agent type normative (1) or categorical (2)
[simulations_agent2] = run_simulations(num_sims,num_blocks,simulations_agent2,condition,agent,0,task_agent_analysis); % use run simulations function

safe_saveall(fullfile(save_dir,'data_agent_condition1_pupil0.06.txt'),simulations_condition1);
safe_saveall(fullfile(save_dir,'data_agent_condition1_highPU.txt'),simulations_condition5);
safe_saveall(fullfile(save_dir,'data_agent_condition2_pupil0.06.txt'),simulations_condition2);
safe_saveall(fullfile(save_dir,'data_agent_condition3.txt'),simulations_condition3);
safe_saveall(fullfile(save_dir,'data_agent_condition4.txt'),simulations_condition4);
safe_saveall(fullfile(save_dir,'agent2_learning.txt'),simulations_agent2);

% SAVE ALL CONDITION SIMULATIONS
agent_cond1 = readtable(fullfile(save_dir,"data_agent_condition1_pupil0.06.txt"));
agent_cond2 = readtable(fullfile(save_dir,"data_agent_condition2_pupil0.06.txt"));
agent_cond3 = readtable(fullfile(save_dir,"data_agent_condition3.txt"));
agent = [agent_cond1;agent_cond2;];

safe_saveall(fullfile(save_dir,"data_agentpupil0.06.txt"),agent);

%% SIMULATE CONDITIONS 1 AND 2 WITH TASK_AGENT_ANALYSIS = 0 (stochastic simulation)
task_agent_analysis = 0; % see toggle definition above: 0 = observation
    % is sampled with perceptual noise and decisions are drawn
    % probabilistically (a genuine stochastic simulation)
simulations_condition1_stochastic = []; % store the simulations
simulations_condition2_stochastic = []; % store the simulations

% Rebuilt fresh here (rather than reusing contrast/congruence from the
% first simulation block above) so this section's row counts can't drift
% out of sync if it's ever run on its own without re-running the whole
% script from the top.
contrast_stochastic = [];
congruence_stochastic = [];
for n = 1:num_blocks
    contrast_stochastic = [contrast_stochastic; repelem(contrast_vars(n),num_trials,1)];
    congruence_stochastic = [congruence_stochastic; repelem(congruence_vars(n),num_trials,1)];
end

condition = 1; % task condition
agent = 1; % agent type normative (1) or categorical (2)
[simulations_condition1_stochastic] = run_simulations(num_sims,num_blocks,simulations_condition1_stochastic,condition,agent,0,task_agent_analysis); % use run simulations function
condition = repelem(condition,num_trials*num_blocks,1);
simulations_condition1_stochastic.choice_cond = repmat(condition,num_sims,1);
simulations_condition1_stochastic.contrast = repmat(contrast_stochastic,num_sims,1);
simulations_condition1_stochastic.congruence = repmat(congruence_stochastic,num_sims,1);

condition = 2; % task condition
agent = 1; % agent type normative (1) or categorical (2)
[simulations_condition2_stochastic] = run_simulations(num_sims,num_blocks,simulations_condition2_stochastic,condition,agent,0,task_agent_analysis); % use run simulations function
condition = repelem(condition,num_trials*num_blocks,1);
simulations_condition2_stochastic.choice_cond = repmat(condition,num_sims,1);
simulations_condition2_stochastic.contrast = repmat(contrast_stochastic,num_sims,1);
simulations_condition2_stochastic.congruence = repmat(congruence_stochastic,num_sims,1);

simulations_stochastic = [simulations_condition1_stochastic; simulations_condition2_stochastic];
safe_saveall(fullfile(save_dir,'data_agent_noInt.txt'),simulations_stochastic);

%% SIMULATE LONG BLOCKS ACROSS SIGMA VALUES, WITH AND WITHOUT INTEGRATION (EVAL_ANA = 0)
% One long, single-block (num_trials_long-trial) simulation per
% combination of sigma (perceptual sensitivity) x condition (true mu),
% run once WITH integration (task_agent_analysis = 1) and once WITHOUT
% (task_agent_analysis = 0), to compare belief (mu) and choice behavior
% between the two observation-handling mechanisms over a long horizon,
% independent of the short 25-trial block-reset structure used above.
% contrast/congruence are not meaningful for this single continuous long
% block (unlike the four distinct short blocks above), so they are left
% unset (NaN, task_agent_int.m's own default) here.
% eval_ana_long = 0 here: the numerical (mu_for_ev-based) belief update --
% a fixed-size histogram reweighted each trial, not a growing polynomial,
% so it stays numerically stable over long blocks. See the eval_ana=1
% section below for the alternative and task_agent_int.m's header for the
% full rationale.
num_trials_long = 1000; % trials in the single long block
num_blocks_long = 1;    % one long block per simulation (not num_blocks's four short ones)
num_sims_long = 30;     % fewer than num_sims=300 above -- this sweep already multiplies sims by 5 sigmas x 2 conditions x 2 integration settings; raise if more precision is needed
sigma_values_long = [0.02, 0.04, 0.06, 0.08, 0.1];
conditions_long = [1, 2]; % condition 1 -> true mu 0.7, condition 2 -> true mu 1 (rangeMu = 0, same mapping as the rest of this script)
beta_long = 100; % Agent's own default (agentvars.m) -- only sigma is swept here, beta held fixed
agent_long = 1; % normative agent
eval_ana_long = 0;

for task_agent_analysis_long = [1]
    simulations_long = [];
    for sigma_long = sigma_values_long
        for condition_long = conditions_long
            sim_chunk = [];
            [sim_chunk] = run_simulations(num_sims_long,num_blocks_long,sim_chunk,condition_long,agent_long,0,task_agent_analysis_long,sigma_long,beta_long,num_trials_long,eval_ana_long); % use run simulations function -- num_trials_long overrides taskvars.m's default 25-trial block length, eval_ana_long routes around the unstable high-degree analytical polynomial path
            sim_chunk.sigma = repmat(sigma_long,height(sim_chunk),1);
            sim_chunk.choice_cond = repmat(condition_long,height(sim_chunk),1);
            simulations_long = [simulations_long; sim_chunk];
        end
    end
    if task_agent_analysis_long == 1
        safe_saveall(fullfile(save_dir,'data_agent_longBlocks_withInt_evalAna0_increasedRange.txt'),simulations_long);
    else
        safe_saveall(fullfile(save_dir,'data_agent_longBlocks_noInt_evalAna0_increasedRange.txt'),simulations_long);
    end
end

%% SIMULATE LONG BLOCKS ACROSS SIGMA VALUES, WITH AND WITHOUT INTEGRATION (EVAL_ANA = 1)
% Same as the eval_ana=0 section above, except eval_ana_long = 1 here:
% Agent's own default (agentvars.m's eval_ana=1) -- the analytical
% polynomial belief update. NOTE: this path's polynomial degree grows by
% 1 every trial with no cap, which empirically became numerically
% unstable somewhere between 50 and 75 trials in this exact simulation
% (see conversation) -- at large num_trials_long values this section is
% very likely to crash (Task/reward_sample indexing error) rather than
% complete. Kept as its own section (rather than a toggle) specifically
% so that crash doesn't block the eval_ana=0 section above from running
% and saving its own data.
num_trials_long = 40;
num_blocks_long = 1;
num_sims_long = 30;
sigma_values_long = [0.02, 0.04, 0.06, 0.08, 0.1];
conditions_long = [1, 2];
beta_long = 100;
agent_long = 1;
eval_ana_long = 1;

parfor task_agent_analysis_long = [0, 1]
    simulations_long = [];
    for sigma_long = sigma_values_long
        for condition_long = conditions_long
            sim_chunk = [];
            [sim_chunk] = run_simulations(num_sims_long,num_blocks_long,sim_chunk,condition_long,agent_long,0,task_agent_analysis_long,sigma_long,beta_long,num_trials_long,eval_ana_long); % use run simulations function
            sim_chunk.sigma = repmat(sigma_long,height(sim_chunk),1);
            sim_chunk.choice_cond = repmat(condition_long,height(sim_chunk),1);
            simulations_long = [simulations_long; sim_chunk];
        end
    end
    if task_agent_analysis_long == 1
        safe_saveall(fullfile(save_dir,'data_agent_longBlocks_withInt_evalAna1.txt'),simulations_long);
    else
        safe_saveall(fullfile(save_dir,'data_agent_longBlocks_noInt_evalAna1.txt'),simulations_long);
    end
end