clc
clearvars

% INITIALISE VARS
num_cols = 14;
condition = 1; % task condition for which the simulation needs to be run 
% (make sure to match this to the condition variable of agentvars and taskvars)
num_trials = 100; % number of trials per condition
contrast = [repelem(0,num_trials./2,1); repelem(1,num_trials./2,1)]; % variable for whether the high or the low contrast block is more rewarding
congruence = [repelem(0,num_trials./4,1); repelem(1,num_trials./4,1);repelem(0,num_trials./4,1); repelem(1,num_trials./4,1)]; % variable for congruent or incongruent blocks
filename_save = 'data_model_ag1_cond3.xlsx'; % filename to save simulated data for each condition
num_conditions = 3; % number of conditions

% SIMULATE CHOICES AND UPDATES USING NORMATIVE AGENT
[sim_data] = simulations(); % run simulation
sim_data.condition = repelem(condition,height(sim_data),1); % store condition variable
sim_data(sim_data.trials == 0,:) = []; % get rid of unwanted trials
writetable(sim_data,filename_save); % save file

% COMBINE ALL CONDITIONS
filename_save = 'data_model_ag1.xlsx';
data = [];
for i = 1:num_conditions
    filename = strcat('data_model_ag1_cond',int2str(i),'.xlsx');
    data_cond = readtable(filename);
    data_cond.congruence = repmat(congruence,99,1); % add congruence
    data_cond.contrast = repmat(contrast,99,1); % add contrast
    data = [data; data_cond];
end
writetable(data,filename_save); % save file