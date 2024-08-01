%% Note
% make sure the behaviour data file has state,action,choice,contrast left,
% contrast right, contrast difference, condition, reward probability,
% contrast of the block.

% choice is the correct economic performance in the pilot study (study 1) files

% FOR study 2
% create condition column with 1,2,3
% create contrast column 0,1 

%% READ, MERGE and PREPROCESS all data files for pilot study
clc
clearvars 

% INITIALISE ALL REQUIRED VARS
t = 25; % number of trials in a block
num_cond = 4; % number of conditions in the task
num_cont = 2; % number of contrast levels in the task
num_blocks = 4; % number of blocks per condition
total_blocks = 16; % total number of blocks per subject

% Get the current working directory
currentDir = pwd;

% CHANGE DIRECTORY ACCORDINGLY
behv_dir = strcat('DATA', filesep, 'pilot_study'); % "DATA\main_study";
save_dir = strcat('saved_files',filesep,'pilot_study', filesep, 'study1'); %"saved_files\study2";
mkdir(save_dir);

% MERGE ALL SUBJECT DATA
subj_ids = [11:17,19:22,24,26:33,36,37,42,46,48,51,53,54,56,58,59,61,62,...
    65,68,69,74,76,83,85,86,87,89,95,96,99,100,109,110,113,114,115,...
    117	119	121	122	124	126	127	128	129	136	138	141	143	144	145	146	147,...
    148	149	150	151	153	154	156	157	159	161	162	163	164	165	167	171	172,...	
    176	177	180	182	183	188	191];
num_subjs = length(subj_ids);
data_all = []; % empty table to merge all subjects data
for i = 1:length(subj_ids)
    tsv_file = fullfile(behv_dir,strcat('sub_',num2str(subj_ids(i))),'behav', ...
        strcat('sub_',num2str(subj_ids(i)),".tsv")); % path and file name for TSV file
    data_table = readtable(tsv_file, "FileType","text",'Delimiter', '\t'); % read
    data_all = [data_all; data_table]; % merge all subjects data
end
%% CALCULATE ECONOMIC PERFORMANCE

% ACROSS CONDITIONS
ecoperf_cond = NaN(num_subjs,num_cond);

% MEAN ECONOMIC PERFORMANCE FOR EACH CONDITION
for i = 1:num_subjs
    for c = 1:num_cond
        ecoperf_cond(i,c) = mean(data_all.ecoperf(and(data_all.ID == subj_ids(i),data_all.condition_int == c)));
    end
end

% ACROSS CONTRAST
ecoperf_cont = NaN(num_subjs,num_cont);

% MEAN ECONOMIC PERFORMANCE FOR EACH CONTRAST
for i = 1:num_subjs
    for c = 1:num_cont
        ecoperf_cont(i,c) = mean(data_all.ecoperf(and(data_all.ID == subj_ids(i),data_all.contrast == c-1)));
    end
end

% ACROSS CONDITION AND CONTRAST
ecoperf_hh = NaN(num_subjs,num_cont);
ecoperf_hl = NaN(num_subjs,num_cont);
ecoperf_lh = NaN(num_subjs,num_cont);
ecoperf_ll = NaN(num_subjs,num_cont);

% MEANS FOR EACH CONDITION, ACROSS CONTRASTS
for i = 1:num_subjs
    data_subj = data_all(data_all.ID == subj_ids(i),:);
    for c = 1:num_cont
        ecoperf_hh(i,c) = nanmean(data_subj.ecoperf(and(data_subj.contrast == c-1,data_subj.condition_int == 1)));
        ecoperf_hl(i,c) = nanmean(data_subj.ecoperf(and(data_subj.contrast == c-1,data_subj.condition_int == 2)));
        ecoperf_lh(i,c) = nanmean(data_subj.ecoperf(and(data_subj.contrast == c-1,data_subj.condition_int == 3)));
        ecoperf_ll(i,c) = nanmean(data_subj.ecoperf(and(data_subj.contrast == c-1,data_subj.condition_int == 4)));
    end
end
ecoperf_cond_cont = [ecoperf_hh,ecoperf_hl,ecoperf_lh,ecoperf_ll];

%%

% INITIALISE ARRAYS FOR SINGLE TRIAL ECONOMIC PERFORMANCE
hh_curve = NaN(num_subjs,t);
hl_curve = NaN(num_subjs,t);
lh_curve = NaN(num_subjs,t);
ll_curve = NaN(num_subjs,t);

for i = [1:9,11:93] % exclude participant 21 because unbalanced blocks
    data_subj = data_all(data_all.ID==subj_ids(i),:);
    uni_mix = unique(data_subj.blocks(data_subj.condition_int==1));
    uni_perc = unique(data_subj.blocks(data_subj.condition_int==2));
    uni_rew = unique(data_subj.blocks(data_subj.condition_int==3));
    uni_no = unique(data_subj.blocks(data_subj.condition_int==4));
    hh_subj = NaN(num_cond,t);
    hl_subj = NaN(num_cond,t);
    lh_subj = NaN(num_cond,t);
    ll_subj = NaN(num_cond,t);
    for b = 1:num_cond
        hh_subj(b,:) = data_subj.ecoperf(and(data_subj.blocks == uni_mix(b),data_subj.condition_int == 1));
        hl_subj(b,:) = data_subj.ecoperf(and(data_subj.blocks == uni_perc(b),data_subj.condition_int == 2));
        lh_subj(b,:) = data_subj.ecoperf(and(data_subj.blocks == uni_rew(b),data_subj.condition_int == 3));
        ll_subj(b,:) = data_subj.ecoperf(and(data_subj.blocks == uni_no(b),data_subj.condition_int == 4));
    end
    hh_curve(i,:) = mean(hh_subj);
    hl_curve(i,:) = mean(hl_subj);
    lh_curve(i,:) = mean(lh_subj);
    ll_curve(i,:) = mean(ll_subj);
end

%% SAVE DATA

safe_saveall(fullfile(save_dir,"study1.txt"),data_all);

safe_saveall(fullfile(save_dir,'ecoperf_hh.mat'),ecoperf_hh); 
safe_saveall(fullfile(save_dir,'ecoperf_hl.mat'),ecoperf_hl);
safe_saveall(fullfile(save_dir,'ecoperf_lh.mat'),ecoperf_lh);
safe_saveall(fullfile(save_dir,'ecoperf_ll.mat'),ecoperf_ll);

safe_saveall(fullfile(save_dir,'hh_curve.mat'),hh_curve); 
safe_saveall(fullfile(save_dir,'hl_curve.mat'),hl_curve);
safe_saveall(fullfile(save_dir,'lh_curve.mat'),lh_curve);
safe_saveall(fullfile(save_dir,'ll_curve.mat'),ll_curve);